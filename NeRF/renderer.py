import os
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import mcubes
import raymarching
from utils.functions import customMeshGrid, safeNormalise
from utils.mesh import meshCleaner, meshDecimator

class Renderer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bound = args.bound
        self.cascade = 1 + math.ceil(math.log2(args.bound))
        self.gridSize = 128
        self.densityThresh = args.densityThresh
        aabbTrain = torch.FloatTensor([-args.bound, -args.bound, -args.bound, args.bound, args.bound, args.bound])
        aabbInfer = aabbTrain.clone()
        self.register_buffer('aabbTrain', aabbTrain)
        self.register_buffer('aabbInfer', aabbInfer)
        self.glctx = None
        densityGrid = torch.zeros([self.cascade, self.gridSize ** 3]) # [CAS, H * H * H]
        densityBitfield = torch.zeros(self.cascade * self.gridSize ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
        self.register_buffer('densityGrid', densityGrid)
        self.register_buffer('densityBitfield', densityBitfield)
        self.mean_density = 0
        self.iter_density = 0
    
    @torch.no_grad()
    def densityBlob(self, x):
        d = (x ** 2).sum(-1)
        y = torch.exp(-d / (2 * self.args.blobRadius ** 2)) if self.args.densityActivation == "exp" else (1 - torch.sqrt(d) / self.args.blobRadius)
        return self.args.blobDensity * y
    
    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def reset_extra_state(self):
        self.densityGrid.zero_()
        self.mean_density = 0
        self.iter_density = 0

    @torch.no_grad()
    def exportMesh(self, path, resolution=None, decimate_target=-1, S=128):        
        if resolution is None:
            resolution = self.gridSize
        densityThresh = min(self.mean_density, self.densityThresh) \
            if np.greater(self.mean_density, 0) else self.densityThresh
        if self.args.densityActivation == 'softplus':
            densityThresh = densityThresh * 25
        sigmas = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        X = torch.linspace(-1, 1, resolution).split(S)
        Y = torch.linspace(-1, 1, resolution).split(S)
        Z = torch.linspace(-1, 1, resolution).split(S)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = customMeshGrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = self.density(pts.to(self.aabbTrain.device))
                    sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val['sigma'].reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
        print(f'[INFO] marching cubes thresh: {densityThresh} ({sigmas.min()} ~ {sigmas.max()})')
        vertices, triangles = mcubes.marching_cubes(sigmas, densityThresh)
        vertices = vertices / (resolution - 1.0) * 2 - 1
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
        vertices, triangles = meshCleaner(vertices, triangles, remesh=True, remeshSize=0.01)
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = meshDecimator(vertices, triangles, decimate_target)
        v = torch.from_numpy(vertices).contiguous().float().to(self.aabbTrain.device)
        f = torch.from_numpy(triangles).contiguous().int().to(self.aabbTrain.device)

        def export(v, f, h0=2048, w0=2048, ssaa=1, name=''):
            device = v.device
            v_np = v.cpu().numpy()
            f_np = f.cpu().numpy()
            print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

            import xatlas
            import nvdiffrast.torch as dr
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            _, ft_np, vt_np = atlas[0]
            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
            uv = vt * 2.0 - 1.0
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)
            if ssaa > 1:
                h = int(h0 * ssaa)
                w = int(w0 * ssaa)
            else:
                h, w = h0, w0
            if self.glctx is None:
                if h <= 2048 and w <= 2048:
                    self.glctx = dr.RasterizeCudaContext()
                else:
                    self.glctx = dr.RasterizeGLContext()
            rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), ft, (h, w))
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)
            mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)
            feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
            if mask.any():
                xyzs = xyzs[mask]
                all_feats = []
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    results_ = self.density(xyzs[head:tail])
                    all_feats.append(results_['albedo'].float())
                    head += 640000
                feats[mask] = torch.cat(all_feats, dim=0) 
            feats = feats.view(h, w, -1)
            mask = mask.view(h, w)
            feats = feats.cpu().numpy()
            feats = (feats * 255).astype(np.uint8)
            mask = mask.cpu().numpy()
            inpaint_region = binary_dilation(mask, iterations=3)
            inpaint_region[mask] = 0
            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=2)
            search_region[not_search_region] = 0
            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)
            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)
            feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]
            feats = cv2.cvtColor(feats, cv2.COLOR_RGB2BGR)
            if ssaa > 1:
                feats = cv2.resize(feats, (w0, h0), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(path, f'{name}albedo.png'), feats)
            obj_file = os.path.join(path, f'{name}mesh.obj')
            mtl_file = os.path.join(path, f'{name}mesh.mtl')
            print(f'[INFO] writing obj mesh to {obj_file}')
            with open(obj_file, "w") as fp:
                fp.write(f'mtllib {name}mesh.mtl \n')
                print(f'[INFO] writing vertices {v_np.shape}')
                for v in v_np:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
                print(f'[INFO] writing vertices texture coords {vt_np.shape}')
                for v in vt_np:
                    fp.write(f'vt {v[0]} {1 - v[1]} \n') 
                print(f'[INFO] writing faces {f_np.shape}')
                fp.write(f'usemtl mat0 \n')
                for i in range(len(f_np)):
                    fp.write(f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")
            with open(mtl_file, "w") as fp:
                fp.write(f'newmtl mat0 \n')
                fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
                fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
                fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
                fp.write(f'Tr 1.000000 \n')
                fp.write(f'illum 1 \n')
                fp.write(f'Ns 0.000000 \n')
                fp.write(f'map_Kd {name}albedo.png \n')
        export(v, f)

    def run(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, T_thresh=1e-4, binarize=False, **kwargs):
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        N = rays_o.shape[0]
        device = rays_o.device
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabbTrain if self.training else self.aabbInfer)
        if light_d is None:
            light_d = safeNormalise(rays_o + torch.randn(3, device=rays_o.device))
        results = {}
        if self.training:
            xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.densityBitfield, self.cascade, self.gridSize, nears, fars, perturb, self.args.dtGamma, self.args.maxSteps)
            dirs = safeNormalise(dirs)
            if light_d.shape[0] > 1:
                flatten_rays = raymarching.flatten_rays(rays, xyzs.shape[0]).long()
                light_d = light_d[flatten_rays]
            sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ts, rays, T_thresh, binarize)
            if self.args.lambdaOrient > 0 and normals is not None:
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()
            if self.args.lambda3dNormalSmooth > 0 and normals is not None:
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                results['loss_normal_perturb'] = (normals - normals_perturb).abs().mean()
            if (self.args.lambda2dNormalSmooth > 0 or self.args.lambdaNormal > 0) and normals is not None:
                _, _, _, normal_image = raymarching.composite_rays_train(sigmas.detach(), (normals + 1) / 2, ts, rays, T_thresh, binarize)
                results['normal_image'] = normal_image
            results['weights'] = weights
        else:
            dtype = torch.float32
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device)
            rays_t = nears.clone()
            step = 0
            while step < self.args.maxSteps:
                n_alive = rays_alive.shape[0]
                if n_alive <= 0:
                    break
                n_step = max(min(N // n_alive, 8), 1)
                xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.densityBitfield, self.cascade, self.gridSize, nears, fars, perturb if step == 0 else False, self.args.dtGamma, self.args.maxSteps)
                dirs = safeNormalise(dirs)
                sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, T_thresh, binarize)
                rays_alive = rays_alive[rays_alive >= 0]
                step += n_step
        if bg_color is None:
            bg_color = self.background(rays_d) if self.args.bgRadius > 0 else 1
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        weights_sum = weights_sum.reshape(*prefix)
        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        return results

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        tmp_grid = - torch.ones_like(self.densityGrid)
        X = torch.arange(self.gridSize, dtype=torch.int32, device=self.aabbTrain.device).split(S)
        Y = torch.arange(self.gridSize, dtype=torch.int32, device=self.aabbTrain.device).split(S)
        Z = torch.arange(self.gridSize, dtype=torch.int32, device=self.aabbTrain.device).split(S)
        for xs in X:
            for ys in Y:
                for zs in Z:
                    xx, yy, zz = customMeshGrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    xyzs = 2 * coords.float() / (self.gridSize - 1) - 1 # [N, 3] in [-1, 1]
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_gridSize = bound / self.gridSize
                        cas_xyzs = xyzs * (bound - half_gridSize)
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_gridSize
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                        tmp_grid[cas, indices] = sigmas
        valid_mask = self.densityGrid >= 0
        self.densityGrid[valid_mask] = torch.maximum(self.densityGrid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.densityGrid[valid_mask]).item()
        self.iter_density += 1

        densityThresh = min(self.mean_density, self.densityThresh)
        self.densityBitfield = raymarching.packbits(self.densityGrid, densityThresh, self.densityBitfield)

    def render(self, rays_o, rays_d, mvp, h, w, staged=False, max_ray_batch=4096, **kwargs):
        return self.run(rays_o, rays_d, **kwargs)