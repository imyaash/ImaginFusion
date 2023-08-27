import os
import math
import cv2
import numpy as np

import torch
import torch.nn as nn


import mcubes
import raymarching
from utils.mesh import meshDecimator, meshCleaner
from utils.functions import customMeshGrid, safeNormalise

class NeRFRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.bound = opt.bound
        self.cascade = 1 + math.ceil(math.log2(opt.bound))
        self.gridSize = 128
        self.densityT = opt.densityThresh
        aabb_train = torch.FloatTensor([-opt.bound, -opt.bound, -opt.bound, opt.bound, opt.bound, opt.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)
        self.glctx = None
        density_grid = torch.zeros([self.cascade, self.gridSize ** 3]) # [CAS, H * H * H]
        density_bitfield = torch.zeros(self.cascade * self.gridSize ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
        self.register_buffer('density_grid', density_grid)
        self.register_buffer('density_bitfield', density_bitfield)
        self.meanDensity = 0
        self.iterDensity = 0
    
    @torch.no_grad()
    def density_blob(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        
        if self.opt.densityActivation == 'exp':
            g = self.opt.blobDensity * torch.exp(- d / (2 * self.opt.blobRadius ** 2))
        else:
            g = self.opt.blobDensity * (1 - torch.sqrt(d) / self.opt.blobRadius)

        return g
    
    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def reset_extra_state(self):
        # density grid
        self.density_grid.zero_()
        self.meanDensity = 0
        self.iterDensity = 0

    @torch.no_grad()
    def export_mesh(self, path, resolution=None, decimate_target=-1, S=128):

        if resolution is None:
            resolution = self.gridSize

        density_thresh = min(self.meanDensity, self.densityT) \
            if np.greater(self.meanDensity, 0) else self.densityT
        
        # TODO: use a larger thresh to extract a surface mesh from the density field, but this value is very empirical...
        if self.opt.densityActivation == 'softplus':
            density_thresh = density_thresh * 25
        
        sigmas = np.zeros([resolution, resolution, resolution], dtype=np.float32)

        # query
        X = torch.linspace(-1, 1, resolution).split(S)
        Y = torch.linspace(-1, 1, resolution).split(S)
        Z = torch.linspace(-1, 1, resolution).split(S)

        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = customMeshGrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = self.density(pts.to(self.aabb_train.device))
                    sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val['sigma'].reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]

        print(f'[INFO] marching cubes thresh: {density_thresh} ({sigmas.min()} ~ {sigmas.max()})')

        vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)
        vertices = vertices / (resolution - 1.0) * 2 - 1

        # clean
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
        vertices, triangles = meshCleaner(vertices, triangles, remesh=True, remeshSize=0.01)
        
        # decimation
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = meshDecimator(vertices, triangles, decimate_target)

        v = torch.from_numpy(vertices).contiguous().float().to(self.aabb_train.device)
        f = torch.from_numpy(triangles).contiguous().int().to(self.aabb_train.device)

        # mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        # mesh.export(os.path.join(path, f'mesh.ply'))

        def _export(v, f, h0=2048, w0=2048, ssaa=1, name=''):
            # v, f: torch Tensor
            device = v.device
            v_np = v.cpu().numpy() # [N, 3]
            f_np = f.cpu().numpy() # [M, 3]

            print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

            # unwrap uvs
            import xatlas
            import nvdiffrast.torch as dr
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4 # for faster unwrap...
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]

            # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

            # render uv maps
            uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

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

            rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, h, w, 3]
            mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]

            # masked query 
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)
            
            feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

            if mask.any():
                xyzs = xyzs[mask] # [M, 3]

                # batched inference to avoid OOM
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

            # quantize [0.0, 1.0] to [0, 255]
            feats = feats.cpu().numpy()
            feats = (feats * 255).astype(np.uint8)

            ### NN search as an antialiasing ...
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

            # do ssaa after the NN search, in numpy
            if ssaa > 1:
                feats = cv2.resize(feats, (w0, h0), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(os.path.join(path, f'{name}albedo.png'), feats)

            # save obj (v, vt, f /)
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

        _export(v, f)

    def run_cuda(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, T_thresh=1e-4, binarize=False, **kwargs):
        # rays_o, rays_d: [B, N, 3]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = safeNormalise(rays_o + torch.randn(3, device=rays_o.device)) # [N, 3]

        results = {}

        if self.training:
            xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.gridSize, nears, fars, perturb, self.opt.dtGamma, self.opt.maxSteps)
            dirs = safeNormalise(dirs)

            if light_d.shape[0] > 1:
                flatten_rays = raymarching.flatten_rays(rays, xyzs.shape[0]).long()
                light_d = light_d[flatten_rays]
            
            sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ts, rays, T_thresh, binarize)
            
            # normals related regularizations
            if self.opt.lambdaOrient > 0 and normals is not None:
                # orientation loss 
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()
            
            if self.opt.lambda3dNormalSmooth > 0 and normals is not None:
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                results['loss_normal_perturb'] = (normals - normals_perturb).abs().mean()
            
            if (self.opt.lambda2dNormalSmooth > 0 or self.opt.lambdaNormal > 0) and normals is not None:
                _, _, _, normal_image = raymarching.composite_rays_train(sigmas.detach(), (normals + 1) / 2, ts, rays, T_thresh, binarize)
                results['normal_image'] = normal_image
            
            # weights normalization
            results['weights'] = weights

        else:
           
            # allocate outputs 
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < self.opt.maxSteps: # hard coded max step

                # count alive rays 
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.gridSize, nears, fars, perturb if step == 0 else False, self.opt.dtGamma, self.opt.maxSteps)
                dirs = safeNormalise(dirs)
                sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, T_thresh, binarize)

                rays_alive = rays_alive[rays_alive >= 0]
                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

        # mix background color
        if bg_color is None:
            if self.opt.bgRadius > 0:
                # use the bg model to calculate bg_color
                bg_color = self.background(rays_d) # [N, 3]
            else:
                bg_color = 1

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
        # call before each epoch to update extra states.

        
        ### update density grid
        tmp_grid = - torch.ones_like(self.density_grid)
        
        X = torch.arange(self.gridSize, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Y = torch.arange(self.gridSize, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Z = torch.arange(self.gridSize, dtype=torch.int32, device=self.aabb_train.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = customMeshGrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    xyzs = 2 * coords.float() / (self.gridSize - 1) - 1 # [N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.gridSize
                        # scale to current cascade's resolution
                        cas_xyzs = xyzs * (bound - half_grid_size)
                        # add noise in [-hgs, hgs]
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                        # query density
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                        # assign 
                        tmp_grid[cas, indices] = sigmas
        # ema update
        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.meanDensity = torch.mean(self.density_grid[valid_mask]).item()
        self.iterDensity += 1

        # convert to bitfield
        density_thresh = min(self.meanDensity, self.densityT)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.meanDensity:.4f}, occ_rate={(self.density_grid > density_thresh).sum() / (128**3 * self.cascade):.3f}')


    def render(self, rays_o, rays_d, mvp, h, w, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3]
        # return: pred_rgb: [B, N, 3]
        B, N = rays_o.shape[:2]
        device = rays_o.device

        results = self.run_cuda(rays_o, rays_d, **kwargs)

        return results