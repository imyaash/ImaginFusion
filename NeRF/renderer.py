import os
import cv2
import math
import torch
import mcubes
import raymarching
import numpy as np
import torch.nn as nn
from utils.mesh import meshDecimator, meshCleaner
from utils.functions import customMeshGrid, safeNormalise

class Renderer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.bound = args.bound
        self.cascade = 1 + math.ceil(math.log2(args.bound))
        self.gridSize = 128
        self.densityT = args.densityThresh
        aabb_train = torch.FloatTensor([-args.bound, -args.bound, -args.bound, args.bound, args.bound, args.bound])
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
    def densityBlob(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        
        if self.args.densityActivation == 'exp':
            return self.args.blobDensity * torch.exp(- d / (2 * self.args.blobRadius ** 2))
        else:
            return self.args.blobDensity * (1 - torch.sqrt(d) / self.args.blobRadius)
    
    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def resetExtraState(self):
        self.density_grid.zero_()
        self.meanDensity = 0
        self.iterDensity = 0

    @torch.no_grad()
    def exportMesh(self, path, resolution = None, decimateT = -1, S = 128):
        if resolution is None:
            resolution = self.gridSize
        densityT = min(self.meanDensity, self.densityT) \
                if np.greater(self.meanDensity, 0) \
                    else self.densityT
        if self.args.densityActivation == 'softplus':
            densityT = densityT * 25
        sigmas = np.zeros(
            [
                resolution,
                resolution,
                resolution
            ], dtype=np.float32
        )
        X = torch.linspace(-1, 1, resolution).split(S)
        Y = torch.linspace(-1, 1, resolution).split(S)
        Z = torch.linspace(-1, 1, resolution).split(S)
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                for k, z in enumerate(Z):
                    xx, yy, zz = customMeshGrid(x, y, z)
                    pts = torch.cat(
                        [
                            xx.reshape(-1, 1),
                            yy.reshape(-1, 1),
                            zz.reshape(-1, 1)
                        ], dim=-1
                    )
                    val = self.density(pts.to(self.aabb_train.device))
                    sigmas[
                        i * S: i * S + len(x), j * S: j * S + len(y), k * S: k * S + len(z)
                    ] = val['sigma'].reshape(len(x), len(y), len(z)).detach().cpu().numpy()
        print(f"Marching Cubes: {densityT} ({sigmas.min()} ~ {sigmas.max()})")
        vertices, triangles = mcubes.marching_cubes(sigmas, densityT)
        vertices = vertices / (resolution - 1.0) * 2 - 1
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
        vertices, triangles = meshCleaner(
            vertices, triangles, remesh = True, remeshSize = 0.01

        )
        if decimateT > 0 and triangles.shape[0] > decimateT:
            vertices, triangles = meshDecimator(vertices, triangles, decimateT)
        v = torch.from_numpy(vertices).contiguous().float().to(self.aabb_train.device)
        f = torch.from_numpy(triangles).contiguous().int().to(self.aabb_train.device)

        def exporter(v, f, h0 = 2048, w0 = 2048, ssaa = 1, name = ""):
            device = v.device
            vnp = v.cpu().numpy()
            fnp = f.cpu().numpy()
            print(f"Unwrapping Mesh with xAtlas: v = {vnp.shape} f = {fnp.shape}")

            import xatlas
            import nvdiffrast.torch as dr
            from sklearn.neighbors import NearestNeighbors as KNN
            from scipy.ndimage import binary_dilation as dilation, binary_erosion as erosion

            atlas = xatlas.Atlas()
            atlas.add_mesh(vnp, fnp)
            chartOptions = xatlas.ChartOptions()
            chartOptions.max_iterations = 4
            atlas.generate(chart_options = chartOptions)
            vMapping, ftnp, vtnp = atlas[0]
            vt = torch.from_numpy(vtnp.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ftnp.astype(np.int64)).int().to(device)
            uv = vt * 2.0 - 1.0
            uv = torch.cat(
                (
                uv, torch.zeros_like(uv[..., :1]),
                torch.ones_like(uv[..., :1])
                ), dim=-1
            )
            h, w = (int(h0 * ssaa), int(w0 * ssaa)) if ssaa > 1 else (h0, w0)
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
            feats = torch.zeros(h * w, 3, device = device, dtype = torch.float32)
            if mask.any():
                xyzs = xyzs[mask]
                allFeats = []
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    results_ = self.density(xyzs[head:tail])
                    allFeats.append(results_['albedo'].float())
                    head += 640000
                feats[mask] = torch.cat(allFeats, dim = 0)
            feats = feats.view(h, w, -1)
            mask = mask.view(h, w)
            feats = feats.cpu().numpy()
            feats = (feats * 255).astype(np.uint8)
            mask = mask.cpu().numpy()
            inpaintRegion = dilation(mask, iterations = 3)
            inpaintRegion[mask] = 0
            searchRegion = mask.copy()
            noSearchRegion = erosion(searchRegion, iterations = 2)
            searchRegion[noSearchRegion] = 0
            searchCoords = np.stack(np.nonzero(searchRegion), axis = -1)
            inpaintCoords = np.stack(np.nonzero(inpaintRegion), axis = -1)
            knn = KNN(n_neighbors = 1, algorithm = 'kd_tree').fit(searchCoords)
            _, indices = knn.kneighbors(inpaintCoords)
            feats[tuple(inpaintCoords.T)] = feats[tuple(searchCoords[indices[:, 0]].T)]
            feats = cv2.cvtColor(feats, cv2.COLOR_RGB2BGR)
            if ssaa > 1:
                feats = cv2.resize(feats, (w0, h0), interpolation = cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(path, f'{name}Albedo.png'), feats)
            objFile = os.path.join(path, f'{name}Mesh.obj')
            mtlFile = os.path.join(path, f'{name}Mesh.mtl')
            print(f"Writing Mesh (.onj) to {objFile}")
            with open(objFile, "w") as fp:
                fp.write(f'mtllib {name}Mesh.mtl \n')
                print(f"Writing Vertices {vnp.shape}")
                for v in vnp:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
                print(f"Writing Vertices Texture Coordinates {vtnp.shape}")
                for v in vtnp:
                    fp.write(f'vt {v[0]} {1 - v[1]} \n')
                print(f"Writing Faces {fnp.shape}")
                fp.write(f'usemtl mat0 \n')
                for i in range(len(fnp)):
                    fp.write(f"f {fnp[i, 0] + 1}/{ftnp[i, 0] + 1} {fnp[i, 1] + 1}/{ftnp[i, 1] + 1} {fnp[i, 2] + 1}/{ftnp[i, 2] + 1} \n")
            with open(mtlFile, "w") as fp:
                fp.write(f'newmtl mat0 \n')
                fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
                fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
                fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
                fp.write(f'Tr 1.000000 \n')
                fp.write(f'illum 1 \n')
                fp.write(f'Ns 0.000000 \n')
                fp.write(f'map_Kd {name}Albedo.png \n')

        exporter(v, f)

    def run(
            self, raysO, raysD, lightD = None,
            ambientRatio = 1.0, shading = 'albedo',
            bgColor = None, perturb = False,
            tThresh = 1e-4, binarise = False, **test
    ):
        pfx = raysO.shape[:-1]
        raysO = raysO.contiguous().view(-1, 3)
        raysD = raysD.contiguous().view(-1, 3)
        N = raysO.shape[0]
        device = raysO.device
        nears, fars = raymarching.near_far_from_aabb(
            raysO, raysD, self.aabb_train \
                if self.training \
                    else self.aabb_infer
        )
        if lightD is None:
            lightD = safeNormalise(
                raysO + torch.randn(3, device=raysO.device)
            )
        results = {}
        if self.training:
            xyzs, dirs, ts, rays = raymarching.march_rays_train(
                raysO, raysD, self.bound, self.density_bitfield,
                self.cascade, self.gridSize, nears, fars, perturb,
                self.args.dtGamma, self.args.maxSteps
            )
            dirs = safeNormalise(dirs)
            if lightD.shape[0] > 1:
                flattenRays = raymarching.flatten_rays(rays, xyzs.shape[0]).long()
                lightD = lightD[flattenRays]
            sigmas, rgbs, normals = self(
                xyzs, dirs, lightD, ratio = ambientRatio, shading = shading
            )
            weights, weightsSum, depth, image = raymarching.composite_rays_train(
                sigmas, rgbs, ts, rays, tThresh, binarise
            )
            if self.args.lambdaOrient > 0 and normals is not None:
                lossOrient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = lossOrient.mean()
            if self.args.lambda3dNormalSmooth > 0 and normals is not None:
                normalsPerturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                results['loss_normal_perturb'] = (normals - normalsPerturb).abs().mean()
            if (self.args.lambda2dNormalSmooth > 0 or self.args.lambdaNormal > 0) and normals is not None:
                _, _, _, normalImage = raymarching.composite_rays_train(
                    sigmas.detach(), (normals + 1) / 2, ts, rays, tThresh, binarise
                )
                results['normal_image'] = normalImage
            results['weights'] = weights
        else:
            dtype = torch.float32
            weightsSum = torch.zeros(N, dtype = dtype, device = device)
            depth = torch.zeros(N, dtype = dtype, device = device)
            image = torch.zeros(N, 3, dtype = dtype, device = device)
            nAlive = N
            raysAlive = torch.arange(nAlive, dtype = torch.int32, device = device)
            raysT = nears.clone()
            step = 0
            while step < self.args.maxSteps:
                nAlive = raysAlive.shape[0]
                if nAlive <= 0:
                    break
                nStep = max(min(N // nAlive, 8), 1)
                xyzs, dirs, ts = raymarching.march_rays(
                    nAlive, nStep, raysAlive, raysT, raysO, raysD, self.bound,
                    self.density_bitfield, self.cascade, self.gridSize, nears, fars,
                    perturb if step == 0 else False, self.args.dtGamma, self.args.maxSteps
                )
                dirs = safeNormalise(dirs)
                sigmas, rgbs, normals = self(
                    xyzs, dirs, lightD, ratio = ambientRatio, shading = shading
                )
                raymarching.composite_rays(
                    nAlive, nStep, raysAlive, raysT, sigmas, rgbs, ts, weightsSum, depth, image, tThresh, binarise
                )
                raysAlive = raysAlive[raysAlive >= 0]
                step += nStep
        if bgColor is None:
            if self.args.bgRadius > 0:
                bgColor = self.background(raysD)
            else:
                bgColor = 1
        image = image + (1 - weightsSum).unsqueeze(-1) * bgColor
        image = image.view(*pfx, 3)
        depth = depth.view(*pfx)
        weightsSum = weightsSum.reshape(*pfx)
        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weightsSum
        return results

    @torch.no_grad()
    def updateExtraState(self, decay=0.95, S=128):
        tempGrid = -torch.ones_like(self.density_grid)
        X = torch.arange(
            self.gridSize, dtype = torch.int32, device = self.aabb_train.device
        ).split(S)
        Y = torch.arange(
            self.gridSize, dtype = torch.int32, device = self.aabb_train.device
        ).split(S)
        Z = torch.arange(
            self.gridSize, dtype = torch.int32, device = self.aabb_train.device
        ).split(S)
        for x in X:
            for y in Y:
                for z in Z:
                    xx, yy, zz = customMeshGrid(x, y, z)
                    coords = torch.cat(
                        [
                            xx.reshape(-1, 1),
                            yy.reshape(-1, 1),
                            zz.reshape(-1, 1)
                        ], dim=-1
                    )
                    indices = raymarching.morton3D(coords).long()
                    xyzs = 2 * coords.float() / (self.gridSize - 1) - 1
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        halfGridSize = bound / self.gridSize
                        casXyzs = xyzs * (bound - halfGridSize)
                        casXyzs += (torch.rand_like(casXyzs) * 2 - 1) * halfGridSize
                        sigmas = self.density(casXyzs)['sigma'].reshape(-1).detach()
                        tempGrid[cas, indices] = sigmas
        validMask = self.density_grid >= 0
        self.density_grid[validMask] = torch.maximum(self.density_grid[validMask] * decay, tempGrid[validMask])
        self.meanDensity = torch.mean(self.density_grid[validMask]).item()
        self.iterDensity += 1
        densityT = min(self.meanDensity, self.densityT)
        self.density_bitfield = raymarching.packbits(self.density_grid, densityT, self.density_bitfield)

    def render(self, raysO, raysD, **kwargs):
        return self.run(raysO, raysD, **kwargs)