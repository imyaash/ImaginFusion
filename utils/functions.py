import os
import torch
import random
import psutil
import numpy as np
import torch.nn.functional as F
from packaging import version as pver

def getViewDirection(thetas, phis, oHead, front):
    res = torch.zeros(thetas.shape[0], dtype = torch.long)
    phis = phis % (2 * np.pi)
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
    res[thetas <= oHead] = 4
    res[thetas >= (np.pi - oHead)] = 5
    return res

def customMeshGrid(*args):
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing = "ij")

def normalise(x, eps = 1e-20):
    return x / torch.sqrt(
        torch.clamp(
            torch.sum(
                x * x, -1, keepdim = True
            ), min = eps
        )
    )

@torch.cuda.amp.autocast(enabled = False)
def getRays(poses, intrinsics, H, W, N = -1, errorMap = None):
    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics
    i, j = customMeshGrid(torch.linspace(0, W - 1, W, device = device), torch.linspace(0, H - 1, H, device = device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    results = {}
    if N > 0:
        N = min(N, H * W)
        if errorMap is None:
            inds = torch.randint(0, H * W, size = [N], device = device)
            inds = inds.expand([B, N])
        else:
            indsCoarse = torch.multinomial(errorMap.to(device), N, replacement = False)
            indsX, indsY = indsCoarse // 128, indsCoarse % 128
            sx, sy = H / 128, W / 128
            indsX = (indsX * sx + torch.rand(B, N, device=device) * sx).long().clamp(max = H - 1)
            indsY = (indsY * sy + torch.rand(B, N, device=device) * sy).long().clamp(max = W - 1)
            inds = indsX * W + indsY
            results['inds_coarse'] = indsCoarse
        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)
        results['inds'] = inds
    else:
        inds = torch.arange(H * W, device = device).expand([B, H * W])
    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim = -1)
    raysD = directions @ poses[:, :3, :3].transpose(-1, -2)
    raysO = poses[..., :3, 3]
    raysO = raysO[..., None, :].expand_as(raysD)
    results['rays_o'] = raysO
    results['rays_d'] = raysD
    return results

def seeder(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def getCPUMem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3

def getGPUMem():
    num = torch.cuda.device_count()
    mem, mems = 0, []
    for i in range(num):
        memFree, memTotal = torch.cuda.mem_get_info(i)
        mems.append(int(((memTotal - memFree) / 1024 ** 3) * 1000) / 1000)
        mem += mems[-1]
    return mem, mems

def circlePoses(
        device,
        radius = torch.tensor([3.2]),
        theta = torch.tensor([60]),
        phi = torch.tensor([0]),
        returnDirs=False,
        angleOverhead=30,
        angleFront=60
):
    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angleOverhead = angleOverhead / 180 * np.pi
    angleFront = angleFront / 180 * np.pi
    centres = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.cos(theta),
        radius * torch.sin(theta) * torch.cos(phi),
    ], dim = -1)
    fVector = normalise(centres)
    uVector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centres), 1)
    rVector = normalise(torch.cross(fVector, uVector, dim = -1))
    uVector = normalise(torch.cross(rVector, fVector, dim = -1))
    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(len(centres), 1, 1)
    poses[:, :3, :3] = torch.stack((rVector, uVector, fVector), dim = -1)
    poses[:, :3, 3] = centres
    if returnDirs:
        dirs = getViewDirection(theta, phi, angleOverhead, angleFront)
    else:
        dirs = None
    return poses, dirs

def randPoses(
        size,
        device,
        args,
        radRange = None,
        thetaRange = None,
        phiRange = None,
        returnDirs = False,
        angleOverhead = 30,
        angleFront = 60,
        uniSphRate = 0.5
):
    if radRange is None:
        radRange = [1, 1.5]
    if thetaRange is None:
        thetaRange = [0, 120]
    if phiRange is None:
        phiRange = [0, 360]
    thetaRange = np.array(thetaRange) / 180 * np.pi
    phiRange = np.array(phiRange) / 180 * np.pi
    angleOverhead = angleOverhead / 180 * np.pi
    angleFront = angleFront / 180 * np.pi
    radius = torch.rand(size, device = device) * (radRange[1] - radRange[0]) + radRange[0]
    if random.random() < uniSphRate:
        unitCentre = F.normalize(
            torch.stack([
                torch.randn(size, device = device),
                torch.abs(torch.randn(size, device = device)),
                torch.randn(size, device = device),
            ], dim = -1), p = 2, dim = 1
        )
        thetas = torch.acos(unitCentre[:,1])
        phis = torch.atan2(unitCentre[:,0], unitCentre[:,2])
        phis[phis < 0] += 2 * np.pi
        centres = unitCentre * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device = device) * (thetaRange[1] - thetaRange[0]) + thetaRange[0]
        phis = torch.rand(size, device = device) * (phiRange[1] - phiRange[0]) + phiRange[0]
        phis[phis < 0] += 2 * np.pi
        centres = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1)
    targets = 0
    if args.jitterPose:
        jitCentre = args.jitterCentre
        jitTarget = args.jitterTarget
        centres += torch.rand_like(centres) * jitCentre - jitCentre/2.0
        targets += torch.randn_like(centres) * jitTarget
    fVector = normalise(centres - targets)
    uVector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    rVector = normalise(torch.cross(fVector, uVector, dim = -1))
    uNoise = torch.randn_like(uVector) * args.jitterUp if args.jitterPose else 0
    uVector = normalise(torch.cross(rVector, fVector, dim = -1) + uNoise)
    poses = torch.eye(4, dtype = torch.float, device = device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((rVector, uVector, fVector), dim = -1)
    poses[:, :3, 3] = centres
    if returnDirs:
        dirs = getViewDirection(thetas, phis, angleOverhead, angleFront)
    else:
        dirs = None
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180
    return poses, dirs, thetas, phis, radius