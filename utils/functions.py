import os
import torch
import psutil
import random
import numpy as np
import torch.nn.functional as F
from packaging import version as pver

def getViewDirections(thetas, phis, overhead, front):
    res = torch.zeros(thetas.shape[0], dtype = torch.long)
    phis = phis % (2 * np.pi)
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res

def circlePoses(
        device, radius = torch.tensor([3.2]), theta = torch.tensor([60]), phi = torch.tensor([0]),
        returnDirs = False, angleOverhead = 30, angleFront = 60
):
    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angleOverhead = angleOverhead / 180 * np.pi
    angleFront = angleFront / 180 * np.pi
    centres = torch.stack(
        [
            radius * torch.sin(theta) * torch.sin(phi),
            radius * torch.cos(theta),
            radius * torch.sin(theta) * torch.cos(phi)
        ], dim = -1
    )
    forwardVector = safeNormalise(centres)
    upVector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centres), 1)
    rightVector = safeNormalise(torch.cross(forwardVector, upVector, dim = -1))
    upVector = safeNormalise(torch.cross(rightVector, forwardVector, dim = -1))
    poses = torch.eye(4, dtype = torch.float, device = device).unsqueeze(0).repeat(len(centres), 1, 1)
    poses[:, :3, :3] = torch.stack((rightVector, upVector, forwardVector), dim = -1)
    poses[:, :3, 3] = centres
    dirs = getViewDirections(theta, phi, angleOverhead, angleFront) if returnDirs else None
    return poses, dirs

def randomPoses(
        size, device, args,
        radiusRange = [1, 1.5],
        thetaRange = [0, 120],
        phiRange = [0, 360],
        returnDirs = False,
        angleOverhead = 30,
        angleFront = 60,
        uniformSphereRate = 0.5
):
    thetaRange = np.array(thetaRange) / 180 * np.pi
    phiRange = np.array(phiRange) / 180 * np.pi
    angleOverhead = angleOverhead / 180 * np.pi
    angleFront = angleFront / 180 * np.pi
    radius = torch.rand(
        size, device = device
    ) * (
        radiusRange[1] - radiusRange[0]
    ) + radiusRange[0]
    if random.random() < uniformSphereRate:
        unitCentres = F.normalize(
            torch.stack(
            [
                torch.randn(size, device = device),
                torch.abs(torch.randn(size, device = device)),
                torch.randn(size, device = device)
            ], dim = -1
            ), p = 2, dim = 1
        )
        thetas = torch.acos(unitCentres[:, 1])
        phis = torch.atan2(unitCentres[:, 0], unitCentres[:, 2])
        phis[phis < 0] += 2 * np.pi
        centres = unitCentres * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(
            size, device = device
        ) * (
            thetaRange[1] - thetaRange[0]
        ) + thetaRange[0]
        phis = torch.rand(
            size, device = device
        ) * (
            phiRange[1] - phiRange[0]
        ) + phiRange[0]
        phis[phis < 0] += 2 * np.pi
        centres = torch.stack(
            [
                radius * torch.sin(thetas) * torch.sin(phis),
                radius * torch.cos(thetas),
                radius * torch.sin(thetas) * torch.cos(phis)
            ], dim = -1
        )
    targets = 0
    if args.jitterPose:
        jitCentre = args.jitterCentre
        jitTarget = args.jitterTarget
        centres += torch.rand_like(centres) * jitCentre - jitCentre / 2.0
        targets += torch.randn_like(centres) * jitTarget
    forwardVector = safeNormalise(centres - targets)
    upVector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    rightVector = safeNormalise(torch.cross(forwardVector, upVector, dim = -1))
    upNoise = torch.randn_like(upVector) * args.jitterUp if args.jitterPose else 0
    upVector = safeNormalise(torch.cross(
        rightVector, forwardVector, dim = -1
    ) + upNoise)
    poses = torch.eye(
        4, dtype = torch.float, device = device
    ).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack(
        (rightVector, upVector, forwardVector), dim = -1
    )
    poses[:, :3, 3] = centres
    dirs = getViewDirections(thetas, phis, angleOverhead, angleFront) if returnDirs else None
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180
    return poses, dirs, thetas, phis, radius

def customMeshGrid(*args):
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing = "ij")

def safeNormalise(x, eps = 1e-20):
    y = x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim = True), min = eps))
    return y

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
            results['indsCoarse'] = indsCoarse
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
    results['raysO'] = raysO
    results['raysD'] = raysD
    return results

def seeder(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

@torch.jit.script
def linear2srgb(x):
    srgb = torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)
    return srgb

@torch.jit.script
def srgb2linear(x):
    linear = torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return linear

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
