import os
import torch
import random
import psutil
import numpy as np
import torch.nn.functional as F
from packaging import version as pver

def getViewDirection(thetas, phis, oHead, front):
    """
    Calculated the view direction based on the angles the thetas and the phis.

    Args:
        thetas (torch.Tensor): Tensor containing theta angles in radians.
        phis (torch.Tensor): Tensor containing phi angles in radians.
        oHead (float): Angle overhead threshold in radians.
        front (float): Angle front threshold in radians.

    Returns:
        torch.Tensor: A tensor of integers representing view directions.
    """
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
    """
    Create a mesh grid for given input tensors.

    Args:
        args: Input tensors for which the mesh grid should be created.

    Returns:
        tuple: A tuple of tensors representing the mesh grid.
    """
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing = "ij")

def normalise(x, eps = 1e-20):
    """
    Normalise a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        eps (float, optional): A small value to prevent division by zero. Defaults to 1e-20.

    Returns:
        torch.Tensor: normalised tensor.
    """
    return x / torch.sqrt(
        torch.clamp(
            torch.sum(
                x * x, -1, keepdim = True
            ), min = eps
        )
    )

@torch.cuda.amp.autocast(enabled = False)
def getRays(poses, intrinsics, H, W, N = -1, errorMap = None):
    """
    Generate rays based on camera poses and intrinsics.

    Args:
        poses (torch.Tensor): Camera poses.
        intrinsics (tuple): Camera intrinsics (fx, fy, cx, cy)
        H (int): Image height.
        W (int): image width.
        N (int, optional): Number of rays to generate. Defaults to -1, generates all rays.
        errorMap (torch.Tensor, optional): Error map for ray sampling. Defaults to None.

    Returns:
        dict: A dictionary containing ray information including origins, directions, and indices.
    """
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
    """
    Set random seed for Python, NumPy, and PyTorch.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def getCPUMem():
    """
    Get current CPU memory usage.

    Returns:
        float: Current CPU memory usage in GB.
    """
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3

def getGPUMem():
    """
    Get GPU memory usage.

    Returns:
        tuple: A tuple containing the total GPU memory and GPU memory usage for each available GPU.
    """
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
    """
    Generate circular camera poses.

    Args:
        device (str): PyTorch device.
        radius (torch.Tensor, optional): Radius of the circle. Defaults to torch.tensor([3.2]).
        theta (torch.Tensor, optional): Theta angles in degrees. Defaults to torch.tensor([60]).
        phi (torch.Tensor, optional): Phi angles in degrees. Defaults to torch.tensor([0]).
        returnDirs (bool, optional): Whether to return view directions. Defaults to False.
        angleOverhead (int, optional): Angle overhead threshold in degrees. Defaults to 30.
        angleFront (int, optional): Angle front threshold in degrees. Defaults to 60.

    Returns:
        tuple: A tuple containing camera poses and view directions (if returnDirs is True).
    """
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
    """
    Generate random camera poses.

    Args:
        size (int): Number of camera poses to generate.
        device (str): PyTorch device.
        args (object): Additional arguments.
        radRange (list, optional): Range for the radius. Defaults to None.
        thetaRange (list, optional): Range for theta angles in degrees. Defaults to None.
        phiRange (list, optional): Range for phi angles in degrees. Defaults to None.
        returnDirs (bool, optional): Whether to return view directions. Defaults to False.
        angleOverhead (int, optional): Angle overhead threshold in degrees. Defaults to 30.
        angleFront (int, optional): Angle front threshold in degrees. Defaults to 60.
        uniSphRate (float, optional): Rate of uniform spherical sampling. Defaults to 0.5.

    Returns:
        tuple: A tuple containing camera poses, view directions, theta angles, phi angles, and radii.
    """
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