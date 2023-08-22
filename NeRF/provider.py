import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.functions import circle_poses, rand_poses, getRays


class NeRFDataset:
    def __init__(self, opt, device, type='train', H=256, W=256, size=100):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.H = H
        self.W = W
        self.size = size

        self.training = self.type in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.near = self.opt.minNear
        self.far = 1000 # infinite

        # [debug] visualize poses
        # poses, dirs, _, _, _ = rand_poses(100, self.device, opt, radius_range=self.opt.radius_range, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=1)
        # visualize_poses(poses.detach().cpu().numpy(), dirs.detach().cpu().numpy())

    def get_default_view_data(self):

        H = int(self.opt.knownViewScale * self.H)
        W = int(self.opt.knownViewScale * self.W)
        cx = H / 2
        cy = W / 2

        radii = torch.FloatTensor(self.opt.refRadii).to(self.device)
        thetas = torch.FloatTensor(self.opt.refPolars).to(self.device)
        phis = torch.FloatTensor(self.opt.refAzimuths).to(self.device)
        poses, dirs = circle_poses(self.device, radius=radii, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angleOverhead, angle_front=self.opt.angleFront)
        fov = self.opt.defaultFovy
        focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, cx, cy])

        projection = torch.tensor([
            [2*focal/W, 0, 0, 0],
            [0, -2*focal/H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(len(radii), 1, 1)

        mvp = projection @ torch.inverse(poses) # [B, 4, 4]

        # sample a low-resolution but full image
        rays = getRays(poses, intrinsics, H, W, -1)

        data = {
            'H': H,
            'W': W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'mvp': mvp,
            'polar': self.opt.refPolars,
            'azimuth': self.opt.refAzimuths,
            'radius': self.opt.refRadii,
        }

        return data

    def collate(self, index):

        B = len(index)

        if self.training:
            # random pose on the fly
            poses, dirs, thetas, phis, radius = rand_poses(B, self.device, self.opt, radius_range=self.opt.radiusRange, theta_range=self.opt.thetaRange, phi_range=self.opt.phiRange, return_dirs=True, angle_overhead=self.opt.angleOverhead, angle_front=self.opt.angleFront, uniform_sphere_rate=self.opt.uniformSphereRate)

            # random focal
            fov = random.random() * (self.opt.fovyRange[1] - self.opt.fovyRange[0]) + self.opt.fovyRange[0]

        elif self.type == 'six_views':
            # six views
            thetas_six = [90, 90,  90,  90, 1e-3, 179.999]
            phis_six =   [ 0, 90, 180, -90,    0,       0]
            thetas = torch.FloatTensor([thetas_six[index[0]]]).to(self.device)
            phis = torch.FloatTensor([phis_six[index[0]]]).to(self.device)
            radius = torch.FloatTensor([self.opt.defaultRadius]).to(self.device)
            poses, dirs = circle_poses(self.device, radius=radius, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angleOverhead, angle_front=self.opt.angleFront)

            # fixed focal
            fov = self.opt.defaultFovy

        else:
            # circle pose
            thetas = torch.FloatTensor([self.opt.defaultPolar]).to(self.device)
            phis = torch.FloatTensor([(index[0] / self.size) * 360]).to(self.device)
            radius = torch.FloatTensor([self.opt.defaultRadius]).to(self.device)
            poses, dirs = circle_poses(self.device, radius=radius, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angleOverhead, angle_front=self.opt.angleFront)

            # fixed focal
            fov = self.opt.defaultFovy

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        projection = torch.tensor([
            [2*focal/self.W, 0, 0, 0],
            [0, -2*focal/self.H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        mvp = projection @ torch.inverse(poses) # [1, 4, 4]

        # sample a low-resolution but full image
        rays = getRays(poses, intrinsics, self.H, self.W, -1)

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - self.opt.defaultPolar
        delta_azimuth = phis - self.opt.defaultAzimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        delta_radius = radius - self.opt.defaultRadius

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'mvp': mvp,
            'polar': delta_polar,
            'azimuth': delta_azimuth,
            'radius': delta_radius,
        }

        return data

    def dataloader(self, batch_size=None):
        batch_size = batch_size or self.opt.batchSize
        loader = DataLoader(list(range(self.size)), batch_size=batch_size, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self
        return loader