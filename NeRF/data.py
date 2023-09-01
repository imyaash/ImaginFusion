import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from utils.functions import circlePoses, randPoses, getRays

class Dataset:
    def __init__(self, args, device, type = "train", H = 256, W = 256, size = 100):
        """
        Initiate the Dataset class.

        Args:
            args (object): A configuration object.
            device (str): The device on which to perform computation.
            type (str, optional): The datase type, either "train" or "all". Defaults to "train".
            H (int, optional): Height of the image. Defaults to 256.
            W (int, optional): Width of the image. Defaults to 256.
            size (int, optional): Size of the dataset. Defaults to 100.
        """
        super().__init__()
        self.args = args
        self.device = device
        self.type = type
        self.H = H
        self.W = W
        self.size = size
        self.training = self.type in ["train", "all"]
        self.cx = self.H / 2
        self.cy = self.W / 2
        self.near = self.args.minNear
        self.far = 1000
    
    def collateFn(self, idx):
        """
        Collate function for creating batches of data.

        Args:
            idx (list): List of indices to select data for batch.

        Returns:
            dict: A dictionary containing batched data.
        """
        if self.training:
            # Generate random poses and directions for training.
            poses, dirs, thetas, phis, radius = randPoses(
                len(idx), self.device, self.args, radRange = self.args.radiusRange,
                thetaRange = self.args.thetaRange, phiRange = self.args.phiRange,
                returnDirs = True, angleOverhead = self.args.angleOverhead,
                angleFront = self.args.angleFront,
                uniSphRate = self.args.uniformSphereRate
            )
            fov = random.random() * (
                self.args.fovyRange[1] - self.args.fovyRange[0]
            ) + self.args.fovyRange[0]
        else:
            # Generate fixed poses for non-training cases.
            thetas = torch.FloatTensor([self.args.defaultPolar]).to(self.device)
            phis = torch.FloatTensor([(idx[0] / self.size) * 360]).to(self.device)
            radius = torch.FloatTensor([self.args.defaultRadius]).to(self.device)
            poses, dirs = circlePoses(
                self.device, radius = radius, theta = thetas, phi = phis, returnDirs = True,
                angleOverhead = self.args.angleOverhead, angleFront = self.args.angleFront
            )
            fov = self.args.defaultFovy
        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array(
            [focal, focal, self.cx, self.cy]
        )
        projection = torch.tensor(
            [
                [2 * focal / self.W, 0, 0, 0],
                [0, -2 * focal / self.H, 0, 0],
                [0, 0, -(self.far + self.near) / (self.far-self.near),
                 -(2 * self.far * self.near) / (self.far - self.near)],
                [0, 0, -1, 0]
            ], dtype = torch.float32, device = self.device
        ).unsqueeze(0)
        mvp = projection @ torch.inverse(poses)
        rays = getRays(poses, intrinsics, self.H, self.W, -1)
        deltaPolar = thetas - self.args.defaultPolar
        deltaAzimuth = phis - self.args.defaultAzimuth
        deltaAzimuth[deltaAzimuth > 180] -= 360
        deltaRadius = radius - self.args.defaultRadius
        return {
            "H": self.H,
            "W": self.W,
            "rays_o": rays["rays_o"],
            "rays_d": rays["rays_d"],
            "dir": dirs,
            "mvp": mvp,
            "polar": deltaPolar,
            "azimuth": deltaAzimuth,
            "radius": deltaRadius
        }
    
    def dataLoader(self, batchSize = None):
        """
        Create a DataLoader for the dataset.

        Args:
            batchSize (int, optional): The batch size. Defaults to None. If not provided, use the default batch size from args.

        Returns:
            DataLoader: A DataLoader object for the dataset.
        """
        batchSize = batchSize or self.args.batchSize
        loader = DataLoader(
            list(
            range(
            self.size
            )
            ), batch_size = batchSize,
            collate_fn = self.collateFn,
            shuffle = self.training,
            num_workers = 0
        )
        loader._data = self
        return loader
    
    def getDefaultViewData(self):
        """
        Get data for default view(s).

        Returns:
            dict: A dictionary containing data for default views.
        """
        H = int(self.args.knownViewScale * self.H)
        W = int(self.args.knownViewScale * self.W)
        cx = H / 2
        cy = W / 2
        radii = torch.FloatTensor(self.args.refRadii).to(self.device)
        thetas = torch.FloatTensor(self.args.refPolars).to(self.device)
        phis = torch.FloatTensor(self.args.refAzimuths).to(self.device)
        poses, dirs = circlePoses(
            self.device, radius = radii, theta = thetas, phi = phis, returnDirs = True,
            angleOverhead = self.args.angleOverhead, angleFront = self.args.angleFront
        )
        fov = self.args.defaultFovy
        focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array(
            [focal, focal, cx, cy]
        )
        projection = torch.tensor(
            [
                [2 * focal / W, 0, 0, 0],
                [0, -2 * focal / H, 0, 0],
                [0, 0, -(self.far + self.near) / (self.far - self.near),
                 -(2 * self.far * self.near) / (self.far - self.near)],
                [0, 0, -1, 0]
            ], dtype = torch.float32, device = self.device
        ).unsqueeze(0).repeat(len(radii), 1, 1)
        mvp = projection @ torch.inverse(poses)
        rays = getRays(poses, intrinsics, H, W, -1)
        return {
            "H": H,
            "W": W,
            "rays_o": rays["rays_o"],
            "rays_d": rays["rays_d"],
            "dir": dirs,
            "mvp": mvp,
            "polar": self.args.refPolars,
            "azimuth": self.args.refAzimuths,
            "radius": self.args.refRadii
        }