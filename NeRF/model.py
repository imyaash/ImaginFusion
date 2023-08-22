import torch
import numpy as np
import torch.nn as nn
import tinycudann as tcnn
import torch.nn.functional as F
from utils.encoder import Encoder
from NeRF.renderer import NeRFRenderer
from utils.functions import safeNormalise
from utils.activator import truncExp, softplusBiased

class NN(nn.Module):
    def __init__(self, inDim, outDim, hiddenDim, nLayers, bias=True):
        super().__init__()
        self.inDim = inDim
        self.outDim = outDim
        self.hiddenDim = hiddenDim
        self.nLayers = nLayers
        self.network = nn.ModuleList([
            nn.Linear(
            self.inDim if layer == 0 else self.hiddenDim,
            self.outDim if layer == nLayers - 1 else self.hiddenDim,
            bias = bias
            ) for layer in range(nLayers)
        ])
    
    def forward(self, x):
        for layer in range(self.nLayers):
            x = self.network[layer](x)
            if layer != self.nLayers - 1:
                x = F.relu(x, inplace=True)
        return x

class NeRF(NeRFRenderer):
    def __init__(
            self, args,
            nLayers = 3,
            hiddenDim = 64,
            nLayersBG = 2,
            hiddenDimBG = 32
    ):
        super().__init__(args)
        self.nLayers = nLayers
        self.hiddenDim = hiddenDim
        self.encoder = tcnn.Encoding(
            n_input_dims = 3,
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "interpolation": "Smoothstep",
                "per_level_scale": np.exp2(
            np.log2(2048 * self.bound / 16) / (16 - 1)
                )
            }, dtype = torch.float32
        )
        self.inDim = self.encoder.n_output_dims
        self.sigmaNet = NN(self.inDim, 4, hiddenDim, nLayers, True)
        self.densityActivation = truncExp if self.args.densityActivation == "exp" else softplusBiased
        if self.args.bgRadius > 0:
            self.nLayersBG = nLayersBG
            self.hiddenDimBG = hiddenDimBG
            self.encoderBG, self.inDimBG = Encoder(3, 6)
            self.bgNet = NN(self.inDimBG, 3, hiddenDimBG, nLayersBG, True)
        else:
            self.bgNet = None
    
    def commonForward(self, x):
        enc = self.encoder(
            (x + self.bound) / (2 * self.bound)
        ).float()
        h = self.sigmaNet(enc)
        sigma = self.densityActivation(h[..., 0] + self.densityBlob(x))
        albedo = torch.sigmoid(h[..., 1:])
        return sigma, albedo
    
    def normal(self, x):
        with torch.enable_grad():
            with torch.cuda.amp.autocast(enabled = False):
                x.requires_grad_(True)
                sigma, albedo = self.commonForward(x)
                normal = -torch.autograd.grad(
                    torch.sum(sigma), x, create_graph = True
                )[0]
        normal = safeNormalise(normal)
        normal = torch.nan_to_num(normal)
        return normal
    
    def forward(self, x, d, l = None, ratio = 1, shading = "albedo"):
        if shading == "albedo":
            sigma, albedo = self.commonForward(x)
            normal = None
            color = albedo
        else:
            with torch.enable_grad():
                with torch.cuda.amp.autocast(enabled = False):
                    x.requires_grad_(True)
                    sigma, albedo = self.commonForward(x)
                    normal = -torch.autograd.grad(
                        torch.sum(sigma), x, create_graph = True
                    )[0]
            normal = safeNormalise(normal)
            normal = torch.nan_to_num(normal)
            lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min = 0)
            if shading == "textureless":
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == "normal":
                color = (normal + 1) / 2
            else:
                color = albedo * lambertian.unsqueeze(-1)
        return sigma, color, normal
    
    def density(self, x):
        sigma, albedo = self.commonForward(x)
        return {
            "sigma": sigma,
            "albedo": albedo
        }
    
    def background(self, d):
        h = self.encoderBG(d)
        h = self.bgNet(h)
        rgbs = torch.sigmoid(h)
        return rgbs
    
    def getParams(self, lr):
        params = [
            {
                "params": self.encoder.parameters(),
                "lr": lr * 10
            },
            {
                "params": self.sigmaNet.parameters(),
                "lr": lr
            }
        ]
        if self.args.bgRadius > 0:
            params.append(
                {
                    "params": self.bgNet.parameters(),
                    "lr": lr
                }
            )
        return params