import torch
import numpy as np
import torch.nn as nn
import tinycudann as tcnn
from .renderer import Renderer
import torch.nn.functional as F
from utils.encoder import encoder
from utils.functions import normalise
from utils.activator import truncExp, softplus

class Network(nn.Module):
    """
    Neural Network module consisting of multiple linear layers.

    Args:
        inDim (int): Input dimension.
        outDim (int): Output dimension.
        hiddenDim (int): Hidden dimension.
        nLayers (int): Number of layers in the network.
        bias (bool): Whether to include bias terms in linear layers.

    Attributes:
        inDim (int): Input dimension.
        outDim (int): Output dimension.
        hiddenDim (int): Hidden dimension.
        nLayers (int): Number of layers in the network.
        network (nn.ModuleList): List of linear layers.

    Methods:
        forward(x): Forward pass through the network.
    """
    def __init__(self, inDim, outDim, hiddenDim, nLayers, bias = True):
        super().__init__()
        self.inDim = inDim
        self.outDim = outDim
        self.hiddenDim = hiddenDim
        self.nLayers = nLayers
        self.network = nn.ModuleList(
            [
                nn.Linear(
            self.inDim \
                if layer == 0 \
                    else self.hiddenDim,
            self.outDim \
                if layer == nLayers - 1 \
                    else self.hiddenDim,
            bias = bias
                )
                for layer in range(nLayers)
            ]
        )
    
    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in range(self.nLayers):
            x = self.network[layer](x)
            if layer != self.nLayers - 1:
                x = F.relu(x, inplace=True)
        return x


class NeRF(Renderer):
    """
    Neural Radiance Fields (NeRF) renderer.

    Args:
        args: Arguments for the renderer.
        nLayers (int): Number of layers in the NeRF network.
        hiddenDim (int): Hidden dimension of the NeRF network.
        nLayersBG (int): Number of layers in the background network.
        hiddenDimBG (int): Hidden dimension of the background network.

    Attributes:
        nLayers (int): Number of layers in the NeRF network.
        hiddenDim (int): Hidden dimension of the NeRF network.
        encoder (tcnn.Encoding): TinyCUDA neural network encoder.
        inDim (int): Input dimension of the encoder.
        sigmaNet (Network): NeRF network for predicting sigma and albedo.
        densityActivation (function): Activation function for density prediction.
        nLayersBG (int): Number of layers in the background network.
        hiddenDimBG (int): Hidden dimension of the background network.
        encoderBG (nn.Module): Background encoder.
        inDimBG (int): Input dimension of the background encoder.
        netBG (Network): Background network.

    Methods:
        forwardC(x): Forward pass for NeRF color prediction.
        normal(x): Compute surface normals.
        forward(x, d, l, ratio, shading): Forward pass for NeRF rendering.
        density(x): Predict density values.
        background(d): Predict background values.
        get_params(lr): Get network parameters and learning rates.
    """
    def __init__(
            self,
            args,
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
            },
            dtype = torch.float32
        )
        self.inDim = self.encoder.n_output_dims
        self.sigmaNet = Network(self.inDim, 4, hiddenDim, nLayers, bias=True)
        self.densityActivation = truncExp \
            if self.args.densityActivation == 'exp' \
                else softplus
        if self.args.bgRadius > 0:
            self.nLayersBG = nLayersBG   
            self.hiddenDimBG = hiddenDimBG
            self.encoderBG, self.inDimBG = encoder(inDim = 3, multiRes = 6)
            self.netBG = Network(self.inDimBG, 3, hiddenDimBG, nLayersBG, bias=True)
        else:
            self.netBG = None

    def forwardC(self, x):
        """
        Forward pass for NeRF colour prediction.

        Args:
            x (torch.Tensor): Input coordinates.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sigma and albedo predictions.
        """
        enc = self.encoder((x + self.bound) / (2 * self.bound)).float()
        h = self.sigmaNet(enc)
        sigma = self.densityActivation(h[..., 0] + self.densityBlob(x))
        albedo = torch.sigmoid(h[..., 1:])
        return sigma, albedo
    
    def normal(self, x):
        """
        Compute surface normals.

        Args:
            x (torch.Tensor): Input coordinates.

        Returns:
            torch.Tensor: Surface normals.
        """
        with torch.enable_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x.requires_grad_(True)
                sigma, albedo = self.forwardC(x)
                normal = -torch.autograd.grad(
                    torch.sum(sigma), x, create_graph=True
                )[0]
        normal = normalise(normal)
        normal = torch.nan_to_num(normal)
        return normal
    
    def forward(self, x, d, l = None, ratio = 1, shading = 'albedo'):
        """
        Forward pass for NeRF rendering.

        Args:
            x (torch.Tensor): Input coordinates.
            d (torch.Tensor): Depth values.
            l (torch.Tensor, optional): Light direction vectors. Defaults to None.
            ratio (int, optional): Lambertian ratio. Defaults to 1.
            shading (str, optional): Shading mode ("albedo", "normal", "textureless"). Defaults to 'albedo'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Sigma, colour, and normal predictions.
        """
        if shading == 'albedo':
            sigma, albedo = self.forwardC(x)
            normal = None
            color = albedo
        else:
            with torch.enable_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x.requires_grad_(True)
                    sigma, albedo = self.forwardC(x)
                    normal = -torch.autograd.grad(
                        torch.sum(sigma), x, create_graph=True
                    )[0]
            normal = normalise(normal)
            normal = torch.nan_to_num(normal)
            lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0)
            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else:
                color = albedo * lambertian.unsqueeze(-1)
        return sigma, color, normal
      
    def density(self, x):
        """
        Predict density values.

        Args:
            x (torch.Tensor): Input coordinates.

        Returns:
            Dict[str, torch.Tensor]: Predicted sigma and albedo.
        """     
        sigma, albedo = self.forwardC(x)
        return {
            'sigma': sigma,
            'albedo': albedo,
        }

    def background(self, d):
        """
        Predict background values.

        Args:
            d (torch.Tensor): Depth values.

        Returns:
            torch.Tensor: Predicted background values.
        """
        h = self.encoderBG(d)
        h = self.netBG(h)
        return torch.sigmoid(h)

    def get_params(self, lr):
        """
        Get network parameters and learning rate.

        Args:
            lr (float): Learning rate.

        Returns:
            List[Dist[str, Union[nn.Parameter, float]]]: List of parameter dictionaries.
        """
        params = [
            {
                'params': self.encoder.parameters(),
                'lr': lr * 10
            },
            {
                'params': self.sigmaNet.parameters(),
                'lr': lr
            }
        ]
        if self.args.bgRadius > 0:
            params.append(
                {
                    'params': self.netBG.parameters(),
                    'lr': lr
                }
            )
        return params