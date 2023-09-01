import os
import torch
import cv2 as cv
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline

# Suppressing unnecessary warnings
logging.set_verbosity_error()

def seeder(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class StableDiffusionModel(nn.Module):
    """
    A class representing a Stable Diffusion Model for image generation.

    Args:
        device (str): The device to run the model on (e.g., "cuda" or "cpu").
        fp16 (bool): Whether to use float16 precision.
        version (str): The version of the stable diffusion model to use.
        hfModelKey (str): HuggingFace model key for a custom pretrained stable-diffusion model.
        tRange (list): A list specifying the range for the number of diffusion steps.

    Attributes:
        device (str): The device on which the model is running.
        version (str): The version of the stable diffusion model being used.
        modelPath (str): The path to the pretrained stable-diffusion model.
        precisionT (torch.dtype): The precision type for model tensors.
        vae (nn.Module): The Variational Autoencoder component of the model.
        tokenizer: The text tokenizer used by the model.
        textEncoder: The text encoder used by the model.
        unet: The UNet component of the model.
        scheduler: The diffusion scheduler used by the model.
        numSteps (int): The total number of diffusion steps.
        minSteps (int): The minimum number of diffusion steps in the specified range.
        maxSteps (int): The maximum number of diffusion steps in the specified range.
        alphas (torch.Tensor): The alpha values used in the diffusion process.

    Methods:
        getTextEmbeddings(prompt): Get text embeddings for a given prompt.
        produceLatents(embeddings, h, w, numSteps, guidanceScale, latents): Generate latent vectors.
        decodeLatents(latents): Decode latent vectors into images.
        encodeImages(images): Encode images into latent vectors.
        trainStep(embeddings, predRGB, guidanceScale, asLatent, gradScale): Perform a training step.

    """
    def __init__(self, device, fp16, version = "2.1", hfModelKey = None, tRange=None):
        if tRange is None:
            tRange = [0.02, 0.98]
        super().__init__()

        self.device = device
        self.version = version

        # Model version check
        if hfModelKey is not None:
            print(f"Using custom pretrained stable-diffusion model from HuggingFace {hfModelKey}")
            modelKey = hfModelKey
        elif self.version == "2.1":
            modelKey = "stabilityai/stable-diffusion-2-1"
        elif self.version == "1.4":
            modelKey = "CompVis/stable-diffusion-v1-4"
        elif self.version == "1.5":
            modelKey = "runwayml/stable-diffusion-v1-5"
        elif self.version == "2.1-base":
            modelKey = "stabilityai/stable-diffusion-2-1-base"
        elif self.version == "2.0-base":
            modelKey = "stabilityai/stable-diffusion-2-base"
        else:
            raise ValueError(f"Unsupported stable-diffusion version {self.version}")
        
        modelPath = f"sdm/pretrained/{modelKey}"
        # Loading model path ckpts.
        if os.path.exists(modelPath):
            print(f"Loading pretrained stable-diffusion model from {modelPath}")
        # Downloading & saving the model ckpts.
        else:
            print(f"Could not find pretrained stable-diffusion model at {modelPath}. Downloading and saving to {modelPath}.")
            os.makedirs(modelPath, exist_ok = True)
            model = StableDiffusionPipeline.from_pretrained(modelKey)
            model.save_pretrained(modelPath)
            print(f"Saved pretrained stable-diffusion model to {modelPath}")

        self.modelPath = modelPath
        self.precisionT = torch.float16 if fp16 else torch.float32

        # Loading the pretrained model
        model = StableDiffusionPipeline.from_pretrained(self.modelPath, torch_dtype = self.precisionT)
        model.to(self.device)

        # Extracting required modules from the model 
        self.vae = model.vae
        self.tokenizer = model.tokenizer
        self.textEncoder = model.text_encoder
        self.unet = model.unet

        # Loading the diffusion scheduler
        self.scheduler = DDIMScheduler.from_pretrained(modelKey, subfolder = "scheduler", torch_dtype = self.precisionT)

        # Deleting the pretrained model to free VRAM
        del model

        # Setting model scheduler parameters
        self.numSteps = self.scheduler.config.num_train_timesteps
        self.minSteps = int(self.numSteps * tRange[0])
        self.maxSteps = int(self.numSteps * tRange[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        print(f"Successfully loaded Stable-Diffusion ({modelKey})")
    
    @torch.no_grad()
    def getTextEmbeddings(self, prompt):
        """
        Get text embeddings for a given text prompt.

        Args:
            prompt (str): The text prompt for which embeddings are to be generated.

        Returns:
            torch.Tensor: Text embeddings dor the input prompt.
        """
        inputs = self.tokenizer(prompt, padding = "max_length", max_length = self.tokenizer.model_max_length, return_tensors = "pt")
        return self.textEncoder(inputs.input_ids.to(self.device))[0]
    
    def produceLatents(self, embeddings, h = 512, w = 512, numSteps = 50, guidanceScale = 7.5, latents = None):
        """
        Generate latent vectors.

        Args:
            embeddings (torch.Tensor): Text embeddings.
            h (int, optional): Height of the generated image. Defaults to 512.
            w (int, optional): Width of the generated image. Defaults to 512.
            numSteps (int, optional): Number of diffusion steps. Defaults to 50.
            guidanceScale (float, optional): Scaling factor for guidance. Defaults to 7.5.
            latents (torch.Tensor, optional): Latent vectors. Defaults to None.

        Returns:
            torch.Tensor: Generated latent vectors.
        """
        if latents is None:
            latents = torch.randn((embeddings.shape[0] // 2, self.unet.config.in_channels, h // 8, w // 8), device = self.device, dtype = self.precisionT)
        self.scheduler.set_timesteps(numSteps)
        for t in self.scheduler.timesteps:
            latentModelInput = torch.cat([latents] * 2)
            noisePred = self.unet(latentModelInput, t, encoder_hidden_states = embeddings)["sample"]
            noisePredUncond, noisePredCond = noisePred.chunk(2)
            noisePred = noisePredUncond + guidanceScale * (noisePredCond - noisePredUncond)
            latents = self.scheduler.step(noisePred, t, latents)["prev_sample"]
        return latents
    
    def decodeLatents(self, latents):
        """
        Decode latent vectors into images.

        Args:
            latents (torch.Tensor): Latent vectors to be decoded.

        Returns:
            torch.Tensor: Decoded images.
        """
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
    
    def encodeImages(self, images):
        """
        Encode images into latent vectors.

        Args:
            images (torch.Tensor): Images to be encoded.

        Returns:
            torch.Tensor: Encoded latent vectors.
        """
        images = 2 * images - 1
        posterior = self.vae.encode(images).latent_dist
        return posterior.sample() * self.vae.config.scaling_factor
    
    def trainStep(self, embeddings, predRGB, guidanceScale = 100, asLatent = False, gradScale = 1):
        """
        Training step.

        Args:
            embeddings (torch.Tensor): Text embeddings.
            predRGB (torch.Tensor): Predicted RGB images.
            guidanceScale (int, optional): Scaling factor for guidance. Defaults to 100.
            asLatent (bool, optional): If True, use "predRGB" as latent vectors. Defaults to False.
            gradScale (int, optional): Scaling factor for gradients. Defaults to 1.

        Returns:
            torch.Tensor: Training loss
        """
        if asLatent:
            latents = F.interpolate(predRGB, (64, 64), mode = "bilinear", align_corners = False) * 2 - 1
        else:
            predRGB512 = F.interpolate(predRGB, (512, 512), mode = "bilinear", align_corners = False)
            latents = self.encodeImages(predRGB512)

        t = torch.randint(self.minSteps, self.maxSteps + 1, (latents.shape[0], ), dtype = torch.long, device = self.device)

        with torch.no_grad():
            noise = torch.randn_like(latents)
            noisyLatents = self.scheduler.add_noise(latents, noise, t)
            latentModelInput = torch.cat([noisyLatents] * 2)
            tt = torch.cat([t] * 2)
            noisePred = self.unet(latentModelInput, tt, encoder_hidden_states = embeddings).sample
            noisePredUncond, noisePredPOS = noisePred.chunk(2)
            noisePred = noisePredUncond + guidanceScale * (noisePredPOS - noisePredUncond)

        w = (1 - self.alphas[t])
        grad = gradScale * w[:, None, None, None] * (noisePred - noise)
        grad = torch.nan_to_num(grad)

        return (grad * latents).sum()