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
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class StableDiffusionModel(nn.Module):
    def __init__(self, device, fp16, version = "2.1", hfModelKey = None, tRange = [0.02, 0.98]):
        super().__init__()

        self.device = device
        self.version = version

        if hfModelKey is not None:
            print(f"Using custom pretrained stable-diffusion model from HuggingFace {hfModelKey}")
            modelKey = hfModelKey
        else:
            # Save model path and name in JSON format and fetch as key
            if self.version == "2.1":
                modelKey = "stabilityai/stable-diffusion-2-1"
            elif self.version == "1.4":
                modelKey = "CompVis/stable-diffusion-v1-4"
            elif self.version == "1.5":
                modelKey = "runwayml/stable-diffusion-v1-5"
            else:
                raise ValueError(f"Unsupported stable-diffusion version {self.version}")
            
        modelPath = f"pretrainedSD/{modelKey}"
        if os.path.exists(modelPath):
            print(f"Loading pretrained stable-diffusion model from {modelPath}")
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

        self.numSteps = self.scheduler.config.num_train_timesteps
        self.minSteps = int(self.numSteps * tRange[0])
        self.maxSteps = int(self.numSteps * tRange[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        
        print(f"Successfully loaded Stable-Diffusion ({modelKey})")
    
    @torch.no_grad()
    def getTextEmbeddings(self, prompt):
        inputs = self.tokenizer(prompt, padding = "max_length", max_length = self.tokenizer.model_max_length, return_tensors = "pt")
        embeddings = self.textEncoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    
    def produceLatents(self, embeddings, h = 512, w = 512, numSteps = 50, guidanceScale = 7.5, latents = None):
        if latents is None:
            latents = torch.randn((embeddings.shape[0] // 2, self.unet.config.in_channels, h // 8, w // 8), device = self.device, dtype = self.precisionT)
        self.scheduler.set_timesteps(numSteps)
        for i, t in enumerate(self.scheduler.timesteps):
            latentModelInput = torch.cat([latents] * 2)
            noisePred = self.unet(latentModelInput, t, encoder_hidden_states = embeddings)["sample"]
            noisePredUncond, noisePredCond = noisePred.chunk(2)
            noisePred = noisePredUncond + guidanceScale * (noisePredCond - noisePredUncond)
            latents = self.scheduler.step(noisePred, t, latents)["prev_sample"]
        return latents
    
    def decodeLatents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
    
    def encodeImages(self, images):
        images = 2 * images - 1
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents
    
    def trainStep(self, embeddings, predRGB, guidanceScale = 100, asLatent = False, gradScale = 1):
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

        loss = (grad * latents).sum()

        return loss
    
    def txt2Img(self, prompt, negetivePrompt = "", h = 512, w = 512, numSteps = 100, guidanceScale = 7.5):
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negetivePrompt, str):
            negetivePrompt = [negetivePrompt]
        
        positiveEmbeddings = self.getTextEmbeddings(prompt)
        negetiveEmbeddings = self.getTextEmbeddings(negetivePrompt)
        embeddings = torch.cat([positiveEmbeddings, negetiveEmbeddings], dim = 0)

        latents = self.produceLatents(embeddings, h = h, w = w, numSteps = numSteps, guidanceScale = guidanceScale)
        image = self.decodeLatents(latents)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        return image