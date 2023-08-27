import os
import time
import glob
import tqdm
import torch
import psutil
import random
import imageio
import cv2 as cv
import numpy as np
import tensorboardX
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from packaging import version as pver
from torch_ema import ExponentialMovingAverage
from utils.functions import getCPUMem, getGPUMem

class NeRFTrainer(object):
    def __init__(
            self,
            expName,
            opt,
            model,
            guidance,
            criterion = None,
            optimiser = None,
            lrScheduler = None,
            emaDecay = None,
            metrics = [],
            device = None,
            verbose = True,
            fp16 = False,
            workspace = "workspace",
            bestMode = "min",
            useLossAsMetric = True,
            reportMetricAtTraining = False,
            useTensorboardX = True,
            schedulerUpdateEveryStep = False
    ):
        self.expName = expName
        self.opt = opt
        self.verbose = verbose
        self.metrics = metrics
        self.workspace = workspace
        self.emaDecay = emaDecay
        self.fp16 = fp16
        self.bestMode = bestMode
        self.useLossAsMetric = useLossAsMetric
        self.reportMetricAtTraining = reportMetricAtTraining
        self.useTensorboardX = useTensorboardX
        self.timeStamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.schedulerUpdateEveryStep = schedulerUpdateEveryStep
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.guidance = guidance

        for parameter in self.guidance.parameters():
            parameter.requires_grad = False
        self.embeddings = {}
        self.prepareEmbeddings()

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimiser is None:
            self.optimiser = optim.Adam(self.model.parameters(), lr = 0.001, weight_decay = 5e-4)
        else:
            self.optimiser = optimiser(self.model)
        
        if lrScheduler is None:
            self.lrScheduler = optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda = lambda epoch: 1)
        else:
            self.lrScheduler = lrScheduler(self.optimiser)
        
        if emaDecay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay = emaDecay)
        else:
            self.ema = None
        
        self.scaler = torch.cuda.amp.GradScaler(enabled = self.fp16)

        self.totalTrainT = 0
        self.epoch = 0
        self.globalStep = 0
        self.localStep = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],
            "ckpts": [],
            "best_result": None
        }

        if len(metrics) == 0 or self.useLossAsMetric:
            self.bestMode = "min"

        self.logPtr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.logPath = os.path.join(workspace, f"log_{self.expName}.txt")
            self.logPtr = open(self.logPath, "a+")
            self.ckptPath = os.path.join(self.workspace, "ckpts")
            self.bestPath = f"{self.ckptPath}/{self.expName}.pth"
            os.makedirs(self.ckptPath, exist_ok = True)
        
        self.log(f'[INFO] opt: {self.opt}')
        self.log(f'[INFO] Trainer: {self.expName} | {self.timeStamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

    @torch.no_grad()
    def prepareEmbeddings(self):
        self.embeddings["default"] = self.guidance.getTextEmbeddings([self.opt.posPrompt])
        self.embeddings["uncond"] = self.guidance.getTextEmbeddings([self.opt.negPrompt])
        for d in ["front", "side", "back"]:
            self.embeddings[d] = self.guidance.getTextEmbeddings([f"{self.opt.posPrompt}, {d} view"])
        
    def __del__(self):
        if self.logPtr:
            self.logPtr.close()
    
    def log(self, *args, **kwargs):
        if self.logPtr:
            print(*args, file = self.logPtr)
            self.logPtr.flush()
    
    def train_step(self, data):
        expIterRatio = (self.globalStep - self.opt.expStartIter) / (self.opt.expEndIter - self.opt.expStartIter)

        if self.opt.progressiveView:
            r = min(1.0, self.opt.progressiveViewInitRatio + 2.0 * expIterRatio)
            self.opt.phiRange = [
                self.opt.defaultAzimuth * (1 - r) + self.opt.fullPhiRange[0] * r,
                self.opt.defaultAzimuth * (1 - r) + self.opt.fullPhiRange[1] * r
            ]
            self.opt.thetaRange = [
                self.opt.defaultPolar * (1 - r) + self.opt.fullThetaRange[0] * r,
                self.opt.defaultPolar * (1 - r) + self.opt.fullThetaRange[1] * r
            ]
            self.opt.radiusRange = [
                self.opt.defaultRadius * (1 - r) + self.opt.fullRadiusRange[0] * r,
                self.opt.defaultRadius * (1 - r) + self.opt.fullRadiusRange[1] * r
            ]
            self.opt.fovyRange = [
                self.opt.defaultFovy * (1 - r) + self.opt.fullFovyRange[0] * r,
                self.opt.defaultFovy * (1 - r) + self.opt.fullFovyRange[1] * r,
            ]
        
        if self.opt.progressiveLevel:
            self.model.max_level = min(1.0, 0.25 + 2.0 * expIterRatio)
        
        raysO = data["rays_o"]
        raysD = data["rays_d"]
        mvp = data["mvp"]

        B, N = raysO.shape[:2]
        H, W = data["H"], data["W"]

        if B > self.opt.batchSize:
            choice = torch.randperm(B)[:self.opt.batchSize]
            B = self.opt.batchSize
            raysO = raysO[choice]
            raysD = raysD[choice]
            mvp = mvp[choice]
        
        if expIterRatio <= self.opt.latentIterRatio:
            ambientRatio = 1.0
            shading = "normal"
            asLatent = True
            binarise = False
            bgColor = None
        
        else:
            ambientRatio = 1.0 if expIterRatio <= self.opt.albedoIterRatio else self.opt.minAmbientRatio + (1.0 - self.opt.minAmbientRatio) * random.random()
            shading = "albedo" if expIterRatio <= self.opt.albedoIterRatio else "textureless" if random.random() >= (1.0 - self.opt.texturelessRatio) else "lambertian"
            asLatent = False
            binarise = False
            bgColor = None if self.opt.bgRadius > 0 and random.random() > 0.5 else torch.rand(3).to(self.device)
        
        # outputs = self.model.render(raysO, raysD)
        outputs = self.model.render(
            raysO, raysD, mvp, H, W,
            staged = False,
            perturb = True,
            bg_color = bgColor,
            ambient_ratio = ambientRatio,
            shading = shading,
            binarize = binarise
        )
        predDepth = outputs["depth"].reshape(B, 1, H, W)
        predMask = outputs["weights_sum"].reshape(B, 1, H, W)
        predNormal = outputs["normal_image"].reshape(B, H, W, 3) if "normal_image" in outputs else None
        predRGB = torch.cat([outputs["image"], outputs["weights_sum"].unsqueeze(-1)], dim = -1).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous() if asLatent else outputs["image"].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
        
        loss = 0
        azimuth = data["azimuth"]
        textZ = [self.embeddings["uncond"]] * azimuth.shape[0]
        for b in range(azimuth.shape[0]):
            if azimuth[b] >= -90 and azimuth[b] < 90:
                r = 1 - azimuth[b] / 90 if azimuth[b] >= 0 else 1 + azimuth[b] / 90
                startZ = self.embeddings["front"]
                endZ = self.embeddings["side"]
            else:
                r = 1 - (azimuth[b] - 90) / 90 if azimuth[b] >= 0 else 1 + (azimuth[b] + 90) / 90
                startZ = self.embeddings["side"]
                endZ = self.embeddings["back"]
            textZ.append(r * startZ + (1 - r) * endZ)
        textZ = torch.cat(textZ, dim = 0)
        loss = loss + self.guidance.trainStep(
            textZ, predRGB, asLatent = asLatent,
            guidanceScale = self.opt.guidanceScale,
            gradScale = self.opt.lambdaGuidance
        )
        
        loss = loss + self.opt.lambdaOpacity * (outputs["weights_sum"] ** 2).mean() if self.opt.lambdaOpacity > 0 else loss
        
        if self.opt.lambdaEntropy > 0:
            alphas = outputs["weights"].clamp(1e-5, 1- 1e-5)
            lossEntropy = (-alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
            lambdaEntropy = self.opt.lambdaEntropy * min(1, 2* self.globalStep / self.opt.iters)
            loss = loss + lambdaEntropy * lossEntropy
        
        if self.opt.lambda2dNormalSmooth > 0 and "normal_image" in outputs:
            lossSmooth = (predNormal[:, 1:, :, :] - predNormal[:, :-1, :, :]).square().mean() + \
                (predNormal[:, :, 1:, :] - predNormal[:, :, :-1, :]).square().mean()
            loss = loss + self.opt.lambda2dNormalSmooth * lossSmooth
        
        loss = loss + self.opt.lambdaOrient * outputs["loss_orient"] if self.opt.lambdaOrient > 0 and "loss_orient" in outputs else loss
        
        loss = loss + self.opt.lambda3dNormalSmooth + outputs["loss_normal_perturb"] if self.opt.lambda3dNormalSmooth > 0 and "loss_normal_perturb" in outputs else loss
                
        return predRGB, predDepth, loss
    
    def post_train_step(self):
        self.scaler.unscale_(self.optimiser)
        if self.opt.gradClip >= 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.gradClip)
    
    def eval_step(self, data):
        raysO = data["rays_o"]
        raysD = data["rays_d"]
        mvp = data["mvp"]
        
        B, N = raysO.shape[:2]
        H, W = data["H"], data["W"]

        shading = data["shading"] if "shading" in data else "albedo"
        ambientRatio = data["ambient_ratio"] if "ambient_ratio" in data else 1.0
        lightD = data["light_d"] if "light_d" in data else None

        # outputs = self.model.render(raysO, raysD)
        outputs = self.model.render(
            raysO, raysD, mvp, H, W, staged = True,
            perturb = False, bg_color = None, light_d = lightD,
            ambient_ratio = ambientRatio, shading = shading
        )
        predRGB = outputs["image"].reshape(B, H, W, 3)
        predDepth = outputs["depth"].reshape(B, H, W)

        loss = torch.zeros([1], device = predRGB.device, dtype = predRGB.dtype)

        return predRGB, predDepth, loss
    
    def test_step(self, data, bgColor = None, perturb = False):
        raysO = data["rays_o"]
        raysD = data["rays_d"]
        mvp = data["mvp"]
        
        B, N = raysO.shape[:2]
        H, W = data["H"], data["W"]

        bgColor = bgColor.to(raysO.device) if bgColor is not None else None

        shading = data["shading"] if "shading" in data else "albedo"
        ambientRatio = data["ambient_ratio"] if "ambient_ratio" in data else 1.0
        lightD = data["light_d"] if "light_d" in data else None

        # outputs = self.model.render(raysO, raysD)
        outputs = self.model.render(
            raysO, raysD, mvp, H, W, staged = True,
            perturb = perturb, bg_color = bgColor, light_d = lightD,
            ambient_ratio = ambientRatio, shading = shading
        )

        predRGB = outputs["image"].reshape(B, H, W, 3)
        predDepth = outputs["depth"].reshape(B, H, W)

        return predRGB, predDepth, None
    
    def saveMesh(self, path = None):
        if path is None:
            path = os.path.join(self.workspace, "mesh")
        
        os.makedirs(path, exist_ok = True)
        
        self.log(f"--> Saving mesh to {path}")
        # self.model.exportMesh(
        #     path, resolution = self.opt.mcubesResolution,
        #     decimateT = self.opt.decimateTarget
        # )
        self.model.export_mesh(
            path, resolution = self.opt.mcubesResolution,
            decimate_target = self.opt.decimateTarget
        )
        self.log(f"--> Finished saving mesh.")
    
    def trainOneEpoch(self, loader, maxEpochs):
        self.log(f"--> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Starting {self.workspace} Epoch {self.epoch} / {maxEpochs}, lr = {self.optimiser.param_groups[0]['lr']:.6f}...")

        totalLoss = 0
        if self.reportMetricAtTraining:
            for metric in self.metrics:
                metric.clear()
        self.model.train()

        pbar = tqdm.tqdm(total = len(loader) * loader.batch_size, bar_format = "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        self.localStep = 0

        for data in loader:
            if self.globalStep % self.opt.updateExtraInterval == 0:
                with torch.cuda.amp.autocast(enabled = self.fp16):
                    # self.model.updateExtraState()
                    self.model.update_extra_state()
            
            self.localStep += 1
            self.globalStep += 1

            self.optimiser.zero_grad()

            with torch.cuda.amp.autocast(enabled = self.fp16):
                predRGBs, predDepth, loss = self.train_step(data)
            
            if self.opt.gradClipRGB >= 0:
                def _hook(grad):
                    if self.fp16:
                        gradScale = self.scaler._get_scale_async()
                        return grad.clamp(gradScale * -self.opt.gradClipRGB, gradScale * self.opt.gradClipRGB)
                    else:
                        return grad.clamp(-self.opt.gradClipRGB, self.opt.gradClipRGB)
                predRGBs.register_hook(_hook)
            
            self.scaler.scale(loss).backward()
            
            self.post_train_step()
            self.scaler.step(self.optimiser)
            self.scaler.update()

            if self.schedulerUpdateEveryStep:
                self.lrScheduler.step()
            
            lossVal = loss.item()
            totalLoss += lossVal

            if self.useTensorboardX:
                self.writer.add_scalar("train/loss", lossVal, self.globalStep)
                self.writer.add_scalar("train/lr", self.optimiser.param_groups[0]["lr"], self.globalStep)
            
            if self.schedulerUpdateEveryStep:
                pbar.set_description(f"loss = {lossVal:.4f} ({totalLoss / self.localStep:.4f}), lr = {self.optimiser.param_groups[0]['lr']:.6f}")
            else:
                pbar.set_description(f"loss = {lossVal:.4f} ({totalLoss / self.localStep:.4f})")
            pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()
        
        averageLoss = totalLoss / self.localStep
        self.stats["loss"].append(averageLoss)

        pbar.close()
        if self.reportMetricAtTraining:
            for metric in self.metrics:
                self.log(metric.report(), style = "red")
                if self.useTensorboardX:
                    metric.write(self.writer, self.epoch, prefix = "train")
                metric.clear()
        
        if not self.schedulerUpdateEveryStep:
            if isinstance(self.lrScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lrScheduler.step()
            else:
                self.lrScheduler.step()
        
        cpuMem, gpuMem = getCPUMem(), getGPUMem()[0]
        self.log(f"--> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Finished Epoch {self.epoch} / {maxEpochs}. CPU = {cpuMem:.1f}GB, GPU = {gpuMem:.1f}GB.")
    
    def evaluateOneEpoch(self, loader, name = None):
        self.log(f"--> Evaluation of {self.workspace} at epoch {self.epoch}....")

        if name is None:
            name = f"{self.expName}_Epoch{self.epoch:04d}"
        
        totalLoss = 0
        for metric in self.metrics:
            metric.clear()
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()
        
        pbar = tqdm.tqdm(total = len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.localStep = 0
            
            for data in loader:
                self.localStep += 1

                with torch.cuda.amp.autocast(enabled = self.fp16):
                    preds, predsDepths, loss = self.eval_step(data)
                
                lossVal = loss.item()
                totalLoss += lossVal

                savePath = os.path.join(self.workspace, "validation", f"{name}_{self.localStep:04d}_rgb.png")
                savePathDepth = os.path.join(self.workspace, "validation", f"{name}_{self.localStep:04d}_depth.png")

                os.makedirs(os.path.dirname(savePath), exist_ok = True)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255.0).astype(np.uint8)

                predDepth = predsDepths[0].detach().cpu().numpy()
                predDepth = (predDepth - predDepth.min()) / (predDepth.max() - predDepth.min() + 1e-6)
                predDepth = (predDepth * 255.0).astype(np.uint8)

                cv.imwrite(savePath, cv.cvtColor(pred, cv.COLOR_RGB2BGR))
                cv.imwrite(savePathDepth, predDepth)

                pbar.set_description(f"loss = {lossVal:.4f} ({totalLoss / self.localStep:.4f})")
                pbar.update(loader.batch_size)
        
        averageLoss = totalLoss / self.localStep
        self.stats["valid_loss"].append(averageLoss)

        pbar.close()
        if not self.useLossAsMetric and len(self.metrics) > 0:
            result = self.metrics[0].measure()
            self.stats["results"].append(result if self.bestMode == "min" else -result)
        else:
            self.stats["results"].append(averageLoss)
        
        for metric in self.metrics:
            self.log(metric.report(), style = "blue")
            if self.useTensorboardX:
                metric.write(self.writer, self.epoch, prefix = "evaluate")
            metric.clear()
        
        if self.ema is not None:
            self.ema.restore()
        
        self.log(f"--> Evaluation epoch (self.epoch) finished.")
    
    def test(self, loader, savePath = None, name = None, writeVideo = True):
        if savePath is None:
            savePath = os.path.join(self.workspace, "results")
        
        if name is None:
            name = f"{self.expName}_Epoch{self.epoch:04f}"
        
        os.makedirs(savePath, exist_ok = True)

        self.log(f"--> Starting testing, saving results to {savePath}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if writeVideo:
            allPreds = []
            allPredsDepth = []
        
        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled = self.fp16):
                    preds, predsDepth, _ = self.test_step(data)
                
                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255.0).astype(np.uint8)

                predDepth = predsDepth[0].detach().cpu().numpy()
                predDepth = (predDepth - predDepth.min()) / (predDepth.max() - predDepth.min() + 1e-6)
                predDepth = (predDepth * 255.0).astype(np.uint8)

                if writeVideo:
                    allPreds.append(pred)
                    allPredsDepth.append(predDepth)
                else:
                    cv.imwrite(os.path.join(savePath, f"{name}_{i:04d}_rgb.png"), cv.cvtColor(pred, cv.COLOR_RGB2BGR))
                    cv.imwrite(os.path.join(savePath, f"{name}_{i:04d}_depth.png"), predDepth)
                pbar.update(loader.batch_size)
        
        if writeVideo:
            allPreds = np.stack(allPreds, axis = 0)
            allPredsDepth = np.stack(allPredsDepth, axis = 0)
            imageio.mimwrite(os.path.join(savePath, f"{name}_rgb.mp4"), allPreds, fps = 25, quality = 8, macro_block_size = 1)
            imageio.mimwrite(os.path.join(savePath, f"{name}_depth.mp4"), allPredsDepth, fps = 25, quality = 8, macro_block_size = 1)
        
        self.log(f"--> Testing finished.")
    
    def train(self, trainLoader, validLoader, testLoader, maxEpochs):
        if self.useTensorboardX:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.expName))
        
        startT = time.time()
        for epoch in range(self.epoch + 1, maxEpochs + 1):
            self.epoch = epoch
            self.trainOneEpoch(trainLoader, maxEpochs)
            
            if self.epoch % self.opt.evalInterval == 0:
                self.evaluateOneEpoch(validLoader)
            
            if self.epoch % self.opt.testInterval == 0 or self.epoch == maxEpochs:
                self.test(testLoader, writeVideo = self.opt.writeVideo)
        
        endT = time.time()
        
        self.totalTrainT = endT - startT + self.totalTrainT

        self.log(f"--> Training took {(self.totalTrainT) / 60:.4f} minutes.")

        if self.useTensorboardX:
            self.writer.close()
    
    def evaluate(self, loader, name = None):
        self.useTensorboardX, useTensorboardX = False, self.useTensorboardX
        self.evaluateOneEpoch(loader, name)
        self.useTensorboardX = useTensorboardX
    
    def saveCkpt(self, name = None, full = False, best = False):
        if name is None:
            name = f"{self.expName}_epoch{self.epoch:04d}"
        state = {
            "epoch": self.epoch,
            "global_step": self.globalStep,
            "stats": self.stats
        }
        state["mean_density"] = self.model.mean_density
        if full:
            state["optimiser"] = self.optimiser.state_dict()
            state["lr_scheduler"] = self.lrScheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()
        if not best:
            state["model"] = self.model.state_dict()
            filePath = f"{name}.pth"
            self.stats["checkpoint"].append(filePath)
            if len(self.stats["ckpts"]) > 2:
                oldCkpt = os.path.join(self.ckptPath, self.state["checkpoints"].pop(0))
                if os.path.exists(oldCkpt):
                    os.remove(oldCkpt)
            torch.save(state, os.path.join(self.ckptPath, filePath))
        else:
            if len(self.stats["results"]) > 0:
                if True:
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()
                    state["model"] = self.model.state_dict()
                    if self.ema is not None:
                        self.ema.restore()
                    torch.save(state, self.bestPath)
            else:
                self.log(f"Warning: Saving checkpoints has been skipped as no evaluated results have been found.")
    
    def loadCkpt(self, ckpt = None, modelOnly = False):
        if ckpt is None:
            ckptList = sorted(glob.glob(f"{self.ckptPath}/*.pth"))
            if ckptList:
                ckpt = ckptList[-1]
            else:
                return
        ckptDict = torch.load(ckpt, map_location = self.device)
        if "model" not in ckptDict:
            self.model.load_state_dict(ckptDict)
            self.log("Model Loaded")
        mKeys, uKeys = self.model.load_state_dict(ckptDict["model"], strict = False)
        self.log("Model Loaded")
        if len(mKeys) > 0 or len(uKeys) > 0:
            self.log("Missing Keys:\n {mKeys}\nUnexpected Keys:\n {uKeys}")
        if self.ema is not None and "ema" in ckptDict:
            try:
                self.ema.load_state_dict(ckptDict["ema"])
                self.log("Loaded Expo. Moving Avg.")
            except:
                self.log("Warning: Failed to loa Expo. Moving Avg.")
        if "mean_density" in ckptDict:
            self.model.mean_density = ckptDict["mean_density"]
        if modelOnly:
            return
        self.stats = ckptDict["stats"]
        self.epoch = ckptDict["epoch"]
        self.globalStep = ckptDict["global_step"]
        self.log(f"Model loaded at {self.epoch} epochs, {self.globalStep} global step.")
        if self.optimiser and "optimiser" in ckptDict:
            try:
                self.optimiser.load_state_dict(ckptDict["optimiser"])
                self.log("Loaded the optimiser.")
            except:
                self.log("Warning: Failed to load optimiser.")
        if self.lrScheduler and "lr_scheduler" in ckptDict:
            try:
                self.lrScheduler.load_state_dict(ckptDict["lr_scheduler"])
                self.log("Loaded the LR Scheduler.")
            except:
                self.log("Warning: Failed to load LR Scheduler.")
        if self.scaler and "scaler" in ckptDict:
            try:
                self.lrScheduler.load_state_dict(ckptDict["scaler"])
                self.log("Loaded the scaler.")
            except:
                self.log("Warning: Failed to load scaler.")