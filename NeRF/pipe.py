import torch
import numpy as np
from .model import NeRF
from .trainer import Trainer
from torch.optim import Adam
from .provider import Dataset
from utils.optimiser import Adan
from utils.functions import seeder
from sdm.model import StableDiffusionModel
from torch.optim.lr_scheduler import LambdaLR

class Pipeline:
    def __init__(self, args):
        self.args = args
        self.args.expStartIter = self.args.expStartIter or 0
        self.args.expEndIter = self.args.expEndIter or self.args.iters
        if self.args.progressiveView:
            if not self.args.dontOverrideTRange:
                self.args.jitterPose = False
            self.args.uniformSphereRate = 0
            self.args.fullRadiusRange = self.args.radiusRange
            self.args.fullThetaRange = self.args.thetaRange
            self.args.fullPhiRange = self.args.phiRange
            self.args.fullFovyRange = self.args.fovyRange
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if self.args.seed is not None:
            seeder(int(self.args.seed))
        for key, value in self.args.__dict__.items():
            print(f"{key}: {value}")
    
    def loadData(self, type = "train"):
        if type == "train":
            return Dataset(
                self.args, device = self.device,
                type = type, H = self.args.h, W = self.args.w,
                size = self.args.datasetSizeTrain * self.args.batchSize
            ).dataLoader()
        elif type == "val":
            return Dataset(
                self.args, device = self.device,
                type = type, H = self.args.H, W = self.args.W,
                size = self.args.datasetSizeValid
            ).dataLoader(batchSize = 1)
        elif type == "test":
            return Dataset(
                self.args, device = self.device,
                type = type, H = self.args.H, W = self.args.W,
                size = self.args.datasetSizeTest
            ).dataLoader(batchSize = 1)
        else:
            raise ValueError
    
    def initiateModel(self):
        model = NeRF(self.args).to(self.device)
        print(model)
        if self.args.optim == "adan":
            self.optimiser = lambda model: Adan(
                model.getParams(5 * self.args.lr),
                eps = 1e-8, weight_decay = 2e-5,
                max_grad_norm = 5.0, foreach = False
            )
        else:
            self.optimiser = lambda model: Adam(
                model.getParams(self.args.lr),
                betas = (0.9, 0.99), eps = 1e-15
            )
        self.scheduler = lambda optimiser: LambdaLR(
            optimiser, lambda iter: 1
        )
        return model
    
    def initiateGuidance(self):
        return StableDiffusionModel(
            device = self.device,
            fp16 = self.args.fp16,
            version = self.args.sdVersion,
            hfModelKey = self.args.hfModelKey,
            tRange = self.args.tRange
        )
    
    def run(
            self, model, guidance,
            trainLoader, valLoader,
            testLoader
    ):
        trainer = Trainer(
            expName = self.args.expName,
            args = self.args,
            model = model,
            guidance = guidance,
            device = self.device,
            workspace = f"outputs/{self.args.workspace}",
            optimiser = self.optimiser,
            emaDecay = self.args.emaDecay,
            fp16 = self.args.fp16,
            lrScheduler = self.scheduler,
            schedulerUpdateEveryStep = self.args.schedulerUpdateEveryStep
        )
        trainer.default_view_data = trainLoader._data.getDefaultViewData()
        maxEpochs = np.ceil(self.args.iters / len(trainLoader)).astype(np.int32)
        trainer.train(
            trainLoader = trainLoader,
            validLoader = valLoader,
            testLoader = testLoader,
            maxEpochs = maxEpochs
        )
        if self.args.saveMesh:
            trainer.saveMesh()
        
    def __call__(self):
        trainLoader = self.loadData(type = "train")
        valLoader = self.loadData(type = "val")
        testLoader = self.loadData(type = "test")
        model = self.initiateModel()
        guidance = self.initiateGuidance()
        self.run(
            model = model,
            guidance = guidance,
            trainLoader = trainLoader,
            valLoader = valLoader,
            testLoader = testLoader
        )