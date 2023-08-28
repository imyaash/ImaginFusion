import torch
import numpy as np
from .model import NeRF
from .data import Dataset
from .trainer import Trainer
from utils.optimiser import Adan
from utils.functions import seeder
from sdm.model import StableDiffusionModel

class Pipeline:
    def __init__(
            self,
            args
    ):
        self.args = args
        self.args.testInterval = max(int((self.args.iters / self.args.datasetSizeTrain) / 5), 10)
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
        self.args.maxEpochs = np.ceil(self.args.iters / (self.args.datasetSizeTrain * self.args.batchSize)).astype(np.int32)
        for key, value in self.args.__dict__.items():
            print(f"{key}: {value}")
        if self.args.seed is not None:
            seeder(int(self.args.seed))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def loadData(self, type = "train"):
        if type == "train":
            return Dataset(
                args = self.args,
                device = self.device,
                type = type,
                H = self.args.h,
                W = self.args.w,
                size = self.args.datasetSizeTrain * self.args.batchSize
            ).dataLoader()
        elif type == "val":
            return Dataset(
                args = self.args,
                device = self.device,
                type = type,
                H = self.args.H,
                W = self.args.W,
                size = self.args.datasetSizeValid
            ).dataLoader(batchSize = 1)
        elif type == "test":
            return Dataset(
                args = self.args,
                device = self.device,
                type = type,
                H = self.args.H,
                W = self.args.W,
                size = self.args.datasetSizeTest
            ).dataLoader(batchSize = 1)
    
    def initiateNeRF(self):
        model = NeRF(self.args).to(self.device)
        print(model)
        if self.args.optim == "adan":
            self.optimiser = lambda model: Adan(
                model.get_params(5 * self.args.lr),
                eps = 1e-8, weight_decay = 2e-5,
                max_grad_norm = 5.0, foreach = False
            )
        else:
            self.optimiser = lambda model: torch.optim.Adam(
                model.get_params(self.args.lr),
                betas = (0.9, 0.99), eps = 1e-15
            )
        self.scheduler = lambda optimiser: torch.optim.lr_scheduler.LambdaLR(optimiser, lambda iter: 1)
        return model
    
    def initaiteGuidance(self):
        return StableDiffusionModel(
            device = self.device,
            fp16 = self.args.fp16,
            version = self.args.sdVersion,
            hfModelKey = self.args.hfModelKey,
            tRange = self.args.tRange
        )
    
    def trainNeRF(self, model, guidance, trainLoader, valLoader, testLoader):
        trainer = Trainer(
            args = self.args,
            expName = self.args.expName,
            model = model,
            guidance = guidance,
            device = self.device,
            workspace = f"outputs/{self.args.workspace}",
            optimiser = self.optimiser,
            emaDecay = self.args.emaDecay,
            fp16 = self.args.fp16,
            lrScheduler = self.scheduler,
            schedulerUpdateEveryStep = True,
            useTensorboardX = False
        )
        trainer.default_view_data = trainLoader._data.getDefaultViewData()
        trainer.train(
            trainLoader = trainLoader,
            validLoader = valLoader,
            testLoader = testLoader,
            maxEpochs = self.args.maxEpochs
        )
        if self.args.saveMesh:
            trainer.saveMesh()
    
    def __call__(self):
        trainLoader = self.loadData("train")
        valLoader = self.loadData("val")
        testLoader = self.loadData("test")
        model = self.initiateNeRF()
        guidance = self.initaiteGuidance()
        self.trainNeRF(
            model = model,
            guidance = guidance,
            trainLoader = trainLoader,
            valLoader = valLoader,
            testLoader = testLoader
        )