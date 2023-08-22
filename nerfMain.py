import numpy as np
from NeRF.args import Args
import torch
from NeRF.provider import NeRFDataset
from NeRF.network import NeRFNetwork
from NeRF.utils import seeder, NeRFTrainer
from utils.optimizer import Adan
from sdm.model import StableDiffusionModel

args = Args(
    posPrompt = "ultra-realistic, delicious hamburger",
    workspace = "testHamburger",
    fp16 = True,
    seed = 0,
    iters = 5000,
    lr = 7.75e-4, # trying for speeds sake, good result on simple shaped object
    # lr = 1e-4, # for slower but better performance, useless for complex and intricate objects takes too long to learn
    # lr = 1e-3, # is the original
    # lr = 5.5e-4, # seems to be a good middle ground
    lambdaEntropy = 1e-4,
    maxSteps = 512,
    h = 64, 
    w = 64,
    writeVideo = True,
    datasetSizeTrain = 20,
    datasetSizeValid = 8,
    datasetSizeTest = 100,
    testInterval = 50
)

args.cuda_ray = True

args.expStartIter = args.expStartIter or 0
args.expEndIter = args.expEndIter or args.iters

args.images = None if len(args.images) == 0 else args.images

if args.progressiveView:
    if not args.dontOverrideTRange:
        args.jitterPose = False
    args.uniformSphereRate = 0
    args.fullRadiusRange = args.radiusRange
    args.fullThetaRange = args.thetaRange
    args.fullPhiRange = args.phiRange
    args.fullFovyRange = args.fovyRange

for key, value in args.__dict__.items():
    print(f"{key}: {value}")

if args.seed is not None:
    seeder(int(args.seed))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeRFNetwork(args).to(device)

print(model)

trainLoader = NeRFDataset(
    args, device = device, type = "train",
    H = args.h, W = args.w, size = args.datasetSizeTrain * args.batchSize
).dataloader()

if args.optim == "adan":
    optimiser = lambda model: Adan(
        model.get_params(5 * args.lr),
        eps = 1e-8, weight_decay = 2e-5,
        max_grad_norm = 5.0, foreach = False
    )
else:
    optimiser = lambda model: torch.optim.Adam(
        model.get_params(args.lr),
        betas = (0.9, 0.99), eps = 1e-15
    )
scheduler = lambda optimiser: torch.optim.lr_scheduler.LambdaLR(optimiser, lambda iter: 1)

guidance = StableDiffusionModel(
    device = device, fp16 = args.fp16, version = args.sdVersion,
    hfModelKey = args.hfModelKey, tRange = args.tRange
)

trainer = NeRFTrainer(
    expName = args.expName,
    opt = args,
    model = model,
    guidance = guidance,
    device = device,
    workspace = f"outputs/{args.workspace}",
    optimiser = optimiser,
    emaDecay = args.emaDecay,
    fp16 = args.fp16,
    lrScheduler = scheduler,
    schedulerUpdateEveryStep = True,
    useTensorboardX = True
)
trainer.default_view_data = trainLoader._data.get_default_view_data()
validLoader = NeRFDataset(args, device = device, type = "val", H = args.H, W = args.W, size = args.datasetSizeValid).dataloader(batch_size = 1)
testLoader = NeRFDataset(args, device = device, type = "test", H = args.H, W = args.W, size = args.datasetSizeTest).dataloader(batch_size = 1)
maxEpoch = np.ceil(args.iters / len(trainLoader)).astype(np.int32)
trainer.train(
    trainLoader = trainLoader,
    validLoader = validLoader,
    testLoader = testLoader,
    maxEpochs = maxEpoch
)
if args.saveMesh:
    trainer.saveMesh()