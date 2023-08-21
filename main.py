from NeRF.pipe import Pipeline
from utils.args import Args

args = Args(
    posPrompt = "ultra-realistic, classic coca-cola bottle",
    workspace = "testCoke",
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

pipe = Pipeline(args)
pipe()