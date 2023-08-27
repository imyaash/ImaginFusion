import argparse
from utils.args import Args
from NeRF.pipe import Pipeline

def main():
    parser = argparse.ArgumentParser(description = "ImaginFusionCLI")
    parser.add_argument("posPrompt", type = str, help = "Positive prompt for ImaginFusion")
    parser.add_argument("workspace", type = str, help = "Workspace for saving results")
    parser.add_argument("--fp16", action = "store_true", help = "Use fp16")
    parser.add_argument("--seed", type = int, default = None, help = "Seed value")
    parser.add_argument("--iters", type = int, default = 5000, help = "Number of iterations")
    parser.add_argument("--lr", type = float, default = 7.75e-4, help = "Learning rate")
    parser.add_argument("--lambdaEntropy", type = float, default = 1e-4, help = "Lambda entropy")
    parser.add_argument("--maxSteps", type = int, default = 512, help = "Max steps")
    parser.add_argument("--h", type = int, default = 64, help = "Height")
    parser.add_argument("--w", type = int, default = 64, help = "Width")
    parser.add_argument("--writeVideo", action = "store_true", help = "Write video")
    parser.add_argument("--datasetSizeTrain", type = int, default = 20, help = "Dataset size for training")
    parser.add_argument("--datasetSizeValid", type = int, default = 8, help = "Dataset size for validation")
    parser.add_argument("--datasetSizeTest", type = int, default = 100, help = "Dataset size for testing")
    parsedArgs = parser.parse_args()
    args = Args(
        posPrompt = parsedArgs.posPrompt,
        workspace = parsedArgs.workspace,
        fp16 = parsedArgs.fp16,
        seed = parsedArgs.seed, # Default None
        iters = parsedArgs.iters,
        lr = parsedArgs.lr,
        # lr = 7.75e-4, # trying for speeds sake, good result on simple shaped object
        # lr = 1e-4, # for slower but better performance, useless for complex and intricate objects takes too long to learn
        # lr = 1e-3, # is the original
        # lr = 5.5e-4, # seems to be a good middle ground
        # lr = 7.75e-5,
        lambdaEntropy = parsedArgs.lambdaEntropy,
        maxSteps = parsedArgs.maxSteps,
        h = parsedArgs.h,
        w = parsedArgs.w,
        writeVideo = parsedArgs.writeVideo,
        datasetSizeTrain = parsedArgs.datasetSizeTrain,
        datasetSizeValid = parsedArgs.datasetSizeValid,
        datasetSizeTest = parsedArgs.datasetSizeTest
    )
    Pipeline(args)()

if __name__ == "__main__":
    main()