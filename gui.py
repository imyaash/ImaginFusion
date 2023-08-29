from utils.args import Args
from NeRF.pipe import Pipeline

import gradio as gr

def main(
    posPrompt,
    workspace,
    sdVersion,
    hfModelKey,
    fp16,
    seed,
    iters,
    lr,
    lambdaEntropy,
    maxSteps,
    h,
    w,
    datasetSizeTrain,
    datasetSizeValid,
    datasetSizeTest
):
    args = Args(
        posPrompt = posPrompt,
        workspace = workspace,
        sdVersion = sdVersion,
        hfModelKey = hfModelKey if hfModelKey != "" else None,
        fp16 = fp16,
        seed = int(seed) if seed != "" else None,
        iters = int(iters),
        lr = float(lr),
        lambdaEntropy = float(lambdaEntropy),
        maxSteps = int(maxSteps),
        h = int(h),
        w = int(w),
        datasetSizeTrain = int(datasetSizeTrain),
        datasetSizeValid = int(datasetSizeValid),
        datasetSizeTest = int(datasetSizeTest),
    )

    Pipeline(args)()

    return f'outputs/{workspace}/mesh/Mesh.obj'

app = gr.Interface(
    fn = main,
    inputs = [
        gr.Textbox(
            label = "Positive Prompt"
        ),
        gr.Textbox(
            label = "Workspace",
            info = "Name of the workspace (output folder)"
        ),
        gr.Dropdown(
            label="Stable Diffusion Version",
            choices=["1.4", "1.5", "2.1", "2.1-base"],
            value = '1.5'
        ),
        gr.Textbox(
            label = "Stable Diffusion HF Model Key"
        ),
        gr.Checkbox(
            label = "Mixed Precision",
            value= True,
            info = "Using mixed precision to reduce training time and precision"
        ),
        gr.Textbox(
            label = "Seed",
            value = None,
            info = "Seed Value for Reproducibility"
        ),
        gr.Number(
            label = "Number of iterations",
            value = 5000
        ),
        gr.Dropdown(
            label = "Learning Rate",
            choices = ["1e-3", "7.75e-4", "5.5e-4", "1e-4"],
            value = '1e-3',
            info = "1e-4 works better with higher iterations, 7.75e-4 & 5.5e-4 should work better for more complex shapes."
        ),
        gr.Textbox(
            label = "Lambda Entropy",
            value = "1e-4",
            info = "Loss Scale for Alpha Entropy"
        ),
        gr.Number(
            label = "Maximum Steps",
            value = 512,
            info = "Maximum No. of Steps Sampled per Ray"
        ),
        gr.Number(
            label = "Training Height",
            value = 64,
            info = "Render Height for NeRF Training"
        ),
        gr.Number(
            label = "Training Width",
            value = 64,
            info = "Render Width for NeRF Training"
        ),
        gr.Number(
            label = "Training Size",
            value = 100,
            info = "Size of the Tarining Dataset"
        ),
        gr.Number(
            label = "Validation Size",
            value = 8,
            info = "Size of the Validation Dataset"
        ),
        gr.Number(
            label = "Test Size",
            value = 100,
            info = "Size of Test Dataset"
        )
    ],
    outputs = gr.Model3D(
        label = "3D Model"
    ),
    theme = gr.themes.Soft(),
    title = "ImaginFusion",
    description = "Text-To-3D Model",
    article = "A Text-To-3D Model Implementation based on Dreamfusion paper, with Stable-Diffusion as Image Generator and TinyCudaNN based InstantNGP NeRF.",
    allow_flagging = "never"
)

if __name__ == "__main__":
    app.launch()