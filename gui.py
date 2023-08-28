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
        posPrompt=posPrompt,
        workspace=workspace,
        sdVersion=sdVersion,
        hfModelKey=hfModelKey if hfModelKey != "" else None,
        fp16=fp16,
        seed=int(seed),
        iters=int(iters),
        lr=float(lr),
        lambdaEntropy=float(lambdaEntropy),
        maxSteps=int(maxSteps),
        h=int(h),
        w=int(w),
        datasetSizeTrain=int(datasetSizeTrain),
        datasetSizeValid=int(datasetSizeValid),
        datasetSizeTest=int(datasetSizeTest),
    )

    # Pipeline(args)()

    return f'outputs/{workspace}/mesh/Mesh.obj'

app = gr.Interface(
    fn = main,
    inputs = [
        gr.Textbox(label = "Positive prompt", info = "Initial Prompt"),
        gr.Textbox(label = "Workspace", info = "Name of the workspace (output folder)"),
        gr.Dropdown(
            label="Stable diffusion version",
            choices=["1.4", "1.5", "2.1", "2.1-base"],
            value = '1.5'
        ),
        gr.Textbox(label="Stable diffusion HF model key", value=""),
        gr.Checkbox(label="Mixed precision", value=True, info = "Using mixed precision to reduce training time and precision"),
        gr.Number(label="Seed", value=0),
        gr.Number(label="Number of iterations", value=5000),
        gr.Dropdown(
            label="Learning rate", choices=["1e-3", "7.75e-4", "5.5e-4", "1e-4"], value='7.75e-4'
        ),
        gr.Textbox(label="Lambda entropy", value="1e-4"),
        gr.Number(label="Number of ray marching steps", value=512),
        gr.Number(label="Training height", value=64),
        gr.Number(label="Training width", value=64),
        gr.Number(label="Training dataset size", value=100),
        gr.Number(label="Validation dataset size", value=8),
        gr.Number(label="Test dataset size", value=100)
    ],
    outputs=gr.Model3D(
        label= "Output mesh"
    ),
    theme= gr.themes.Base()
)

if __name__ == "__main__":
    app.launch()
