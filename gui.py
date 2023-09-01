import gradio as gr
from utils.args import Args
from NeRF.pipe import Pipeline

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
    """
    Main function to running the ImaginFusion Application.

    Args:
        posPrompt (str): Positive prompt for ImaginFusion.
        workspace (str): Workspace name for saving results.
        sdVersion (str): Stable Diffusion version
        hfModelKey (str): HuggingFace model key for Stable Diffusion
        fp16 (bool): Use mixed precision for training.
        seed (int): Seed value for reproducibility.
        iters (int): Number of iterations.
        lr (float): Learning Rate.
        lambdaEntropy (float): Loss scale foe alpha entropy.
        maxSteps (int): Maximum number of steps sampled per ray.
        h (int): Render height for NeRF training.
        w (int): Render width for NeRF training.
        datasetSizeTrain (int): Size of training dataset.
        datasetSizeValid (int): Size of validation dataset.
        datasetSizeTest (int): Size of test dataset.

    Returns:
        str: Path to generated 3D model.
    """
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

    # Initialising & running the pipeline with Args object
    Pipeline(args)()

    # Return the path to the generated 3D model
    return f'outputs/{workspace}/mesh/Mesh.obj'

# Creating an instance of Gradio Interface
app = gr.Interface(
    fn = main,
    # Defining the input components for the GUI
    inputs = [
        # Textbox for Positive Prompt
        gr.Textbox(
            label = "Positive Prompt"
        ),
        # Textbox for Workspace
        gr.Textbox(
            label = "Workspace",
            info = "Name of the workspace (output folder)"
        ),
        # Dropdown for selecting Stable Diffusion version. Defaults to "1.5"
        gr.Dropdown(
            label="Stable Diffusion Version",
            choices=["1.4", "1.5", "2.1", "2.1-base"],
            value = "1.5"
        ),
        # Textbox for HuggingFace model key for Stable Diffusion
        gr.Textbox(
            label = "Stable Diffusion HF Model Key"
        ),
        # Checkbox for enabling Mixed Precision
        gr.Checkbox(
            label = "Mixed Precision",
            value= True, # Initially checked
            info = "Using mixed precision to reduce training time and precision"
        ),
        # Textbox for seed value
        gr.Textbox(
            label = "Seed",
            value = None,
            info = "Seed Value for Reproducibility"
        ),
        # Number box for number of iteration
        gr.Number(
            label = "Number of iterations",
            value = 5000 # Default value
        ),
        # Dropdown for Learning rate
        gr.Dropdown(
            label = "Learning Rate",
            choices = ["1e-3", "7.75e-4", "5.5e-4", "1e-4"],
            value = '1e-3', # Default choice
            info = "1e-4 works better with higher iterations, 7.75e-4 & 5.5e-4 should work better for more complex shapes."
        ),
        # Textbox for Lambda Entropy
        gr.Textbox(
            label = "Lambda Entropy",
            value = "1e-4", # Default value
            info = "Loss Scale for Alpha Entropy"
        ),
        # Number box for Max Steps
        gr.Number(
            label = "Maximum Steps",
            value = 512, # Default value
            info = "Maximum No. of Steps Sampled per Ray"
        ),
        # Number box for Training height
        gr.Number(
            label = "Training Height",
            value = 64, # Default value
            info = "Render Height for NeRF Training"
        ),
        # Number box for Training height
        gr.Number(
            label = "Training Width",
            value = 64, # Default value
            info = "Render Width for NeRF Training"
        ),
        # Number box for training dataset file
        gr.Number(
            label = "Training Size",
            value = 100, # Default value
            info = "Size of the Tarining Dataset"
        ),
        # Number box for validation dataset file
        gr.Number(
            label = "Validation Size",
            value = 8, # Default value
            info = "Size of the Validation Dataset"
        ),
        # Number box for test dataset file
        gr.Number(
            label = "Test Size",
            value = 100, # Default value
            info = "Size of Test Dataset"
        )
    ],
    # Defining output block
    outputs = gr.Model3D(
        label = "3D Model"
    ),
    # Setting theme for interface
    theme = gr.themes.Soft(),
    # Defining the title, description, & article for the interface
    title = "ImaginFusion",
    description = "Text-To-3D Model",
    article = "A Text-To-3D Model Implementation based on Dreamfusion paper, with Stable-Diffusion as Image Generator and TinyCudaNN based InstantNGP NeRF.",
    # Setting flagging of content to "never"
    allow_flagging = "never"
)

# Launching the Gradio interface when the script is run.
if __name__ == "__main__":
    app.launch()