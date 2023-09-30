# ImaginFusion Documentation

ImaginFusion is an application to generate 3D models based on natural language prompts. It is based on DreamFusion, but instead of using Imagen and Mip-NeRF to generate 2D priors and 3D synthesis respectively, uses Stable Diffusion and torch-ngp. It includes two interfaces, a CLI & a GUI, for better accessibility. The installation and usage instructions are in the [readme](readme.md) file.

## Table of Content
- [Key External Libraries](#key-external-libraries)
- [Key Internal Modules/Functions](#key-internal-modulesfunctions)
    - [Command Line Interface/Graphical User Interface](#command-line-interfacegraphical-user-interface)
        - [Modules Used](#modules-used)
        - [User Inputs](#user-inputs)
        - [Returns](#returns)
    - [Args](#args)
        - [Properties](#properties)
    - [Pipeline](#pipeline)
        - [Modules Used](#modules-used-1)
        - [Properties](#properties-1)
        - [Methods](#methods)
    - [Trainer](#trainer)
        - [Modules Used](#modules-used-2)
        - [Properties](#properties-2)
        - [Methods](#methods-1)
    - [Dataset](#dataset)
        - [Modules Used](#modules-used-3)
        - [Properties](#properties-3)
        - [Methods](#methods-2)
    - [Renderer](#renderer)
        - [Modules Used](#modules-used-4)
        - [Properties](#properties-4)
        - [Methods](#methods-3)
    - [NeRF](#nerf)
        - [Modules Used](#modules-used-5)
        - [Properties](#properties-5)
        - [Methods](#methods-4)
    - [StableDiffusionModel](#stablediffusionmodel)
        - [Properties](#properties-6)
        - [Methods](#methods-6)
    - [encoder](#encoder)
    - [TruncExp](#truncexp)
    - [softplus](#softplus)
    - [meshDecimator](#meshdecimator)
    - [meshCleaner](#meshcleaner)
    - [getViewDirections](#getviewdirections)
    - [customMeshGrid](#custommeshgrid)
    - [normalise](#normalise)
    - [getRays](#getrays)
    - [getCPUMem](#getcpumem)
    - [getGPUMem](#getgpumem)
    - [circlePoses](#circleposes)
    - [randPoses](#randposes)

## Key External Libraries
- [Tiny CUDA Neural Network Framework](https://github.com/NVlabs/tiny-cuda-nn)
- [torch-ngp: A PyTorch implementation of instant-ngp](https://github.com/ashawkey/torch-ngp)
- [Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild](https://github.com/gradio-app/gradio)
- [Adan: A PyTorch implementation of Adaptive Nesterov Momentum Algorithm for Faster Optimizing by Deep Models](https://github.com/sail-sg/Adan)

## Key Internal Modules/Functions
### [Command Line Interface](cli.py)/[Graphical User Interface](gui.py)

These are the Entry Point for ImaginFusion application.

#### Modules Used:
- [Args](utils\args.py)
- [Pipeline](NeRF\pipe.py)

#### User Inputs:
- posPrompt (str): Positive prompt for ImaginFusion.
- workspace (str): Workspace name for saving results.
- sdVersion (str): Stable Diffusion version
- hfModelKey (str): HuggingFace model key for Stable Diffusion
- fp16 (bool): Use mixed precision for training.
- seed (int): Seed value for reproducibility.
- iters (int): Number of iterations.
- lr (float): Learning Rate.
- lambdaEntropy (float): Loss scale for alpha entropy.
- maxSteps (int): Maximum number of steps sampled per ray.
- h (int): Render height for NeRF training.
- w (int): Render width for NeRF training.
- datasetSizeTrain (int): Size of training dataset.
- datasetSizeValid (int): Size of validation dataset.
- datasetSizeTest (int): Size of test dataset.

#### Returns:
- 3D Model (mesh.obj, mesh.mtl files and albedo.png)
- 360&deg; Video

### [Args](utils\args.py)

Stores & manages all the parameters & hyperparameters for the application. All the user inputs are initially passed here before the entire Args object is passed to the Pipeline.

#### Properties:
- posPrompt (str): A positive text prompt.
- negPrompt (str): A negative text prompt.
- expName (str): Experiment name.
- workspace (str): Workspace directory.
- fp16 (bool): Whether to use FP16 precision.
- seed (int): Random seed for reproducibility.
- sdVersion (str): Stable Diffusion version.
- hfModelKey: High-frequency model key.
- evalInterval (int): Number of training iterations between evaluations on the validation set.
- testInterval (int): Number of training iterations between testing on the test set.
- guidanceScale (int): Guidance scale for stable diffusion.
- saveMesh (bool): Whether to save the mesh.
- mcubesResolution (int): Resolution for extracting the mesh.
- decimateTarget (float): Target for mesh decimation.
- iters (int): Number of training iterations.
- lr (float): Maximum learning rate.
- maxSteps (int): Maximum number of steps sampled per ray.
- updateExtraInterval (int): Iteration interval to update extra status.
- latentIterRatio (float): Ratio of latent iterations.
- albedoIterRatio (float): Ratio of albedo iterations.
- minAmbientRatio (float): Minimum ambient ratio.
- texturelessRatio (float): Textureless ratio.
- jitterPose (bool): Adding jitter to randomly sampled camera poses.
- jitterCentre (float): Amount of jitter to add to sampled camera pose's center.
- jitterTarget (float): Amount of jitter to add to sampled camera pose's target.
- jitterUp (float): Amount of jitter to add to sampled camera pose's up-axis.
- uniformSphereRate (float): Probability of sampling camera location uniformly.
- gradClip (float): Clip grad for all gradients.
- gradClipRGB (float): Clip grad of RGB space grad.
- bgRadius (float): Radius of the background sphere.
- densityActivation (str): Density activation function ("exp" or "softplus").
- densityThresh (float): Threshold for density grid to be occupied.
- blobDensity (float): Max density for density blob.
- blobRadius (float): Controlling the radius for density blob.
- optim (str): Optimization function.
- w (int): Render width for training NeRF.
- h (int): Render height for training NeRF.
- knownViewScale (float): Multiply h/w by this for known view rendering.
- batchSize (int): Number of images to be rendered per batch.
- bound (int): Assume the scene is bounded in box(-bound, bound)x.
- dtGamma (float): dt_gamma (>=0) for adaptive ray marching. Set to 0 to disable, >0 to accelerate rendering (but usually with worse quality).
- minNear (float): Minimum near distance for the camera.
- radiusRange (list): Training camera radius range.
- thetaRange (list): Training camera along the polar axis (up-down).
- phiRange (list): Training camera along the azimuth axis (left-right).
- fovyRange (list): Training camera fovy range.
- defaultRadius (float): Radius for the default view.
- defaultPolar (float): Polar for the default view.
- defaultAzimuth (float): Azimuth for the default view.
- defaultFovy (float): Fovy for the default view.
- progressiveView (bool): Progressively expand view sampling range from default to full.
- progressiveViewInitRatio (float): Initial ratio of the final range.
- progressiveLevel (bool): Progressively increase grid encoder's max level.
- angleOverhead (float): Overhead angle.
- angleFront (float): Front angle.
- tRange (list): Range for t values.
- dontOverrideTRange (bool): Whether to override t range.
- lambdaEntropy (float): Loss scale for alpha entropy.
- lambdaOpacity (float): Loss scale for alpha value.
- lambdaOrient (float): Loss scale for orientation.
- lambdaGuidance (float): Loss scale for guidance.
- lambdaNormal (float): Loss scale for normal map.
- lambda2dNormalSmooth (float): Loss scale for 2D normal image smoothness.
- lambda3dNormalSmooth (float): Loss scale for 3D normal image smoothness.
- H (int): Mesh height for validation.
- W (int): Mesh width for validation.
- datasetSizeTrain (int): Length of the training dataset.
- datasetSizeValid (int): Number of frames to render in the turntable video during validation.
- datasetSizeTest (int): Number of frames to render in the turntable video during test time.
- expStartIter (int): Start iteration for experiment.
- expEndIter (int): End iteration for experiment.
- writeVideo (bool): Whether to write video during testing.
- emaDecay (float): Exponential moving average decay for training NeRF.
- schedulerUpdateEveryStep (bool): Update scheduler every training step.
- refRadii (list): Reference radii.
- refPolars (list): Reference polar angles.
- refAzimuths (list): Reference azimuth angles.

### [Pipeline](NeRF\pipe.py)

The class for managing the entire training pipeline. In encompasses various stages of the training process, including data loading, model initialisation, and training.

#### Modules Used:
- [NeRF](NeRF\model.py)
- [Dataset](NeRF\data.py)
- [Trainer](NeRF\trainer.py)
- [seeder](utils\functions.py)
- [StableDiffusionModel](sdm\model.py)

#### Properties:
- args (object): A configuration object containing various parameters for the pipeline.
- device: The computing device (CPU or GPU) used for training.

#### Methods:
- loadData: Load the dataset for training, validation and testing.
    - Inputs:
        - type (str, optional): The type of dataset to load ("train", "val", or "test"). Defaults to "train".
    - Returns:
        - DataLoader: A PyTorch DataLoader object containing the loaded dataset.
- InitiateNeRF: Initialises the NeRF model.
    - Returns:
        - NeRF: An instance of NeRf model.
- InitiateGuidance: Initialises the guidance model.
    - Returns:
        - StableDiffusionModel: An instance of guidance model.
- trainNeRF: Train the NeRF model.
    - Inputs:
        - model (NeRF): The NeRF model to be trained.
        - guidance (StableDiffusionModel): The guidance model.
        - trainLoader (DataLoader): DataLoader for training data.
        - valLoader (DataLoader): DataLoader for validation data.
        - testLoader (DataLoader): DataLoader for testing data.
- Pipeline: Starts the training pipeline by loading data, initialising models and training.

### [Trainer](NeRF\trainer.py)

The class for training, evaluation & testing the Text-to-3D model.

#### Modules Used:
- [getCPUMem](utils\functions.py)
- [getGPUMem](utils\functions.py)

#### Properties:
- args (object): Arguments for training.
- model (nn.Module): The neural network model.
- guidance (object): Guidance for training.
- expName (str): Experiment name.
- criterion (nn.Module, optional): Loss function. Default is None.
- optimiser (callable, optional): Optimizer for model training. Default is None.
- lrScheduler (callable, optional): Learning rate scheduler. Default is None.
- emaDecay (float, optional): Exponential moving average decay rate. Default is None.
- metrics (list, optional): List of metrics for evaluation. Default is an empty list.
- device (str, optional): Device for training (CPU or GPU). Default is None.
- verbose (bool, optional): Whether to print verbose output. Default is True.
- fp16 (bool, optional): Whether to use mixed-precision training. Default is False.
- workspace (str, optional): Workspace directory for saving logs and checkpoints. Default is "workspace".
- bestMode (str, optional): Best mode for selecting checkpoints (min or max). Default is "min".
- useLossAsMetric (bool, optional): Whether to use loss as a metric. Default is True.
- reportMetricAtTraining (bool, optional): Whether to report metrics during training. Default is False.
- useTensorboardX (bool, optional): Whether to use TensorboardX for logging. Default is True.
- schedulerUpdateEveryStep (bool, optional): Whether to update the learning rate scheduler at every step. Default is False.

#### Methods:
- prepareEmbeddings: Prepares text embeddings during training.
- log: Logs messages to a file.
    - Inputs:
        - args: Variable-length argument list.
        - kwargs: Arbitrary keyword arguments.
- train_step: Performs a single training step.
    - Inputs:
        - data (dict): Training data.
    - Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Predicted RGB and depth values, and training loss.
- post_train_step: Perform post-training step actions like gradient scaling and clipping.
- eval_step: Performs a single evaluation step.
    - Inputs:
        - data (dict): Evaluation data.
    - Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Predicted RGB and depth values, and evaluation loss.
- test_step: Performs a single test step.
    - Inputs:
        - data (dict): Testing data.
        - bgColor (torch.Tensor, optional): Background colour. Defaults to None.
        - perturb (bool, optional): Whether to perturb the rendering. Defaults to False.
    - Returns:
        - Tuple[torch.Tensor, torch.Tensor, None]: Predicted RGB and depth values, and placeholder.
- saveMesh: Saves a 3D mesh representation of the test predictions.
    - Inputs:
        - path (str, optional): Path to save the mesh. Defaults to None.
- trainOneEpoch: Performs one epoch of the training.
    - Inputs:
        - loader (torch.utils.data.DataLoader): DataLoader for training data.
        - maxEpochs (int): Maximum number of epochs.
- evaluateOneEpoch: Performs one epoch of the evaluation.
    - Inputs:
        - loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        - name (str, optional): Name for the evaluation. Defaults to None.
- test: Performs prediction.
    - Inputs:
        - loader (torch.utils.data.DataLoader): DataLoader for testing data.
        - savePath (str, optional): Path to save test results. Defaults to None.
        - name (str, optional): Name for the test. Defaults to None.
        - writeVideo (bool, optional): Whether to write test results as video. Defaults to True.
- train: Performs training.
    - Inputs:
        - trainLoader (torch.utils.data.DataLoader): DataLoader for training data.
        - validLoader (torch.utils.data.DataLoader): DataLoader for validation data.
        - testLoader (torch.utils.data.DataLoader): DataLoader for testing data.
        - maxEpochs (int): Maximum number of epochs.
- evaluate: Performs evaluation.
    - Input:
        - loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        - name (str, optional): Name for the evaluation. Defaults to None.

### [Dataset](NeRF\data.py)

The class for managing the data used in training and inference, including camera poses, directions, intrinsics, and more & and creates DataLoader object.

#### Modules Used:
- [circlePose](utils\functions.py)
- [randPose](utils\functions.py)
- [getRays](utils\functions.py)

#### Properties:
- args (object): A configuration object.
- device (str): The device on which to perform computation.
- type (str, optional): The datase type, either "train" or "all". Defaults to "train".
- H (int, optional): Height of the image. Defaults to 256.
- W (int, optional): Width of the image. Defaults to 256.
- size (int, optional): Size of the dataset. Defaults to 100.

#### Methods:
- collateFn: Collate function for creating batches of data.
    - Inputs:
        - idx (list): List of indices to select data for batch.
    - Returns:
        - dict: A dictionary containing batched data.
- dataLoader: Creates a DataLoader for the dataset.
    - Inputs:
        - batchSize (int, optional): The batch size. Defaults to None. If not provided, use the default batch size from args.
    - Returns:
        - DataLoader: A DataLoader object for the dataset.
- getDefaultViewData: Get data for default view(s).
    - Returns:
        - dict: A dictionary containing data for default views.

### [Renderer](NeRF\renderer.py)

The class for rendering 3D scenes, conducting raymarching, exporting 3D meshes, and managing density grids.

#### Modules Used:
- [meshDecimator](utils\mesh.py)
- [meshCleaner](utils\mesh.py)
- [customMeshGrid](utils\functions.py)
- [normalise](utils\functions.py)

#### Properties:
- args (dict): Configuration arguments.
- bound (float): The bounding box size.
- cascade (int): Number of cascades.
- gridSize (int): Size of the 3D grid.
- densityT (float): Density threshold.
- aabb_train (torch.Tensor): Training bounding box.
- aabb_infer (torch.Tensor): Inference bounding box.
- glctx: Graphics context for rendering.
- density_grid (torch.Tensor): Density grid for raymarching.
- density_bitfield (torch.Tensor): Bitfield for density grid.
- meanDensity (float): Mean density value.
- iterDensity (int): Iteration count for density updates.

#### Methods:
- densityBlob: Calculate density values for given points.
    - Inputs:
        - x (torch.Tensor): Input points with shape [B, N, 3].
    - Returns:
        - torch.Tensor: Density values for input points.
- forward: Placeholder.
- density: Placeholder.
- resetExtraState: Reset additional state variables.
- exportMesh: Export 3D mesh to a file.
    - Inputs:
        - path (str): Path to save the exported mesh.
        - resolution (int, optional): Resolution for mesh generation. Defaults to None.
        - decimateT (int, optional): Decimation threshold. Defaults to -1.
        - S (int, optional): Split size for mesh grid generation. Defaults to 128.
- run(raysO, raysD, lightD, ambientRatio, shading, bgColor, perturb, tThresh, binarise, **test): Perform raymarching and rendering.
    - Inputs:
        - raysO (torch.Tensor): Ray origins with shape [B, N, 3].
        - raysD (torch.Tensor): Ray directions with shape [B, N, 3].
        - lightD (torch.Tensor, optional): Light directions. Defaults to None.
        - ambientRatio (float, optional): Ambient light ratio. Defaults to 1.0.
        - shading (str, optional): Shadind mode. Defaults to 'albedo'.
        - bgColor (float or torch.Tensor, optional): Background colour. Defaults to None.
        - perturb (bool, optional): Enable ray perturbation. Defaults to False.
        - tThresh (float, optional): Threshold of t. Defaults to 1e-4.
        - binarise (bool, optional): Binarise the output image. Defaults to False.
- updateExtraState: Update additional state variables.
    - Inputs:
        - decay (float, optional): Decay factor for updating the density grid. Defaults to 0.95.
        - S (int, optional): Split size for grid generation. Defaults to 128.
- render(raysO, raysD, **kwargs): Render a scene using raymarching.
    - Inputs:
        - raysO (torch.Tensor): Ray origins with shape [B, N, 3].
        - raysD (torch.Tensor): Ray directions with shape [B, N, 3].
        - **kwargs: Additional arguments passed to the "run" method.
    - Returns:
        - dict: Rendered image, depth & weights.

### [NeRF](NeRF\model.py)

The main instantNGP model (torch-ngp). Inherits Renderer and extends from it.

#### Modules Used:
- [Renderer](NeRF\renderer.py)
- [encoder](utils\encoder.py)
- [normalise](utils\functions.py)
- [truncExp](utils\activator.py)
- [softplus](utils\activator.py)

#### Properties:
- nLayers (int): Number of layers in the NeRF network.
- hiddenDim (int): Hidden dimension of the NeRF network.
- encoder (tcnn.Encoding): TinyCUDA neural network encoder.
- inDim (int): Input dimension of the encoder.
- sigmaNet (Network): NeRF network for predicting sigma and albedo.
- densityActivation (function): Activation function for density prediction.
- nLayersBG (int): Number of layers in the background network.
- hiddenDimBG (int): Hidden dimension of the background network.
- encoderBG (nn.Module): Background encoder.
- inDimBG (int): Input dimension of the background encoder.
- netBG (Network): Background network.

#### Methods:
- forwardC: Forward pass for NeRF color prediction.
    - Inputs:
        - x (torch.Tensor): Input coordinates.
    - Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Sigma and albedo predictions.
- normal: Compute surface normals.
    - Inputs:
        - x (torch.Tensor): Input coordinates.
    - Returns:
        - torch.Tensor: Surface normals.
- forward: Forward pass for NeRF rendering.
    - Inputs:
        - x (torch.Tensor): Input coordinates.
        - d (torch.Tensor): Depth values.
        - l (torch.Tensor, optional): Light direction vectors. Defaults to None.
        - ratio (int, optional): Lambertian ratio. Defaults to 1.
        - shading (str, optional): Shading mode ("albedo", "normal", "textureless"). Defaults to 'albedo'.
    - Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Sigma, colour, and normal predictions.
- density: Predict density values.
    - Inputs:
        - x (torch.Tensor): Input coordinates.
    - Returns:
        - Dict[str, torch.Tensor]: Predicted sigma and albedo.
- background: Predict background values.
    - Args:
        - d (torch.Tensor): Depth values.
    - Returns:
        - torch.Tensor: Predicted background values.
- get_params: Get network parameters and learning rates.
    - Args:
        - lr (float): Learning rate.
    - Returns:
        - List[Dist[str, Union[nn.Parameter, float]]]: List of parameter dictionaries.

### [StableDiffusionModel](sdm\model.py)

An implementation of Stable Diffusion to generate images based on natural language propmts and act as guiding model for NeRF.

#### Properties:
- device (str): The device on which the model is running.
- version (str): The version of the stable diffusion model being used.
- modelPath (str): The path to the pretrained stable-diffusion model.
- precisionT (torch.dtype): The precision type for model tensors.
- vae (nn.Module): The Variational Autoencoder component of the model.
- tokenizer: The text tokenizer used by the model.
- textEncoder: The text encoder used by the model.
- unet: The UNet component of the model.
- scheduler: The diffusion scheduler used by the model.
- numSteps (int): The total number of diffusion steps.
- minSteps (int): The minimum number of diffusion steps in the specified range.
- maxSteps (int): The maximum number of diffusion steps in the specified range.
- alphas (torch.Tensor): The alpha values used in the diffusion process.

#### Methods:
- getTextEmbeddings: Get text embeddings for a given prompt.
    - Inputs:
        - prompt (str): The text prompt for which embeddings are to be generated.
    - Returns:
        - torch.Tensor: Text embeddings dor the input prompt.
- produceLatents: Generate latent vectors.
    - Inputs:
        - embeddings (torch.Tensor): Text embeddings.
        - h (int, optional): Height of the generated image. Defaults to 512.
        - w (int, optional): Width of the generated image. Defaults to 512.
        - numSteps (int, optional): Number of diffusion steps. Defaults to 50.
        - guidanceScale (float, optional): Scaling factor for guidance. Defaults to 7.5.
        - latents (torch.Tensor, optional): Latent vectors. Defaults to None.
    - Returns:
        - torch.Tensor: Generated latent vectors.
- decodeLatents: Decode latent vectors into images.
    - Inputs:
        - latents (torch.Tensor): Latent vectors to be decoded.
    - Returns:
        - torch.Tensor: Decoded images.
- encodeImages: Encode images into latent vectors.
    - Inputs:
        - images (torch.Tensor): Images to be encoded.

    - Returns:
        - torch.Tensor: Encoded latent vectors.
- trainStep: Perform a training step.
    - Inputs:
        - embeddings (torch.Tensor): Text embeddings.
        - predRGB (torch.Tensor): Predicted RGB images.
        - guidanceScale (int, optional): Scaling factor for guidance. Defaults to 100.
        - asLatent (bool, optional): If True, use "predRGB" as latent vectors. Defaults to False.
        - gradScale (int, optional): Scaling factor for gradients. Defaults to 1.
    - Returns:
        - torch.Tensor: Training loss

### [encoder](utils\encoder.py)

Function to create a FreqEncoder instance.

#### Inputs:

- inDim (int, optional): The input dimension. Defaults to 3.
- multiRes (int, optional): The degree of multi-resolution encoding. Defaults to 6.

#### Returns:

-Tuple: A tuple containing the frequency encoder and its output dimension.

### [TruncExp](utils\activator.py)

Custom autograd Function for the truncated exponential operation. This function computes the exponential of input values while clamping the output to a maximum value of 15.

#### Inputs:

- ctx (Context): A PyTorch context object to save intermediate values for backpropagation.
- x (Tensor): The input tensor to compute the truncated exponential for.

#### Returns:

- Tensor: The tensor containing the truncated exponential of the input values.

### [softplus](utils\activator.py)

Fuction to compute biased softplus activation. The softplus function is defined as softplus(x) = ln(1 + exp(x)), and this implementation allows an optional bias to be applied before computing the softplus.

#### Inputs:
- x (Tensor): The input tensor to apply the softplus activation to.
- bias (float): An optional bias value to be subtracted from the input tensor before applying softplus.

#### Returns:
- Tensor: The tensor containing the softplus activations.

### [meshDecimator](utils\mesh.py)

Function to decimate a mesh while preserving it's shape.

#### Inputs:
- vertices (numpy.ndarray): The vertices of the input mesh.
- faces (numpy.ndarray): The faces of the input mesh.
- target (int): The target number of faces after decimation.
- remesh (bool, optional): Whether to remesh the mesh after decimation. Defaults to False.
- optimalPlacement (bool, optional): Whether to use optimal placement during decimatin. Defaults to True.

#### Returns:
- Tuple[numpy.ndarray, numpy.ndarray]: The vertices and faces of the decimated mesh.

### [meshCleaner](utils\mesh.py)

Function to clean and repair 3D mesh.

#### Inputs:
- vertices (numpy.ndarray): The vertices of the input mesh.
- faces (numpy.ndarray): The faces of the input mesh.
- vPct (int, optional): Percentage of close vertices of merge. Defaults to 1.
- minF (int, optional): Minimum number of faces in connected components to keep. Defaults to 8.
- minD (int, optional): Minimum diameter of connected components to keep. Defaults to 5.
- repair (bool, optional): Whether to repair non-manifold edges and vertices. Defaults to True.
- remesh (bool, optional): Whether to remesh the mesh after cleaning. Defaults to True.
- remeshSize (float, optional): Target edge length for remeshing. Defaults to 0.01.

#### Returns:
- Tuple[numpy.ndarray, numpy.ndarray]: The vertices and faces of the cleaned and repaired mesh.

### [getViewDirections](utils\functions.py)

Function to calculate the view direction based on the angles, the thetas, and the phis.

#### Inputs:
- thetas (torch.Tensor): Tensor containing theta angles in radians.
- phis (torch.Tensor): Tensor containing phi angles in radians.
- oHead (float): Angle overhead threshold in radians.
- front (float): Angle front threshold in radians.

#### Returns:
- torch.Tensor: A tensor of integers representing view directions.

### [customMeshGrid](utils\functions.py)

Function to create a mesh grid for given input tensors.

#### Inputs:
- args: Input tensors for which the mesh grid should be created.

#### Returns:
- tuple: A tuple of tensors representing the mesh grid.

### [normalise](utils\functions.py)

Function to normalise a tensor.

#### Inputs:
- x (torch.Tensor): Input tensor.
- eps (float, optional): A small value to prevent division by zero. Defaults to 1e-20.

#### Returns:
- torch.Tensor: normalised tensor.

### [getRays](utils\functions.py)

Function to generate rays based on camera poses and intrinsics.

#### Inputs:
- poses (torch.Tensor): Camera poses.
- intrinsics (tuple): Camera intrinsics (fx, fy, cx, cy)
- H (int): Image height.
- W (int): image width.
- N (int, optional): Number of rays to generate. Defaults to -1, generates all rays.
- errorMap (torch.Tensor, optional): Error map for ray sampling. Defaults to None.

#### Returns:
- dict: A dictionary containing ray information including origins, directions, and indices.

### [seeder](utils\functions.py)

Function to set random seed for Python, NumPy, and PyTorch.

#### Inputs:
- seed (int): Random seed value.

### [getCPUMem](utils\functions.py)

Function to get current memory usage.

#### Returns:
- float: Current CPU memory usage in GB.

### [getGPUMem](utils\functions.py)

Function to get current GPU usage.

#### Returns:
- tuple: A tuple containing the total GPU memory and GPU memory usage for each available GPU.

### [circlePoses](utils\functions.py)

Function to generate circular camera poses.

#### Inputs:
- device (str): PyTorch device.
- radius (torch.Tensor, optional): Radius of the circle. Defaults to torch.tensor([3.2]).
- theta (torch.Tensor, optional): Theta angles in degrees. Defaults to torch.tensor([60]).
- phi (torch.Tensor, optional): Phi angles in degrees. Defaults to torch.tensor([0]).
- returnDirs (bool, optional): Whether to return view directions. Defaults to False.
- angleOverhead (int, optional): Angle overhead threshold in degrees. Defaults to 30.
- angleFront (int, optional): Angle front threshold in degrees. Defaults to 60.

#### Returns:
- tuple: A tuple containing camera poses and view directions (if returnDirs is True).

### [randPoses](utils\functions.py)

Function to genarate random camera poses.

#### Inputs:
- size (int): Number of camera poses to generate.
- device (str): PyTorch device.
- args (object): Additional arguments.
- radRange (list, optional): Range for the radius. Defaults to None.
- thetaRange (list, optional): Range for theta angles in degrees. Defaults to None.
- phiRange (list, optional): Range for phi angles in degrees. Defaults to None.
- returnDirs (bool, optional): Whether to return view directions. Defaults to False.
- angleOverhead (int, optional): Angle overhead threshold in degrees. Defaults to 30.
- angleFront (int, optional): Angle front threshold in degrees. Defaults to 60.
- uniSphRate (float, optional): Rate of uniform spherical sampling. Defaults to 0.5.

#### Returns:
- tuple: A tuple containing camera poses, view directions, theta angles, phi angles, and radii.