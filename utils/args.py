class Args(object):
    """
    A configuration class for managing various parameters and hyperparameters used in the NeRF training pipeline.

    Args:
        posPrompt (str): A positive text prompt.
        negPrompt (str): A negative text prompt.
        expName (str): Experiment name.
        workspace (str): Workspace directory.
        fp16 (bool): Whether to use FP16 precision.
        seed (int): Random seed for reproducibility.
        sdVersion (str): Stable Diffusion version.
        hfModelKey: High-frequency model key.
        evalInterval (int): Number of training iterations between evaluations on the validation set.
        testInterval (int): Number of training iterations between testing on the test set.
        guidanceScale (int): Guidance scale for stable diffusion.
        saveMesh (bool): Whether to save the mesh.
        mcubesResolution (int): Resolution for extracting the mesh.
        decimateTarget (float): Target for mesh decimation.
        iters (int): Number of training iterations.
        lr (float): Maximum learning rate.
        maxSteps (int): Maximum number of steps sampled per ray.
        updateExtraInterval (int): Iteration interval to update extra status.
        latentIterRatio (float): Ratio of latent iterations.
        albedoIterRatio (float): Ratio of albedo iterations.
        minAmbientRatio (float): Minimum ambient ratio.
        texturelessRatio (float): Textureless ratio.
        jitterPose (bool): Adding jitter to randomly sampled camera poses.
        jitterCentre (float): Amount of jitter to add to sampled camera pose's center.
        jitterTarget (float): Amount of jitter to add to sampled camera pose's target.
        jitterUp (float): Amount of jitter to add to sampled camera pose's up-axis.
        uniformSphereRate (float): Probability of sampling camera location uniformly.
        gradClip (float): Clip grad for all gradients.
        gradClipRGB (float): Clip grad of RGB space grad.
        bgRadius (float): Radius of the background sphere.
        densityActivation (str): Density activation function ("exp" or "softplus").
        densityThresh (float): Threshold for density grid to be occupied.
        blobDensity (float): Max density for density blob.
        blobRadius (float): Controlling the radius for density blob.
        optim (str): Optimization function.
        w (int): Render width for training NeRF.
        h (int): Render height for training NeRF.
        knownViewScale (float): Multiply h/w by this for known view rendering.
        batchSize (int): Number of images to be rendered per batch.
        bound (int): Assume the scene is bounded in box(-bound, bound)x.
        dtGamma (float): dt_gamma (>=0) for adaptive ray marching. Set to 0 to disable, >0 to accelerate rendering (but usually with worse quality).
        minNear (float): Minimum near distance for the camera.
        radiusRange (list): Training camera radius range.
        thetaRange (list): Training camera along the polar axis (up-down).
        phiRange (list): Training camera along the azimuth axis (left-right).
        fovyRange (list): Training camera fovy range.
        defaultRadius (float): Radius for the default view.
        defaultPolar (float): Polar for the default view.
        defaultAzimuth (float): Azimuth for the default view.
        defaultFovy (float): Fovy for the default view.
        progressiveView (bool): Progressively expand view sampling range from default to full.
        progressiveViewInitRatio (float): Initial ratio of the final range.
        progressiveLevel (bool): Progressively increase grid encoder's max level.
        angleOverhead (float): Overhead angle.
        angleFront (float): Front angle.
        tRange (list): Range for t values.
        dontOverrideTRange (bool): Whether to override t range.
        lambdaEntropy (float): Loss scale for alpha entropy.
        lambdaOpacity (float): Loss scale for alpha value.
        lambdaOrient (float): Loss scale for orientation.
        lambdaGuidance (float): Loss scale for guidance.
        lambdaNormal (float): Loss scale for normal map.
        lambda2dNormalSmooth (float): Loss scale for 2D normal image smoothness.
        lambda3dNormalSmooth (float): Loss scale for 3D normal image smoothness.
        H (int): Mesh height for validation.
        W (int): Mesh width for validation.
        datasetSizeTrain (int): Length of the training dataset.
        datasetSizeValid (int): Number of frames to render in the turntable video during validation.
        datasetSizeTest (int): Number of frames to render in the turntable video during test time.
        expStartIter (int): Start iteration for experiment.
        expEndIter (int): End iteration for experiment.
        writeVideo (bool): Whether to write video during testing.
        emaDecay (float): Exponential moving average decay for training NeRF.
        schedulerUpdateEveryStep (bool): Update scheduler every training step.
        refRadii (list): Reference radii.
        refPolars (list): Reference polar angles.
        refAzimuths (list): Reference azimuth angles.
    """
    def __init__(
            self,
            posPrompt = "",
            negPrompt = "",
            expName = "df",
            workspace = "workspace",
            fp16 = False,
            seed = None,
            sdVersion = "2.1",
            hfModelKey = None,
            evalInterval = 1,
            testInterval = 100,
            guidanceScale = 100,
            saveMesh = True,
            mcubesResolution = 256,
            decimateTarget = 5e4,
            iters = 10000,
            lr = 1e-3,
            maxSteps = 1024,
            updateExtraInterval = 16,
            latentIterRatio = 0.2,
            albedoIterRatio = 0,
            minAmbientRatio = 0.1,
            texturelessRatio = 0.2,
            jitterPose = True,
            jitterCentre = 0.2,
            jitterTarget = 0.2,
            jitterUp = 0.02,
            uniformSphereRate = 0,
            gradClip = -1,
            gradClipRGB = 1,
            bgRadius = 1.4,
            densityActivation = "exp",
            densityThresh = 10,
            blobDensity = 5,
            blobRadius = 0.2,
            optim = "adan",
            w = 64,
            h = 64,
            knownViewScale = 1.5,
            batchSize = 1,
            bound = 1,
            dtGamma = 0,
            minNear = 0.01,
            radiusRange=None,
            thetaRange=None,
            phiRange=None,
            fovyRange=None,
            defaultRadius = 3.2,
            defaultPolar = 90,
            defaultAzimuth = 0,
            defaultFovy = 20,
            progressiveView = True,
            progressiveViewInitRatio = 0.2,
            progressiveLevel = True,
            angleOverhead = 30,
            angleFront = 60,
            tRange=None,
            dontOverrideTRange = True,
            lambdaEntropy = 1e-3,
            lambdaOpacity = 0,
            lambdaOrient = 1e-2,
            lambdaGuidance = 1,
            lambdaNormal = 0,
            lambda2dNormalSmooth = 0,
            lambda3dNormalSmooth = 0,
            H = 800,
            W = 800,
            datasetSizeTrain = 100,
            datasetSizeValid = 8,
            datasetSizeTest = 100,
            expStartIter = None,
            expEndIter = None,
            writeVideo = True,
            emaDecay = 0.95,
            schedulerUpdateEveryStep = True
    ):
        if radiusRange is None:
            radiusRange = [3.0, 3.5]
        if thetaRange is None:
            thetaRange = [45, 105]
        if phiRange is None:
            phiRange = [-180, 180]
        if fovyRange is None:
            fovyRange = [10, 30]
        if tRange is None:
            tRange = [0.02, 0.98]
        self.posPrompt = posPrompt
        self.negPrompt = negPrompt
        self.expName = expName
        self.workspace = workspace
        self.fp16 = fp16
        self.seed = seed
        self.sdVersion = sdVersion
        self.hfModelKey = hfModelKey
        self.evalInterval = evalInterval
        self.testInterval = testInterval
        self.guidanceScale = guidanceScale
        self.saveMesh = saveMesh
        self.mcubesResolution = mcubesResolution
        self.decimateTarget = decimateTarget
        self.iters = iters
        self.lr = lr
        self.maxSteps = maxSteps
        self.updateExtraInterval = updateExtraInterval
        self.latentIterRatio = latentIterRatio
        self.albedoIterRatio = albedoIterRatio
        self.minAmbientRatio = minAmbientRatio
        self.texturelessRatio = texturelessRatio
        self.jitterPose = jitterPose
        self.jitterCentre = jitterCentre
        self.jitterTarget = jitterTarget
        self.jitterUp = jitterUp
        self.uniformSphereRate = uniformSphereRate
        self.gradClip = gradClip
        self.gradClipRGB = gradClipRGB
        self.bgRadius = bgRadius
        self.densityActivation = densityActivation
        self.densityThresh = densityThresh
        self.blobDensity = blobDensity
        self.blobRadius = blobRadius
        self.optim = optim
        self.w = w
        self.h = h
        self.knownViewScale = knownViewScale
        self.batchSize = batchSize
        self.bound = bound
        self.dtGamma = dtGamma
        self.minNear = minNear
        self.radiusRange = radiusRange
        self.thetaRange = thetaRange
        self.phiRange = phiRange
        self.fovyRange = fovyRange
        self.defaultRadius = defaultRadius
        self.defaultPolar = defaultPolar
        self.defaultAzimuth = defaultAzimuth
        self.defaultFovy = defaultFovy
        self.progressiveView = progressiveView
        self.progressiveViewInitRatio = progressiveViewInitRatio
        self.progressiveLevel = progressiveLevel
        self.angleOverhead = angleOverhead
        self.angleFront = angleFront
        self.tRange = tRange
        self.dontOverrideTRange = dontOverrideTRange
        self.lambdaEntropy = lambdaEntropy
        self.lambdaOpacity = lambdaOpacity
        self.lambdaOrient = lambdaOrient
        self.lambdaGuidance = lambdaGuidance
        self.lambdaNormal = lambdaNormal
        self.lambda2dNormalSmooth = lambda2dNormalSmooth
        self.lambda3dNormalSmooth = lambda3dNormalSmooth
        self.H = H
        self.W = W
        self.datasetSizeTrain = datasetSizeTrain
        self.datasetSizeValid = datasetSizeValid
        self.datasetSizeTest = datasetSizeTest
        self.expStartIter = expStartIter
        self.expEndIter = expEndIter
        self.writeVideo = writeVideo
        self.emaDecay = emaDecay
        self.schedulerUpdateEveryStep = schedulerUpdateEveryStep
        self.refRadii = []
        self.refPolars = []
        self.refAzimuths = []