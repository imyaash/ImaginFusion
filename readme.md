# ImaginFusion: A text guided 3D model generation application.

Welcome to ImaginFusion! This application is part of my Master's Dissertation project in Data Science. It allows you to generate 3D models based on natural language text, and it comes with both a Command Line Interface (CLI) and a Graphical User Interface (GUI) for your convenience.

This implementation is based on the [Dreamfusion](https://dreamfusion3d.github.io/) approach, but uses [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for text guided image generation instead of [Imagen](https://imagen.research.google/) and [torch-ngp](https://github.com/ashawkey/torch-ngp) instead of [Mip-NeRF](https://jonbarron.info/mipnerf/) for 3D synthesis.

## System Requirements

1. This project has been tested on Windows 10 with python 3.9. It is recommended to use a similar environment for optimal performance and compatibility.
2. CPU Requirement: A modern CPU with atleast 6-cores is recommended.
3. RAM Requirement: 16GB of RAM is highly recommended for this application to function properly.
4. GPU Requirement: A Nvidia GPU with atleast 8GB of VRAM is mandatory for this application to function properly. Please ensure that your system meets this requirement.

## Software requirements
Install these beforehand
1. [**CUDA toolkit 11.8**](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. [**VS2022**](https://visualstudio.microsoft.com/)  Select Desktop development with C++ and Install the MSVCs + Windows SDK.


## Installation

Follow the below steps to get the application up and running.

1. Download the Project and save it to the desired location.
2. Install Dependencies: Open your command prompt (CMD) or PowerShell, navigate to the project folder, and run the following command to install the required dependencies:
    ```
    setup.bat
    ```
3. GPU Requirements: Ensure that your system has a Nvidia GPU with atleast 8GB of VRAM. This is a crucial requirement for the application to perform efficiently.

## Usage

### Command Line Interface (CLI)

To use the CLI follow the these steps:
1. Open your CMD or PowerShell.
2. Navigate to the project folder.
3. Run the following command to see all the available CLI options:
    ```
    python cli.py -h
    ```
4. You can also find the sample CLI commands in the "testScript.bat" file.

### Graphical User Interface (GUI)
To use the GUI, follow these steps:
1. Simply double click on the "gui.py" file.
2. Then, once the server is running, open you r web browser and navigate to "localhoast:7860". This will launch the GUI allowing you to interact with the application using an intuitive web interface.

## Note:
1. The requirements may take upto an hour to install, depending on your system performance and internet speed.
2. When you run the application for the first time, it might take longer than normal as it will download the StableDiffusion checkpoints (about 4GB) and save them to disk for repeated use.
3. The application might take anywhere from 25 to 150 minutes to generate a 3D model based on your system, object being generated, and the confuguration/options/arguments being used.

## Contact
If you encounter any issues or have any questions about this application, please don't hesitate to get in touch with me on my [email](yashppanchal1997@gmail.com). Your feedback and insights are valuable for my dissertation research.

## Acknowledgements
- Credit for the [Dreamfusion](https://dreamfusion3d.github.io/) approach goes to [Ben Poole](https://cs.stanford.edu/~poole/).
    ```
    @article{poole2022dreamfusion,
    author = {Poole, Ben and Jain, Ajay and Barron, Jonathan T. and Mildenhall, Ben},
    title = {DreamFusion: Text-to-3D using 2D Diffusion},
    journal = {arXiv},
    year = {2022},
    }
    ```
- Credits for the [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [instant-ngp](https://github.com/NVlabs/instant-ngp) goes to [Thomas Müller](https://tom94.net/).
    ```
    @misc{tiny-cuda-nn,
        Author = {Thomas M\"uller},
        Year = {2021},
        Note = {https://github.com/nvlabs/tiny-cuda-nn},
        Title = {Tiny {CUDA} Neural Network Framework}
    }

    @article{mueller2022instant,
        title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
        author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
        journal = {arXiv:2201.05989},
        year = {2022},
        month = jan
    }
    ```
- Credits for the brilliant [torch-ngp: A PyTorch implementation of instant-ngp](https://github.com/ashawkey/torch-ngp) goes to [Jiaxiang Tang](https://me.kiui.moe/) their implementation has been a huge help in throughout this project.
    ```
    @misc{torch-ngp,
        Author = {Jiaxiang Tang},
        Year = {2022},
        Note = {https://github.com/ashawkey/torch-ngp},
        Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
    }

    @article{tang2022compressible,
        title = {Compressible-composable NeRF via Rank-residual Decomposition},
        author = {Tang, Jiaxiang and Chen, Xiaokang and Wang, Jingbo and Zeng, Gang},
        journal = {arXiv preprint arXiv:2205.14870},
        year = {2022}
    }
    ```
- Credits for [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4), [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [Stable Diffusion 2.0](https://huggingface.co/stabilityai/stable-diffusion-2) & [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) and their checkpoints goes to [CompVis](https://ommer-lab.com/), [Runway](https://runwayml.com/) & [Stability AI](https://stability.ai/).
    ```
    @InProceedings{Rombach_2022_CVPR,
        author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
        title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2022},
        pages     = {10684-10695}
    }
    ```
- Credit for [Gradio](https://github.com/gradio-app/gradio) framework goes to [Abubakar Abid](https://abidlabs.github.io/).
    ```
    @article{abid2019gradio,
    title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
    author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
    journal = {arXiv preprint arXiv:1906.02569},
    year = {2019},
    }
    ```

# Thank you for using ImaginFusion!