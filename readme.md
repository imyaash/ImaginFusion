# ImaginFusion: A text guided 3D model generation application.

Welcome to ImaginFusion! This application is part of my Master's Dissertation project in Data Science. It allows you to generate 3D models based on natural language text, and it comes with both a Command Line Interface (CLI) and a Graphical User Interface (GUI) for your convenience.

## Installation

Follow the below steps to get the application up and running.

1. Download the Project and save it to the desired location.
2. Install Dependencies: Open your command prompt (CMD) or PowerShell, navigate to the project folder, and run the following command to install the required dependencies:
```
pip install -r requirements.txt
```
Please note that this may take upto 30 minutes to complete, depending on your system performance and internet speed.
Note: When you run the application for the first time, it might take longer than normal as it will download the StableDiffusion checkpoints (about 4GB) and save them to disk for repeated use.
3. GPU Requirements: Ensure that your system has a Nvidia GPU with atleast 8GB of VRAM. This is a crucial requirement for the application to perform efficiently.

## Usage

Note: The application might take anywhere from 25 to 150 minutes to generate a 3D model based on your system, object being generated, and the confuguration/options/arguments being used.

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

## System Requirements

1. This project has been tested on Windows 10 with python 3.9. It is recommended to use a similar environment for optimal performance and compatibility.
2. GPU Requirement: A Nvidia GPU with atleast 8GB of VRAM is mandatory for this application to function properly. Please ensure that your system meets this requirement.

## Contact
If you encounter any issues or have any questions about this application, please don't hesitate to get in touch with me. Your feedback and insights are valuable for my dissertation research.

Thank you for using ImaginFusion!