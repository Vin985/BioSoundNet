# BioSoundNet

BioSoundNet is an acoustic bird vocalization detector based on deep learning. It is based on the [mouffet](https://github.com/Vin985/mouffet) framework that allow flexible training and evaluation of models with the use of scenarios.

BioSoundNet comes with a pre-trained model and examples to easily use it. Also included is the code used to create the figures presented inside the paper (TBD).


# Installation

## Installing git

Git is a version control manager. It allows you to save your code and keep a history of all changes made to the code.
A popular interface using git is the Github website on which this code is stored.
To install easily our packages we will need to install git first. You can find it there:

https://git-scm.com/downloads

## Installing python

If you do not already have one, first install a python version on your computer. A python 3 version is required.
We recommend downloading the latest miniconda version that can be found here:
https://docs.conda.io/en/latest/miniconda.html

The site provides information on how to install it on all platforms. Note that you will be required to
use a terminal for program to work. On windows, miniconda installs a terminal directly. You can find it in the
application start menu.

Once miniconda is install, open a terminal. To make sure the installation has worked correctly, you can try to enter
the following command in the terminal:

    python

This should launch a python terminal. To exit the python terminal, type:

    exit()

## Setting up an environment
The following steps are optional but are good practice in Python. If you do no want to proceed, go directly to the next section 

To isolate our work, we will create a python environment. Python environments allow to create an isolated place
where we can avoid package conflicts. To do so, type the following commands:

    cd path/where_I_want_my_environment   # Moves into the working directory
    python -m venv biosoundnet_env

This should create the environment in the subfolder biosoundnet_env. Now that it is created, we need to activate it
to let python know where to install our packages

    biosoundnet_env/Scripts/activate            # This should be the path on Windows
    source biosoundnet_env/bin/activate         # This should work on Linux and Mac

## Installing the dependencies

Now we need to install the dependencies. To do so we will install them using the pip package manager that comes with python.
If this is the first time you use python, you will probably need to update pip. For that, type in you terminal:

    pip install pip --upgrade

Once this is done, you can install the mouffet package using this command:

    pip install -U git+https://github.com/vin985/biosoundnet.git


# Using BioSoundNet

Examples on how to use BioSoundNet can be found in the [examples](examples) folder. 

## Generate predictions on a single audio file
An example on how to use BioSoundNet to generate predictions on a single audio file can be found here: [predict_single.py](examples/predict_single.py).

## Generate predictions on multiple audio files
An example on how to use BioSoundNet to generate predictions on multiple audio files can be found here: [predict_multiple.py](examples/predict_multiple.py)

