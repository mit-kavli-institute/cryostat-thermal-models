# cryostat-thermal-models
Python toolkit for simple thermal modeling of cryogenic systems. 

# Set Up and Install:
Instructions can be found in [setup.md](setup.md) to set up a new, easily replicated, python environment and install the repository. 

# Usage:
Example notebooks are found in [examples](examples). Start with the 3-stage sample model: [3-stage example model](examples/example_thermal_model_3stage.ipynb) which describes a simple 3-stage cryostat with some basic wiring and solves for the middle stage temperate assuming fixed inner and outer temperatures. 

The up-to-date model of the detector test cryostat is currently in [4-stage example model](examples/example_thermal_model_4stage.ipynb).

# Implemented features:
Radiative loading between stages: [radiation.py](src/cryotherm/radiation.py)

Conductive loading between stages: [conduction.py](src/cryotherm/conduction.py)

# Features in progress:
Smart modeling of windows/filters with transmission/emission in and out of band, and multi-filter stacks. 

# Using with Google Colab
This module is not yet on PyPI, but it can be used in cloud tools like google Colab like this:

`!pip install -q git+https://github.com/mit-kavli-institute/cryostat-thermal-models.git`

Then you can `import cryotherm` modules.





