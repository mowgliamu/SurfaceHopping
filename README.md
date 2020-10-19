# Surface Hopping

Welcome to Surface Hopping, a program to perform Tully's Fewest-Switches Surface Hopping (FSSH) to simulate nonadiabatic dynamics for one-dimensional model systems. 

Currently, standard decoherence-corrected FSSH is implemented with momentum adjustments, but more features will be added in future.

## What you need to run the program

- python3
- numpy
- scipy
- autograd (automatic differentiation. Install using ```pip install autograd ```)

If your system is running default python2, you can install python3 alongside it and make use of virtual enrionment. This can be easily done by Anaconda distribution, for example. 

## Defining the model potential

The user has to define the model potential in diabatic picture in input_pes.py in the main module. Each element of the diabatic potential energy matrix, both diagonal and off-diagonal, need to be defined individually. At the end, one needs to collect all the diabats (diagonal elements) and all the couplings (off-diagonal elements) in two separate lists. This step is important as this defines the vibronic model for the program. For a two-state example, check the ```input_pes.py``` in the main module.


## Running the program

Once the potneital has been defined, the Surface Hopping program is invoked from the top level directory by simply executing:

```
python surface_hopping.py
```

The input for SH dynamics must be defined in surface_hopping.py. The keywords are self-explanatory. This run will create three output files:

```
output_sh_dyn
md_data
populations
```

The main output file (output_sh_dyn) keeps track of hopping and one can see this by doing ``` grep Hopping output_sh_dyn``` (case sensitive). The md_data file contains all the relevant information on dynamics as a function of time. The populations file writes all the electronic state populations.


## Running the tests

The unit tests can be run by invoking the following command in the top-level directory:

```
python -m unittest discover -v
```

## Documentation

https://mowgliamu.github.io/SurfaceHopping
