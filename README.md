## RBniCS - reduced order modelling in FEniCS ##
![RBniCS - reduced order modelling in FEniCS](https://www.rbnicsproject.org/_images/rbnics-logo-small.png "RBniCS - reduced order modelling in FEniCS")

The **RBniCS** Project contains an implementation in **FEniCS** of several reduced order modelling techniques for parametrized problems.

**RBniCS** is currently developed and maintained at the [Catholic University of the Sacred Heart](https://www.unicatt.it/) by [Dr. Francesco Ballarin](https://www.francescoballarin.it) in collaboration with [Prof. Gianluigi Rozza](https://people.sissa.it/~grozza/)'s group at [SISSA mathLab](http://mathlab.sissa.it/). The financial support of the [AROMA-CFD ERC CoG project](https://people.sissa.it/~grozza/aroma-cfd/) is gratefully acknowledged.

Like all core **FEniCS** components, **RBniCS** is freely available under the GNU LGPL, version 3.

Visit [www.rbnicsproject.org](https://www.rbnicsproject.org) for more information.


# To install in a conda environment
On Linux (Ubuntu 18.06) the following set-up was giving a version of fenics and mshr that works with old scripts
```
conda create -n fenics-2018 -c conda-forge python=3.7 ipython jupyter fenics=2018 mshr=2018 matplotlib scipy ipykernel
```

Then one has to activate the environment
```
conda activate fenics-2018
```

Finally, one needs to install RBniCS. Go in the main folder of the repository and type
```
python setup.py install
```

If interested in jupyter notebooks, remember to add the kernel to the jupyter kernerls
```
python -m ipykernel install --user --name=fenics-2018
```


