# mwhalo-shape-2016
Investigation into the shape of the inner DM halo with stream and disk-star kinematics.

This repository contains the code associated with the paper Bovy,
Bahmanyar, Fritz, \& Kallivayalil (2016), which you should cite if you
re-use any of this code (in addition to, most likely,
[galpy](https://github.com/jobovy/galpy)). The ipython notebooks used
to generate the plots in this paper can be found in the ``py/``
directory of this repository. This code uses
[galpy](https://github.com/jobovy/galpy).

There are three useful notebooks, in addition to others that were used
during code development that are not described further here.

## 1. [MWPotential2014-varyc.ipynb](py/MWPotential2014-varyc.ipynb)

(render this notebook on [nbviewer](http://nbviewer.ipython.org/github/jobovy/mwhalo-shape-2016/blob/master/py/MWPotential2014-varyc.ipynb), where you can toggle the code)

This notebook contains the fits of sections 2 and 6 of three-component
Milky-Way potential models to a variety of dynamical data and the
newly derived Pal 5 and GD-1 measurements. A variety of fits are
explored, most of which are described in the paper. The full set of
two-dimensional PDFs is also incldued for each fit. Figures 1 and 9 in
the paper are produced by this notebook. Figure 10 of the best-fit
force field and the constraints from disk stars, Pal 5, and GD-1 data
is also made by this notebook.

## 2. [pal5Modeling.ipynb](py/pal5Modeling.ipynb)

(render this notebook on [nbviewer](http://nbviewer.ipython.org/github/jobovy/mwhalo-shape-2016/blob/master/py/pal5Modeling.ipynb), where you can toggle the code)

This notebook explores the constraints on the Milky Way's
gravitational potential from the Pal 5 stream data. Most of the code
to predict the stream track for Pal 5 is actually contained in
[pal5_util.py](py/pal5_util.py), which is used in this notebook to
compute stream tracks (Figure 2). The MCMC exploration of the 32
potential families is performed by the [mcmc_pal5.py](py/mcmc_pal5.py)
code. The results of the MCMC (Figures 3, 4, and 5 in the paper) are
analyzed in this notebook.

The MCMC analyses were run with commands like
```
python mcmc_pal5.py -i 0 -o ../pal5_mcmc/mwpot14-fitsigma-0.dat --dt=600. --td=10. --fitsigma -m 6
```

## 3. [gd1Modeling.ipynb](py/gd1Modeling.ipynb)

(render this notebook on [nbviewer](http://nbviewer.ipython.org/github/jobovy/mwhalo-shape-2016/blob/master/py/gd1Modeling.ipynb), where you can toggle the code)

This notebook explores the constraints on the Milky Way's
gravitational potential from the GD-1 stream data. Most of the code to
predict the stream track for GD-1 is actually contained in
[gd1_util.py](py/gd1_util.py), which is used in this notebook to
compute stream tracks (Figure 6). This code is very similar to the Pal
5 code above. The MCMC exploration of the 32 potential families is
performed by the [mcmc_gd1.py](py/mcmc_gd1.py) code. This code is
again very similar to the Pal 5 MCMC code above, but is slightly
different because of the different parameterization of the progenitor
and stream properties. The results of the MCMC (Figures 7 and 8 in the
paper) are analyzed in this notebook.

The MCMC analyses were run with commands like
```
python mcmc_gd1.py -i 0 -o ../gd1_mcmc/mwpot14-0.dat --dt=1440. --td=10. -m 7
```
