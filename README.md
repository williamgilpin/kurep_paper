## Kuramoto model with repulsion

Simulate coupled oscillators using the Kuramoto model with short-range repulsion.

## Quickstart

Initialize and simulate a coupled oscillator model

```python
from oscillators import *
import numpy as np
from scipy.integrate import solve_ivp

## Define and construct model
n = 40 # number of oscillators
w_set = np.random.normal(1, 0.2, n) # individual frequencies
kmat = 0.4 * np.ones((n, n)) # coupling matrix
ts = CoupledOscillators(n, 
                        w_set, 
                        kmat, 
                        repulsion="gaussian",
                        repel_val=0.8, 
                        repel_length=1.02, 
                        num_repel=n
                        )
rhs = lambda t, y: ts.dyn_eq(y, t)
jac = lambda t, y: ts.jac(y, t)

## Integrate system
ic = 2 * np.pi * np.random.random(n) # initial phases
tvals = np.linspace(0, 500, 2000) # timepoints to evaluate
sol = solve_ivp(rhs, (0, 500), ic, t_eval=tvals, jac=jac, vectorized=True, method="LSODA", max_step=1e-4)
```

The resulting solution array `sol` will contain the phases evaluated at each timepoint in `tvals`. A longer version of this demo is included in [`examples.ipynb`](examples.ipynb)


### Dependencies

+ Python 3
+ matplotlib
+ numpy
+ numba (for faster simulations)
+ jupyter notebooks (for demo notebooks)

Note: the easiest way to install numba is as part of an Anaconda environment.

## Citation

If using this code in any published work, please cite the paper

> Gilpin, William. "Desynchronization of jammed oscillators by avalanches." Physical Review Research 3.2 (2021): 023206. [url](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023206)


### Development

This code will be kept mostly unchanged, but a more advanced version will soon be included in [oscy](https://github.com/williamgilpin/oscy), an in-progress sofware package upon which this repo is based.