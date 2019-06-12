"""
An efficient library for simulating the Kuramoto model and variations using Python

NOTE: Code frozen 5.31.2019 in order to remain static for peer review. 
Updated oscillator code for future projects available in another repository


Depending on availability and settings, can use either vectorized Numpy, 
JIT compiled Numpy (via Numba), or compiled stepper functions (via Cython)

Requirements
+ Numpy
+ Matplotlib
+ Scipy
+ Numba (optional)
+ Cython (optional)

Development:
Add in analytical forms of Jacobians, this should speed up integration depending on solver
See [here](https://github.com/Jonas77/Kuramoto2)

"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

try:
    from numba import jit, njit
    has_numba = True
except ModuleNotFoundError:
    warnings.warn("Numba not installed, some functions will run slower")
    has_numba = False
    # Define placeholder functions
    def jit(func):
        return func
    def njit(func):
        return func

try:
    from oscillator_funcs import DiffEq
    has_c = True
except ModuleNotFoundError:
    warnings.warn("Compiled Cython library not found, some functions will run slower")





    

def find_daido(phase_vals, n=1, coord='polar'):
    """
    Find the nth order Daido order parameters

    Args:
        phase_vals : T x M array, where M is the number of
                    oscillators.
        n : the order of the daido parameter
        coord : whether to return a polar (standard) or cartesian 
                parameter
    Returns:
        (r_rot, th_rot): Nx1 arrays for the two Daido order parameters
    """
    re, im = (np.mean(np.cos(n*phase_vals), axis=1), 
              np.mean(np.sin(n*phase_vals), axis=1))
    
    r, th = np.sqrt(im**2 + re**2), np.arctan2(im, re)
    
    if coord == 'polar':
        return r, th
    if coord == 'cartesian':
        return re, im
    else:
        return r, th


@jit
def heun_solver(yprime, y0, tpts):
    """
    A Python implementation of Heun's method (RK2) in 1D
    """
    dt = tpts[1] - tpts[0]
    npts = len(tpts)
    tpts= np.append(tpts, tpts[-1] + dt)
    sol = np.zeros((npts, len(y0)))
    sol[0, :] = y0
    ycurr = y0
    for ii in range(1, npts):
        ypp = yprime(ycurr, tpts[ii])
        yt = ycurr + dt*ypp
        ycurr = ycurr + (dt/2)*(ypp + yprime(yt, tpts[ii + 1]))
        sol[ii, :] = ycurr
    return sol

@jit
def rk4_solver(yprime, y0, tpts):
    """
    A Python implementation of the fourth-order Runge-Kutta method in 1D
    """
    dt = tpts[1] - tpts[0]
    npts = len(tpts)
    tpts= np.append(tpts, tpts[-1] + dt)
    sol = np.zeros((npts, len(y0)))
    sol[0, :] = y0
    ycurr = y0
    for ii in range(1, npts):
        k1 = yprime(ycurr, tpts[ii])
        k2 = yprime(ycurr + k1/2, tpts[ii] + dt/2)
        k3 = yprime(ycurr + k2/2, tpts[ii] + dt/2)
        k4 = yprime(ycurr + k3, tpts[ii] + dt)
        ycurr = ycurr + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        sol[ii, :] = ycurr
    return sol


@njit
def soft_square(x, rad, hardness=50):
    """A softened Heaviside step"""
    return np.exp(-(x/rad)**hardness)

@njit
def dsoft_square(x, rad, hardness=50):
    """A softened derivative of a Heaviside step"""
    return -(hardness/rad)*np.exp(-(x/rad)**hardness)*(x/rad)**(hardness-1)

@njit
def dsoft_triangle(x, rad, hardness=50):
    """
    Approximate the derivative of a unit triangular potential 
    using two softened square pulses

    Args:
        x : np.array, the x values at which to calculate
        rad : float, the radius of the potential
        hardness : int, the hardness of the potential

    Returns:
        out : the y values of the triangle
    """
    # if hardness % 2 == 1:
    #     hardness = hardness+1
        
    p1 = (1/rad)*np.exp(-((x - rad/2)/(rad/2))**hardness)
    p2 = (1/rad)*np.exp(-((x + rad/2)/(rad/2))**hardness)
    out = p1 - p2
    return out
    
class CoupledOscillators(object):
    """
    Define and simulate a system of coupled oscillators
    """

    def __init__(self, n_osc, w_vals, k_matrix, 
                 a_vec=[0.0], coupling='kuramoto', use_c=False, 
                 repel_val=1.0, repel_length=1, num_repel=1, hardness=50):
        """
        Args:
            n_osc : int for the total number of oscillators
            w_vals : np.array of oscillator natural frequencies
            k_matrix : (n_osc x n_osc) np.array. A matrix specifying 
                    the couplings among oscillators. This does not need to be normalized
                    by the number of oscillators
            a_vec : n_osc array of fixed phase offsets for the array
            coupling : str for the type of dynamical equation
            use_c : bool. if True, uses Cython or an C++ library to represent the ODE
            repel_val : if using Kuramoto with repulsion, the amplitude of the repulsive term
            repel_length : the distance over which repulsion acts (in units of 2 pi/n_osc)
            hardness : the stiffness of the repulsive term
        """
        
        if not (k_matrix.shape[0] == k_matrix.shape[1] == n_osc):
            warnings.warn("Coupling matrix unequal to number of oscillators")
        
        self.n = n_osc
        self.k = k_matrix
        self.lonely_dist = 2*np.pi/n_osc

        self.num_repel = num_repel
        self.repel_val = repel_val
        
        if len(a_vec) == 1:
            a_vec = a_vec*np.ones(n_osc)

        if use_c and has_c:
            self.c_flag = True
        else:
            self.c_flag = False
            if use_c:
                warnings.warn("")

        
        if coupling == 'kuramoto':
            
            if self.c_flag:
                ndot = DiffEq(w_vals, k_matrix, a_vec, model="kuramoto")

            else:
                @njit
                def ndot(phase_vals, tvals):
                    """
                    Automatically generated function for the dynamical equation
                    """
                    # phase_diff = phase_vals[:,np.newaxis].T - phase_vals[:,np.newaxis] - (a_vec[:,np.newaxis])
                    phase_diff = phase_vals.reshape((1, -1)) - phase_vals.reshape((-1, 1)) - a_vec.reshape((-1, 1))
                    sin_diff = np.sin(phase_diff)
                    
                    return w_vals + (1/n_osc)*np.sum(k_matrix*sin_diff, axis=1)

                @njit
                def jac(phase_vals, tvals):
                    """
                    Automatically generated jacobian function for the dynamical equation
                    """
                    # phase_diff = phase_vals[:,np.newaxis].T - phase_vals[:,np.newaxis] - (a_vec[:,np.newaxis])
                    phase_diff = phase_vals.reshape((1, -1)) - phase_vals.reshape((-1, 1)) - a_vec.reshape((-1, 1))

                    cos_diff = k_matrix*np.cos(phase_diff)
                    row_sum = np.sum(cos_diff, axis=1)-1

                    out = cos_diff
                    out[np.diag_indices(n_osc)] = -row_sum

                    return out

        elif coupling == 'kuramoto_single_repulsion':

            if self.c_flag:
                ndot = DiffEq(w_vals, k_matrix, a_vec, model="kuramoto_single_repulsion",
                        repel_length=repel_length, num_repel=num_repel, 
                        repel_val=repel_val, hardness=hardness)

            else:
                where_repel = range(num_repel)
                mask_arr = np.zeros((n_osc, n_osc))

                for ind in where_repel:
                    mask_arr[ind, :] = 1
                    mask_arr[:, ind] = 1
                    mask_arr[ind, ind] = 0

                @njit
                def ndot(phase_vals, tvals):
                    """
                    Automatically generated function for the dynamical equation
                    """

                    phase_diff = phase_vals.reshape((1, -1)) - phase_vals.reshape((-1, 1)) - a_vec.reshape((-1, 1))
                    sin_diff = np.sin(phase_diff)
                    
                    phase_diff = np.fmod(phase_diff, 2*np.pi)
                    repel_term = repel_val*dsoft_triangle(phase_diff, repel_length*2*np.pi/n_osc, hardness)
                    repel_term = repel_term*mask_arr

                    return w_vals + (1/n_osc)*np.sum(k_matrix*sin_diff, axis=1) - (1/num_repel)*np.sum(repel_term, axis=1)  

                ## this is missing a term:
                @njit
                def jac(phase_vals, tvals):
                    """
                    Automatically generated jacobian function for the dynamical equation
                    """
                    phase_diff = phase_vals.reshape((1, -1)) - phase_vals.reshape((-1, 1)) - a_vec.reshape((-1, 1))

                    cos_diff = k_matrix*np.cos(phase_diff)
                    row_sum = np.sum(cos_diff, axis=1)-1

                    out = cos_diff
                    out[np.diag_indices(n_osc)] = -row_sum
                    return out

        else:
            @njit
            def ndot(phase_vals, tvals):
                """
                Automatically generated function for the dynamical equation
                """
                return w_vals

            @njit
            def jac(phase_vals, tvals):
                """
                Jacobian of the above function
                """
                return np.zeros((n_osc,n_osc))
            warnings.warn("Dynamical equation type not recognized, no coupling used.")
        
        self.dyn_eq = ndot
        self.jac = 0#jac

    def run_simulation(self, y0, tvals, method='odeint'):
        """
        Simulate the system using odeint's built in solver
        """
        if method == "odeint":
            sol = spi.odeint(self.dyn_eq, y0, tvals, args=())

        elif method == "ode":
            
            f = lambda x, y: self.dyn_eq(y,x)
            ode = spi.ode(f)
            dt = tvals[1] - tvals[0]

            # BDF method suited to stiff systems of ODEs
            ode.set_integrator('vode', nsteps=len(tvals), method='bdf')
            ode.set_initial_value(y0, tvals[0])

            ts = []
            ys = []
            while ode.successful() and ode.t < tvals[-1]:
                ode.integrate(ode.t + dt)
                ts.append(ode.t)
                ys.append(ode.y)

            t = np.vstack(ts)
            sol = np.vstack(ys)
            
        elif method == "heun":
            sol = heun_solver(self.dyn_eq, y0, tvals)

        elif method == "rk4":
            sol = rk4_solver(self.dyn_eq, y0, tvals)

        else:
            warnings.warn("Solution method not recognized, falling back to odeint")
            sol = spi.odeint(self.dyn_eq, y0, tvals, args=())

        return sol


    

    

