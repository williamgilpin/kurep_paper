"""
An efficient library for simulating the Kuramoto model and variations using Python

Depending on availability and settings, can use either vectorized Numpy, 
JIT compiled Numpy (via Numba), or compiled stepper functions (via Cython)

Requirements
+ Numpy
+ Matplotlib
+ Scipy
+ Numba (optional)
+ Cython (optional)

Development:
Add in analytical forms of Jacobians, to see whether this speeds up the numerical integration

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


@njit
def ang_dist(ang1, ang2, signed=False):
    """
    Find the absolute distance between two angles, 
    which can be at most pi.
    
    DEV: is there a reasonable heuristic for determining
    the sign? Possible just the sign of the raw difference
    times the output
    
    Args:
        ang1, ang2: np.arrays 
        signed: bool. **Not yet implemented**. If True, returns a 
            signed distance in the form ang2 - ang1
    Returns: 
        out: array of distances between the two arrays, with
            dimension determined by numpy broadcasting rules
    """
    t = ang1 - ang2
    out = np.abs(np.arctan2(np.sin(t), np.cos(t)))
    return  out

@njit
def mod_c(arr, n):
    """
    The signed mod along a circle of circumference n
    """
    return np.mod(arr-n/2, n) - n/2

def is_overlap(ang1, ang2, rad):
    """
    Do two angles fall within rad of each other on a circle?

    ang1, ang2 : both 1xN or Nx1 vectors
    rad : 1xN or Nx1 or scalar list of distances
    """
    # flag = False
    # if ang_dist(ang1, ang2)<rad:
        # flag = True
    # else:
        # flag = False
    # return flag
    out = ang_dist(ang1, ang2)<rad
    return out


@njit
def soft_square(x, rad, hardness=50):
    """A softened Heaviside step"""
    return np.exp(-(x/rad)**hardness)

@njit
def dsoft_square(x, rad, hardness=50):
    """A softened derivative of a Heaviside step"""
    return -(hardness/rad)*np.exp(-(x/rad)**hardness)*(x/rad)**(hardness-1)

@njit
def dtriangle_repel(x, rad, hardness=50):
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

@njit
def dgaussian_repel(x, std0):
    """
    Calculate derivative of a truncated gaussian 
    repulsive potential
    """
    # derivative of PDF
    #val = x*np.exp(-(x**2)/((2*std)**2))
    #pre = 1/((std**3)*np.sqrt(2*np.pi)) #proper
    
    # rescale to be independent of width
    std = std0/4
    val = x*np.exp(-(x**2)/((2*std)**2))
    pre = 1/((std**2)*np.sqrt(2*np.pi)) #rescaled
    out = pre*val
    
    # cut off large radii (not necessary)
    # out = out*soft_square(x, 5*std, hardness=50)
    
    return out

@njit
def ddgaussian_repel(x, std0):
    """
    Calculate derivative of a truncated gaussian 
    repulsive potential
    """
    std = std0
    out = -(8*np.sqrt(2/np.pi)*(8*x**2 - std**2))/(np.exp((4*x**2)/std**2)*std**4)
    return out


@njit
def dcauchy_repel(x, std):
    """
    Calculate derivative of a Cauchy repulsive potential. 
    The width is rescaled by a numerical prefactor, in order to match
    the width of an equivalent Gaussian distribution
    """
    # derivative of PDF
    #return (2*x)/(std**3*np.pi*(1 + (x/std)**2)**2) 
    
    # empirically rescale to match size/width of Gaussian
    std = std/2
    pre=3.3105
    out = pre*(2*x)/(std**2*np.pi*(1 + (x/std)**2)**2) # rescaled
    
    # cut off large radii (not necessary)
    # out = out*soft_square(x, 5*std, hardness=50)
    
    return out 

@njit
def ddcauchy_repel(x, std):
    """
    Calculate derivative of a Cauchy repulsive potential. 
    The width is rescaled by a numerical prefactor, in order to match
    the width of an equivalent Gaussian distribution
    """
    std = std/2
    pre= 3.3105/np.pi
    out = pre*(2*std**2)*(std**2 - 3*x**2)/(x**2 + std**2)**3
    return out 

    
class CoupledOscillators(object):
    """
    Simulate a system of coupled oscillators
    
    Parameters
    ----------
    n_osc : int
        The total number of oscillators
    w_vals : np.array 
        Oscillator natural frequencies
    k_matrix : (n_osc x n_osc) np.array. 
        A matrix specifying the couplings among oscillators. This does not need to be 
        normalized by the number of oscillators
    a_vec : ndarray
        n_osc array of fixed phase offsets for the array
    coupling : str 
        The type of dynamical equation
    repulsion : {None, "gaussian", "cauchy"}
        The type of short-range repulsion to use
    repel_val : float
        If using Kuramoto with repulsion, the amplitude of the repulsive term
    repel_length : float
        The distance over which repulsion acts (in units of 2 pi/n_osc)
    hardness : float
        the stiffness of the repulsive term
    """

    def __init__(self, 
                 n_osc, 
                 w_vals, 
                 k_matrix, 
                 a_vec=[0.0], 
                 repulsion="gaussian", 
                 repel_val=1.0, 
                 repel_length=1, 
                 num_repel=1, 
                 hardness=50, 
                 **kwargs):
        
        if not (k_matrix.shape[0] == k_matrix.shape[1] == n_osc):
            warnings.warn("Coupling matrix unequal to number of oscillators")
        
        self.w = w_vals
        self.n = n_osc
        self.k = k_matrix
        self.lonely_dist = 2*np.pi/n_osc

        self.num_repel = num_repel
        self.repel_val = repel_val
        
        if len(a_vec) == 1:
            a_vec = a_vec*np.ones(n_osc)
    
        if not repulsion:
            @njit
            def ndot(phase_vals, tvals):
                """
                Kuramoto model
                """
                # phase_vals = np.mod(phase_vals, 2*np.pi)
                phase_diff = phase_vals.reshape((1, -1)) - phase_vals.reshape((-1, 1)) - a_vec.reshape((-1, 1))
                sin_diff = np.sin(phase_diff)
                return w_vals + (1/n_osc)*np.sum(k_matrix*sin_diff, axis=1)
            
            @njit
            def jac(phase_vals, tvals):
                """
                Jacobian function for the dynamical equation
                """
                phase_diff = phase_vals.reshape((1, -1)) - phase_vals.reshape((-1, 1)) - a_vec.reshape((-1, 1))

                cos_diff = (1/n_osc)*k_matrix*np.cos(phase_diff)
                row_sum = np.sum(cos_diff, axis=1) - np.diag((1/n_osc)*k_matrix)

                out = cos_diff
                #out[np.diag_indices(n_osc)] = -row_sum
                out = out*(1 - np.identity(n_osc)) + -row_sum*np.identity(n_osc)
                return out

        elif repulsion=="hard":
            @njit
            def ndot(phase_vals, tvals):
                """
                Automatically generated function for the dynamical equation
                """
                phase_diff = phase_vals.reshape((1, -1)) - phase_vals.reshape((-1, 1)) - a_vec.reshape((-1, 1))
                sin_diff = np.sin(phase_diff)

                phase_diff = mod_c(phase_diff, 2*np.pi) # correct way

                repel_term = repel_val*dtriangle_repel(phase_diff, repel_length*2*np.pi/n_osc)
                return w_vals + (1/n_osc)*np.sum(k_matrix*sin_diff, axis=1) - (1/num_repel)*np.sum(repel_term, axis=1) 
            @njit
            def jac(phase_vals, tvals):
                """
                Jacobian function for the dynamical equation (not implemented)
                """
                phase_diff = phase_vals.reshape((1, -1)) - phase_vals.reshape((-1, 1)) - a_vec.reshape((-1, 1))

                cos_diff = (1/n_osc)*k_matrix*np.cos(phase_diff)
                phase_diff = mod_c(phase_diff, 2*np.pi) # correct way
                
                interaction_term = cos_diff #- repel_term
                np.fill_diagonal(interaction_term, 0)
                row_sum = np.sum(interaction_term, axis=1)
                
                out = cos_diff
                #out[np.diag_indices(n_osc)] = -row_sum
                out = out*(1 - np.identity(n_osc)) + -row_sum*np.identity(n_osc)
                return out
            
        elif repulsion=="gaussian":
            @njit
            def ndot(phase_vals, tvals):
                """
                Automatically generated function for the dynamical equation
                """
                phase_diff = phase_vals.reshape((1, -1)) - phase_vals.reshape((-1, 1)) - a_vec.reshape((-1, 1))
                sin_diff = np.sin(phase_diff)

                phase_diff = mod_c(phase_diff, 2*np.pi) # correct way

                repel_term = repel_val*dgaussian_repel(phase_diff, repel_length*2*np.pi/n_osc)
                return w_vals + (1/n_osc)*np.sum(k_matrix*sin_diff, axis=1) - (1/num_repel)*np.sum(repel_term, axis=1) 
            @njit
            def jac(phase_vals, tvals):
                """
                Jacobian function for the dynamical equation (not implemented)
                """
                phase_diff = phase_vals.reshape((1, -1)) - phase_vals.reshape((-1, 1)) - a_vec.reshape((-1, 1))

                cos_diff = (1/n_osc)*k_matrix*np.cos(phase_diff)
                phase_diff = mod_c(phase_diff, 2*np.pi) # correct way
                repel_term = repel_val*ddgaussian_repel(phase_diff, repel_length*2*np.pi/n_osc)
                
                interaction_term = cos_diff - repel_term
                np.fill_diagonal(interaction_term, 0)
                row_sum = np.sum(interaction_term, axis=1)

                out = cos_diff
                #out[np.diag_indices(n_osc)] = -row_sum
                out = out*(1 - np.identity(n_osc)) + -row_sum*np.identity(n_osc)
                return out

        elif repulsion=="cauchy":
            @njit
            def ndot(phase_vals, tvals):
                """
                Automatically generated function for the dynamical equation
                """
                phase_diff = phase_vals.reshape((1, -1)) - phase_vals.reshape((-1, 1)) - a_vec.reshape((-1, 1))
                sin_diff = np.sin(phase_diff)

                phase_diff = mod_c(phase_diff, 2*np.pi) # correct way

                repel_term = repel_val*dcauchy_repel(phase_diff, repel_length*2*np.pi/n_osc)
                return w_vals + (1/n_osc)*np.sum(k_matrix*sin_diff, axis=1) - (1/num_repel)*np.sum(repel_term, axis=1) 
            @njit
            def jac(phase_vals, tvals):
                """
                Jacobian function for the dynamical equation (not implemented)
                """
                phase_diff = phase_vals.reshape((1, -1)) - phase_vals.reshape((-1, 1)) - a_vec.reshape((-1, 1))

                cos_diff = (1/n_osc)*k_matrix*np.cos(phase_diff)
                phase_diff = mod_c(phase_diff, 2*np.pi) # correct way
                repel_term = repel_val*ddcauchy_repel(phase_diff, repel_length*2*np.pi/n_osc)
                
                interaction_term = cos_diff - repel_term
                np.fill_diagonal(interaction_term, 0)
                row_sum = np.sum(interaction_term, axis=1)

                out = cos_diff
                #out[np.diag_indices(n_osc)] = -row_sum
                out = out*(1 - np.identity(n_osc)) + -row_sum*np.identity(n_osc)
                return out
        else:
            warnings.warn("repulsion method not recognized, defaulting to no repulsion.")
            @njit
            def ndot(phase_vals, tvals):
                """
                Kuramoto model
                """
                # phase_vals = np.mod(phase_vals, 2*np.pi)
                phase_diff = phase_vals.reshape((1, -1)) - phase_vals.reshape((-1, 1)) - a_vec.reshape((-1, 1))
                sin_diff = np.sin(phase_diff)
                return w_vals + (1/n_osc)*np.sum(k_matrix*sin_diff, axis=1)
            @njit
            def jac(phase_vals, tvals):
                """
                Jacobian function for the dynamical equation (not implemented)
                """
                return None
        self.dyn_eq = ndot
        self.jac = jac

    

    

