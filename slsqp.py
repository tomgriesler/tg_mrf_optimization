"""
NOTE: The code in this script is based in part on code published by Philip Lee (https://github.com/ecat/adbs). 

This is an implementation of the SLSQP optimization algorithm as introduced by Dieter Kraft. I use a scipy implemtation, documentation can be found at https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html. This was my first project using advanced optimization and I'm sure there are more pythonic ways to write it, but as far as I know it does what it should. As optimize_sequence is the only function you should typically call, the typehints and docstrings for the other functions are rudimentary at best.

Tom Griesler, 05/24
tomgr@umich.edu
"""

import torch
import numpy as np
import scipy
from typing import List, Union, Optional

from signalmodel_bloch import calculate_crlb_fisp_sc_bloch, calculate_crlb_fisp_mc_bloch, calculate_orth_bloch
from signalmodel_epg import calculate_crlb_fisp_sc_epg, calculate_crlb_fisp_mc_epg, calculate_orth_epg
from tools import to_tensor


def calculate_cost(costfunction, weighting, t1, t2, m0, ratio, beats, shots, fa, tr, ph, prep, ti, t2te, te, inv_eff, delta_B1, n_iso, deph):

    if costfunction == 'crlb_sc_bloch':
        crlb = calculate_crlb_fisp_sc_bloch(t1, t2, m0, beats, shots, fa, tr, ph, prep, ti, t2te, te, inv_eff, delta_B1, n_iso, deph)
        weighting = to_tensor(weighting, torch.double)
        cost = torch.sum(torch.sqrt(torch.diagonal(crlb)) * weighting)

    elif costfunction == 'crlb_sc_epg': 
        crlb = calculate_crlb_fisp_sc_epg(t1, t2, m0, beats, shots, fa, tr, ph, prep, ti, t2te, te, inv_eff, delta_B1)
        weighting = to_tensor(weighting, torch.double)
        cost = torch.sum(torch.sqrt(torch.diagonal(crlb)) * weighting)

    elif costfunction == 'crlb_mc_bloch': 
        crlb = calculate_crlb_fisp_mc_bloch(t1, t2, m0, ratio, beats, shots, fa, tr, ph, prep, ti, t2te, te, inv_eff, delta_B1, n_iso, deph)
        cost = torch.sqrt(crlb)

    elif costfunction == 'crlb_mc_epg':
        crlb = calculate_crlb_fisp_mc_epg(t1, t2, m0, ratio, beats, shots, fa, tr, ph, prep, ti, t2te, te, inv_eff, delta_B1)
        cost = torch.sqrt(crlb)    

    elif costfunction == 'orth_bloch': 
        cost = calculate_orth_bloch(t1, t2, beats, shots, fa, tr, ph, prep, ti, t2te, te, inv_eff, delta_B1, n_iso, deph)

    elif costfunction == 'orth_epg': 
        cost = calculate_orth_epg(t1, t2, beats, shots, fa, tr, ph, prep, ti, t2te, te, inv_eff, delta_B1)

    return cost


def cost_wrapper(fatr: np.ndarray, *args):
    """
    Wrapper function for the cost function to be used in the SLSQP algorithm.
    """
    
    fa, tr = np.split(fatr, 2)
    
    fa = torch.tensor(fa, dtype=torch.double, requires_grad=True)
    tr = torch.tensor(tr, dtype=torch.double, requires_grad=True)

    costfunction, weighting, t1, t2, m0, ratio, beats, shots, ph, prep, ti, t2te, te, inv_eff, delta_B1, n_iso, deph, slsqp_scaling, _, _ = args

    return calculate_cost(costfunction, weighting, t1, t2, m0, ratio, beats, shots, fa, tr, ph, prep, ti, t2te, te, inv_eff, delta_B1, n_iso, deph).detach().numpy() * slsqp_scaling


def grad_wrapper(fatr: np.ndarray, *args):
    """
    Wrapper function for the gradient of the cost function to be used in the SLSQP algorithm.
    """

    fa, tr = np.split(fatr, 2)

    fa = torch.tensor(fa, dtype=torch.double, requires_grad=True)
    tr = torch.tensor(tr, dtype=torch.double, requires_grad=True)

    costfunction, weighting, t1, t2, m0, ratio, beats, shots, ph, prep, ti, t2te, te, inv_eff, delta_B1, n_iso, deph, slsqp_scaling, optimize_tr, _ = args

    cost = calculate_cost(costfunction, weighting, t1, t2, m0, ratio, beats, shots, fa, tr, ph, prep, ti, t2te, te, inv_eff, delta_B1, n_iso, deph) * slsqp_scaling

    cost.backward()

    if not optimize_tr:
        tr.grad[:] = 0

    return np.concatenate((fa.grad, tr.grad), axis=0)


def f_ieqcons_function(fatr: np.ndarray, *args):
    """
    Function to be used in the SLSQP algorithm to define inequality constraints.
    """

    (_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, fa_maxdiff) = args

    fa, _ = np.split(fatr, 2)

    etl = len(fa)

    out = np.zeros(etl)

    if fa_maxdiff:
        fa_constraint = fa_maxdiff - np.abs(fa[1:]-fa[0:(etl-1)])
        out[1:etl] = fa_constraint

    return out


def optimize_sequence(costfunction: str, t1: Union[float, List[float]], t2: Union[float, List[float]], m0: float, beats: int, shots: int, fa: np.ndarray, tr: np.ndarray, ph: np.ndarray, prep: List[int], ti: List[int], t2te: List[int], te: float, fa_min: float, fa_max: float, fa_maxdiff: float, n_iter_max: int, weighting: Optional[List[float]]=None, ratio: Optional[float]=None, slsqp_scaling: float=1e3, optimize_tr: bool=False, tr_min: float=0, tr_max: float=torch.inf, inv_eff: float=1., delta_B1: float=1., n_iso: int=200, deph: int=2, acc: float=1e-4):
    """
    Function to optimize the flip angles and repetition times in an MRF sequence using the SLSQP algorithm.

    Args:
        costfunction (str): cost function to be used, must be 'crlb_sc_bloch', 'crlb_sc_epg', 'crlb_mc_bloch', 'crlb_mc_epg', 'orth_bloch', 'orth_epg'
        t1 (float/list of floats): target longitudinal relaxation time(s) in ms
        t2 (float/list of floats): target transversal relaxation time(s) in ms
        m0 (float): equilibrium magnetization
        beats (int): number of magnetization prepared blocks
        shots (int): number of excitations per block
        fa (array): initial flip angles in degrees
        tr (array): initial repetition times in ms
        ph (array): excitation phases in degrees
        prep (list of int): types of magnetization preparations (0: no prep, 1: inversion, 2: t2 prep)
        ti (list of int): inversion times in ms
        t2te (list of int): t2 prep times in ms
        te (float): echo time in ms
        fa_min (float): minimal allowed flip angle in degrees
        fa_max (float): maximal allowed flip angle in degrees
        fa_maxdiff (float): maximal allowed flip angle difference between two consecutive excitations in degrees
        n_iter_max (int): maximal number of iterations

    Keyword Args: 
        weighting (list of floats): diagonal entries of the CRLB weighting matrix. Only necessary for 'crlb_sc_bloch' and 'crlb_sc_epg' cost functions
        ratio (float): ratio of the first component of the total voxel. Only necessary for 'crlb_mc_bloch' and 'crlb_mc_epg' cost functions
        slsqp_scaling (float): The SLSQP algorithm doesn't have a learning rate parameter to control the convergence behavior. However, convergence depends on the absolute values of the gradients and can be artificially controlled by multiplying the cost function values by a factor. Appropriate values depend on the used cost function and have to be determined somewhat heuristically by monitoring optimization behavior. 
        optimize_tr (bool): if set to False, only flip angles are optimized
        tr_min (float): minimal allowed repetition time in ms
        tr_max (float): maximal allowed repetition time in ms
        inv_eff (float): inversion efficiency
        delta_B1 (float): B1 correction factor
        n_iso (int): number of isochromats. Only used when using isochromat summation 
        deph (int): dephasing across voxel in multiples of pi. Only used when using isochromat summation 
        acc (float): accuracy, controls convergence

    Returns:
        fa (ndarray): optimized flip angles in degrees
        tr (ndarray): optimized repetition times in ms
        fx (float): final value of the cost function
        its (int): number of iterations
        smode (int): termination mode of the optimization        
    """

    # Concatenate flip angles and repetition times
    fatr_init = np.concatenate((fa, tr), axis=0)

    # Set bounds
    fa_min = fa_min if fa_min else 0
    tr_min = tr_min if tr_min else 0
    fa_max = fa_max if fa_max else np.inf
    tr_max = tr_max if tr_max else np.inf

    fa_bounds = [(fa_min, fa_max) for _ in range(0, len(fa))]
    tr_bounds = [(tr_min, tr_max) for _ in range(0, len(tr))]

    bounds = fa_bounds + tr_bounds

    # Create args tuple
    args = costfunction, weighting, t1, t2, m0, ratio, beats, shots, ph, prep, ti, t2te, te, inv_eff, delta_B1, n_iso, deph, slsqp_scaling, optimize_tr, fa_maxdiff

    # Run optimization
    fatr_final, fx, its, _, smode = scipy.optimize.fmin_slsqp(cost_wrapper, fatr_init, bounds=bounds, acc=acc, iprint=2, args=args, iter=n_iter_max, fprime=grad_wrapper, f_ieqcons=f_ieqcons_function, full_output=True)
    
    fa, tr = np.split(fatr_final, 2)

    return fa, tr, fx, its, smode