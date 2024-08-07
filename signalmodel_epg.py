"""
NOTE: A lot of the implementations of the torch tensors in this script may look a bit funny and inefficient (i.e. using stack operations to create multidimensional tensors instead of defining the tensors in a more straightforward way). I found this to be necessary to retain the gradients for the backpropagation step. Also, in place operations (e.g., +=) aren't used for the same reason.

Tom Griesler, 05/24
tomgr@umich.edu
"""

import torch
import numpy as np
from typing import Union, List

from tools import to_tensor


# create zero tensor
zero = torch.tensor(0, dtype=torch.cfloat)


def q_epg(alpha: torch.tensor, phi=torch.tensor(np.pi/2)):
    """
    Calculate EPG excitation matrix with flip angle alpha and phase phi.

    Args:
        alpha (tensor): flip angle in rad
        
    Keyword Args: 
        phi (tensor): phase in rad

    Returns:
        (tensor): excitation matrix
    """

    # The conjugate operation basically changes the sign of the complex part of the resulting signal and has been added for consistency with the isochromat implementation. 
    return torch.conj(torch.stack([
        torch.stack([
            (torch.cos(alpha/2.))**2., 
            torch.exp(2.*1j*phi)*(torch.sin(alpha/2.))**2., 
            -1j*torch.exp(1j*phi)*torch.sin(alpha)
        ]), 
        torch.stack([
            torch.exp(-2.*1j*phi)*(torch.sin(alpha/2.))**2., 
            (torch.cos(alpha/2.))**2., 
            1j*torch.exp(-1j*phi)*torch.sin(alpha)
        ]),
        torch.stack([
            -1j/2.*torch.exp(-1j*phi)*torch.sin(alpha), 
            1j/2.*torch.exp(1j*phi)*torch.sin(alpha), 
            torch.cos(alpha)])
        ]).type(torch.cfloat))


def r_epg(t1: torch.tensor, t2: torch.tensor, t: torch.tensor):
    """
    Calculate relaxation matrix with time constants T1 and T2.

    Args:
        t1 (tensor): longitudinal relaxation time in ms
        t2 (tensor): transversal relaxation time in ms
        t (tensor): time in ms

    Returns:
        (tensor): relaxation matrix
    """

    E1 = torch.exp(-t/t1)
    E2 = torch.exp(-t/t2)

    return torch.stack([torch.stack([E2, zero, zero]),
                        torch.stack([zero, E2, zero]),
                        torch.stack([zero, zero, E1])]).type(torch.cfloat)


def dr_dt1_epg(t1: torch.tensor, t: torch.tensor):
    """
    Derivative of relaxation matrix with respect to T1.

    Args:
        t1 (tensor): longitudinal relaxation time in ms
        t (tensor): time in ms

    Returns:
        (tensor): derivative of relaxation matrix with respect to T1
    """

    return t/t1**2 * torch.exp(-t/t1) * torch.tensor([[0, 0, 0],
                                                      [0, 0, 0],
                                                      [0, 0, 1]],
                                                      dtype=torch.cfloat)


def dr_dt2_epg(t2: torch.tensor, t: torch.tensor):
    """
    Derivative of relaxation matrix with respect to T2.

    Args:
        t2 (tensor): transversal relaxation time in ms
        t (tensor): time in ms

    Returns:
        (tensor): derivative of relaxation matrix with respect to T2
    """

    return t/t2**2 * torch.exp(-t/t2) * torch.tensor([[1, 0, 0],
                                                      [0, 1, 0],
                                                      [0, 0, 0]],
                                                      dtype=torch.cfloat)


def b_epg(t1: torch.tensor, t: torch.tensor):
    """
    Calculate longitudinal relaxation term.

    Args:
        t1 (tensor): longitudinal relaxation time in ms
        t (tensor): time in ms

    Returns:
        (tensor): longitudinal relaxation term 
    """
    return (1-torch.exp(-t/t1))


def db_dt1_epg(t1: torch.tensor, t: torch.tensor): 
    """
    Derivative of longitudinal relaxation term with respect to T1.

    Args: 
        t1 (tensor): longitudinal relaxation time in ms
        t (tensor): time in ms

    Returns:
        (tensor): derivative of longitudinal relaxation term with respect to T1

    """

    return (-t/t1**2 * torch.exp(-t/t1))


def epg_grad(omega: torch.tensor):
    """
    Apply gradient operation to EPG state matrix.

    Args:
        omega (tensor): current state matrix

    Returns: 
        (tensor): state matrix after one gradient cycle
    """

    omega = torch.hstack([omega, torch.tensor([[0],[0],[0]])])
    omega[0, 1:] = omega.clone()[0, :-1]
    omega[1, :-1] = omega.clone()[1, 1:]
    omega[1, -1] = 0
    omega[0, 0] = torch.conj(omega[1,0])
    return omega


def inversion_epg(inv_eff: torch.tensor):
    """
    Calculate inversion matrix. 

    Args: 
        inv_eff (tensor): inversion efficiency

    Returns: 
        (tensor): inversion matrix
    """

    return torch.stack([torch.stack([zero, zero, zero]),
                        torch.stack([zero, zero, zero]),
                        torch.stack([zero, zero, -inv_eff])]).type(torch.cfloat)


def t2prep_epg(t2: torch.tensor, t2te: torch.tensor):
    """
    Calculate T2 preparation matrix. 

    Args: 
        t2 (tensor): transversal relaxation time in ms
        t2te (tensor): preparation time in ms

    Returns: 
        (tensor): T2 preparation matrix
    """
    
    return torch.stack([torch.stack([zero, zero, zero]),
                        torch.stack([zero, zero, zero]),
                        torch.stack([zero, zero, -t2te/t2])]).type(torch.cfloat)


def dt2prep_dt2_epg(t2: torch.tensor, t2te: torch.tensor):
    """
    Derivative of T2 preparation matrix with respect to T2.

    Args: 
        t2 (tensor): transversal relaxation time in ms
        t2te (tensor): preparation time in ms

    Returns: 
        (tensor): Derivative of T2 preparation matrix with respect to T2
    """
    
    return torch.stack([torch.stack([zero, zero, zero]),
                        torch.stack([zero, zero, zero]),
                        torch.stack([zero, zero, t2te/t2**2 * np.exp(-t2te/t2)])]).type(torch.cfloat)


def calculate_signal_fisp_epg(t1: float, t2:float, m0: float, beats: int, shots: int, fa: Union[np.ndarray, torch.tensor], tr: Union[np.ndarray, torch.tensor], ph: Union[np.ndarray, torch.tensor], prep: List[int], ti: List[int], t2te: List[int], te: float, inv_eff: float=1., delta_B1: float=1.):
    """
    Function to calculate the resulting complex signal of a voxel for a given MRF sequence using EPG.

    Args:
        t1 (float): longitudinal relaxation time in ms
        t2 (float): transversal relaxation time in ms 
        m0 (float): equilibrium magnetization
        beats (int): number of magnetization prepared blocks
        shots (int): number of excitations per block
        fa (array/tensor): flip angles in degrees
        tr (array/tensor): repetition times in ms
        ph (array/tensor): excitation phases in degrees
        prep (list of int): types of magnetization preparations (0: no prep, 1: inversion, 2: t2 prep)
        ti (list of int): inversion times in ms
        t2te (list of int): t2 prep times in ms
        te (float): echo time in ms

    Keyword Args: 
        inv_eff (float): inversion efficiency
        delta_B1 (float): B1 correction factor

    Returns:
        (tensor): resulting signal at every echo time
    """

    # Convert to tensors
    t1 = to_tensor(t1, dtype=torch.double)
    t2 = to_tensor(t2, dtype=torch.double)
    m0 = to_tensor(m0, dtype=torch.double)
    fa = to_tensor(fa, dtype=torch.double)
    tr = to_tensor(tr, dtype=torch.double)
    ph = to_tensor(ph, dtype=torch.double)
    te = to_tensor(te, dtype=torch.double)
    inv_eff = to_tensor(inv_eff, dtype=torch.double)

    # Calculate relaxation matrix, longitudinal relaxation term, and inversion matrix
    r_te = r_epg(t1, t2, te)
    b_te = b_epg(t1, te)
    inv = inversion_epg(inv_eff)

    # Initial state matrix
    omega = torch.vstack([zero, zero, m0]).type(torch.cfloat)

    # Initialize signal tensor
    signal = torch.empty(beats*shots, dtype=torch.cfloat)

    # Iterate over magnetization prepared blocks
    for ii in range(beats):

        # Inversion preparation
        if prep[ii] == 1:
            omega = r_epg(t1, t2, ti[ii]) @ inv @ omega
            omega[2, 0] = omega[2, 0] + m0*b_epg(t1, ti[ii])

        # T2 preparation
        elif prep[ii] == 2:
            omega = t2prep_epg(t2, t2te[ii]) @ omega

        # Iterate over excitations
        for jj in range(shots):

            # Index of current excitation
            n = ii*shots + jj

            # Excitation matrix
            q_n = q_epg(delta_B1*torch.deg2rad(fa[n]), torch.deg2rad(ph[n]))

            # Update state matrix (excitation, relaxation during TE)
            omega = r_te @ q_n @ omega
            omega[2, 0] = omega[2, 0] + m0*b_te
            
            # Calculate complex signal
            signal[n] = omega[0,0] * torch.exp(-1j*torch.deg2rad(ph[n]))

            # Update state matrix (relaxation during TR-TE, gradient dephasing)
            omega = epg_grad(r_epg(t1, t2, tr[n]-te) @ omega)
            omega[2, 0] = omega[2, 0] + m0*b_epg(t1, tr[n]-te)

    return signal


def calculate_crlb_fisp_sc_epg(t1: float, t2:float, m0: float, beats: int, shots: int, fa: Union[np.ndarray, torch.tensor], tr: Union[np.ndarray, torch.tensor], ph: Union[np.ndarray, torch.tensor], prep: List[int], ti: List[int], t2te: List[int], te: float, inv_eff: float=1., delta_B1: float=1.):
    """
    Function to calculate the CRLB matrix of a single component voxel for a given MRF sequence using EPG.

    Args:
        t1 (float): longitudinal relaxation time in ms
        t2 (float): transversal relaxation time in ms 
        m0 (float): equilibrium magnetization
        beats (int): number of magnetization prepared blocks
        shots (int): number of excitations per block
        fa (array/tensor): flip angles in degrees
        tr (array/tensor): repetition times in ms
        ph (array/tensor): excitation phases in degrees
        prep (list of int): types of magnetization preparations (0: no prep, 1: inversion, 2: t2 prep)
        ti (list of int): inversion times in ms
        t2te (list of int): t2 prep times in ms
        te (float): echo time in ms

    Keyword Args: 
        inv_eff (float): inversion efficiency
        delta_B1 (float): B1 correction factor

    Returns:
        (tensor): CRLB matrix
    """

    # Convert to tensors
    t1 = to_tensor(t1, dtype=torch.double)
    t2 = to_tensor(t2, dtype=torch.double)
    m0 = to_tensor(m0, dtype=torch.double)
    fa = to_tensor(fa, dtype=torch.double)
    tr = to_tensor(tr, dtype=torch.double)
    ph = to_tensor(ph, dtype=torch.double)
    te = to_tensor(te, dtype=torch.double)
    inv_eff = to_tensor(inv_eff, dtype=torch.double)

    # Calculate relaxation matrix and inversion matrix
    r_te = r_epg(t1, t2, te)
    inv = inversion_epg(inv_eff)

    # Initial state matrix and derivatives
    omega = torch.vstack([zero, zero, m0]).type(torch.cfloat)
    domega_dt1 = torch.zeros(3, 1, dtype=torch.cfloat)
    domega_dt2 = torch.zeros(3, 1, dtype=torch.cfloat)
    domega_dm0 = torch.vstack([zero, zero, torch.tensor(1., dtype=torch.cfloat)]).type(torch.cfloat)

    # Initialize Fisher Information Matrix
    fim = torch.zeros(3, 3, dtype=torch.double)

    # Iterate over magnetization prepared blocks
    for ii in range(beats):

        # Inversion preparation
        if prep[ii] == 1:

            r_ti = r_epg(t1, t2, ti[ii])

            domega_dt1 = dr_dt1_epg(t1, ti[ii]) @ inv @ omega + r_ti @ inv @ domega_dt1
            domega_dt1[2, 0] = domega_dt1[2, 0] + m0 * db_dt1_epg(t1, ti[ii])

            domega_dt2 = dr_dt2_epg(t2, ti[ii]) @ inv @ omega + r_ti @ inv @ domega_dt2

            domega_dm0 = r_ti @ inv @ domega_dm0

            omega = r_ti @ inv @ omega
            omega[2, 0] = omega[2, 0] + m0 * b_epg(t1, ti[ii])

        # T2 preparation
        elif prep[ii] == 2:

            t2prep_ii = t2prep_epg(t2, t2te[ii])
            dt2prep_ii_dt2 = dt2prep_dt2_epg(t2, t2te[ii])

            domega_dt1 = t2prep_ii @ domega_dt1

            domega_dt2 = dt2prep_ii_dt2 @ omega + t2prep_ii @ domega_dt2

            domega_dm0 = t2prep_ii @ domega_dm0

            omega = t2prep_ii @ omega

        # Iterate over excitations
        for jj in range(shots):
            
            # Index of current excitation
            n = ii*shots + jj

            # Excitation matrix
            q_n = q_epg(delta_B1 * torch.deg2rad(fa[n]), torch.deg2rad(ph[n]))

            # Derivatives of signal
            dsignal_dt1 = (r_te @ q_n @ domega_dt1)[0, 0]
            dsignal_dt2 = (dr_dt2_epg(t2, te) @ q_n @ omega + r_te @ q_n @ domega_dt2)[0, 0]
            dsignal_dm0 = (r_te @ q_n @ domega_dm0)[0, 0]

            # Calculate Jacobian
            J_n = torch.stack([torch.stack([torch.real(dsignal_dt1), torch.real(dsignal_dt2), torch.real(dsignal_dm0)]), torch.stack([torch.imag(dsignal_dt1), torch.imag(dsignal_dt2), torch.imag(dsignal_dm0)])])

            # Update FIM
            fim = fim + torch.t(J_n) @ J_n

            # Calculate new state matrix and derivatives
            r_tr = r_epg(t1, t2, tr[n])
            b_tr = m0 * b_epg(t1, tr[n])
            
            domega_dt1 = epg_grad(dr_dt1_epg(t1, tr[n]) @ q_n @ omega + r_tr @ q_n @ domega_dt1) 
            domega_dt1[2, 0] = domega_dt1[2, 0] + m0 * db_dt1_epg(t1, tr[n])

            domega_dt2 = epg_grad(dr_dt2_epg(t2, tr[n]) @ q_n @ omega + r_tr @ q_n @ domega_dt2)

            domega_dm0 = epg_grad(r_tr @ q_n @ domega_dm0)
            domega_dm0[2, 0] = domega_dm0[2, 0] + b_tr/m0          

            omega = epg_grad(r_tr @ q_n @ omega)
            omega[2, 0] = omega[2, 0] + b_tr

    # Invert FIM
    v = torch.linalg.inv(fim)

    return v  


def calculate_crlb_fisp_mc_epg(t1: List[float], t2: List[float], m0: float, ratio: float, beats: int, shots: int, fa: Union[np.ndarray, torch.tensor], tr: Union[np.ndarray, torch.tensor], ph: Union[np.ndarray, torch.tensor], prep: List[int], ti: List[int], t2te: List[int], te: float, inv_eff: float=1., delta_B1: float=1.):
    """
    Function to calculate the CRLB of a two component voxel for a given MRF sequence using EPG.

    Args:
        t1 (list of float): longitudinal relaxation times in ms
        t2 (list of float): transversal relaxation times in ms 
        m0 (float): equilibrium magnetization
        ratio (float): ratio of the first component of the total voxel
        beats (int): number of magnetization prepared blocks
        shots (int): number of excitations per block
        fa (array/tensor): flip angles in degrees
        tr (array/tensor): repetition times in ms
        ph (array/tensor): excitation phases in degrees
        prep (list of int): types of magnetization preparations (0: no prep, 1: inversion, 2: t2 prep)
        ti (list of int): inversion times in ms
        t2te (list of int): t2 prep times in ms
        te (float): echo time in ms

    Keyword Args: 
        inv_eff (float): inversion efficiency
        delta_B1 (float): B1 correction factor

    Returns:
        (tensor): CRLB
    """

    # Convert to tensors
    t1 = to_tensor(t1, dtype=torch.double)
    t2 = to_tensor(t2, dtype=torch.double)
    m0 = to_tensor(m0, dtype=torch.double)
    fa = to_tensor(fa, dtype=torch.double)
    tr = to_tensor(tr, dtype=torch.double)
    ph = to_tensor(ph, dtype=torch.double)
    te = to_tensor(te, dtype=torch.double)
    inv_eff = to_tensor(inv_eff, dtype=torch.double)

    # Calculate relaxation matrices and inversion matrix
    r_te_1 = r_epg(t1[0], t2[0], te)
    r_te_2 = r_epg(t1[1], t2[1], te)
    inv = inversion_epg(inv_eff)

    # Initial state matrices and derivatives
    omega_1 = torch.vstack([zero, zero, ratio*m0]).type(torch.cfloat)
    omega_2 = torch.vstack([zero, zero, (1-ratio)*m0]).type(torch.cfloat)

    domega_1_dratio = torch.vstack([zero, zero, m0]).type(torch.cfloat)
    domega_2_dratio = torch.vstack([zero, zero, -m0]).type(torch.cfloat)

    # Initialize Fisher Information Matrix
    fim = torch.zeros(1, dtype=torch.double)

    # Iterate over magnetization prepared blocks
    for ii in range(beats):
        
        # Inversion preparation
        if prep[ii] == 1:

            r_ti_1 = r_epg(t1[0], t2[0], ti[ii])
            r_ti_2 = r_epg(t1[1], t2[1], ti[ii])
            b_ti_1 = b_epg(t1[0], ti[ii])
            b_ti_2 = b_epg(t1[1], ti[ii])

            domega_1_dratio = r_ti_1 @ inv @ domega_1_dratio
            domega_1_dratio[2, 0] = domega_1_dratio[2, 0] + m0*b_ti_1

            domega_2_dratio = r_ti_2 @ inv @ domega_2_dratio
            domega_2_dratio[2, 0] = domega_2_dratio[2, 0] - m0*b_ti_2

            omega_1 = r_ti_1 @ inv @ omega_1
            omega_1[2, 0] = omega_1[2, 0] + ratio*m0*b_ti_1

            omega_2 = r_ti_2 @ inv @ omega_2
            omega_2[2, 0] = omega_2[2, 0] + (1-ratio)*m0*b_ti_2

        # T2 preparation
        elif prep[ii] == 2:

            t2prep_ii_1 = t2prep_epg(t2[0], t2te[ii])
            t2prep_ii_2 = t2prep_epg(t2[1], t2te[ii])

            domega_1_dratio = t2prep_ii_1 @ domega_1_dratio
            domega_2_dratio = t2prep_ii_2 @ domega_2_dratio

            omega_1 = t2prep_ii_1 @ omega_1
            omega_2 = t2prep_ii_2 @ omega_2

        # Iterate over excitations
        for jj in range(shots):

            # Index of current excitation
            n = ii*shots + jj
            
            # Excitation matrix
            q_n = q_epg(delta_B1*torch.deg2rad(fa[n]), torch.deg2rad(ph[n]))

            # Derivative of signal with respect to tissue ratio
            dsignal_dratio = (r_te_1 @ q_n @ domega_1_dratio)[0, 0] + (r_te_2 @ q_n @ domega_2_dratio)[0, 0]

            # Update FIM
            fim = fim + torch.abs(dsignal_dratio)**2

            # Calculate new state matrices and derivatives
            r_tr_1 = r_epg(t1[0], t2[0], tr[n])
            r_tr_2 = r_epg(t1[1], t2[1], tr[n])
            b_tr_1 = b_epg(t1[0], tr[n])
            b_tr_2 = b_epg(t1[1], tr[n])

            domega_1_dratio = epg_grad(r_tr_1 @ q_n @ domega_1_dratio)
            domega_1_dratio[2, 0] = domega_1_dratio[2, 0] + m0*b_tr_1
            domega_2_dratio = epg_grad(r_tr_2 @ q_n @ domega_2_dratio)
            domega_2_dratio[2, 0] = domega_2_dratio[2, 0] - m0*b_tr_2

            omega_1 = epg_grad(r_tr_1 @ q_n @ omega_1)
            omega_1[2, 0] = omega_1[2, 0] + ratio*m0*b_tr_1
            omega_2 = epg_grad(r_tr_2 @ q_n @ omega_2)
            omega_2[2, 0] = omega_2[2, 0] + (1-ratio)*m0*b_tr_2

    return 1/fim

            
def calculate_orth_epg(t1: List[float], t2: List[float], beats: int, shots: int, fa: Union[np.ndarray, torch.tensor], tr: Union[np.ndarray, torch.tensor], ph: Union[np.ndarray, torch.tensor], prep: List[int], ti: List[int], t2te: List[int], te: float, inv_eff: float=1., delta_B1: float=1.):
    """
    Function to calculate the orthogonality cost function for a number of tissues for a given MRF sequence using Boch/isochromat summation. 

    Args:
        t1 (list of float): longitudinal relaxation times in ms
        t2 (list of float): transversal relaxation times in ms 
        beats (int): number of magnetization prepared blocks
        shots (int): number of excitations per block
        fa (array/tensor): flip angles in degrees
        tr (array/tensor): repetition times in ms
        ph (array/tensor): excitation phases in degrees
        prep (list of int): types of magnetization preparations (0: no prep, 1: inversion, 2: t2 prep)
        ti (list of int): inversion times in ms
        t2te (list of int): t2 prep times in ms
        te (float): echo time in ms

    Keyword Args: 
        inv_eff (float): inversion efficiency
        delta_B1 (float): B1 correction factor

    Returns:
        (tensor): orthogonality cost function
    """

    # Number of different tissues
    n_tissues = len(t1)

    # Initialize signal matrix
    s = torch.zeros((n_tissues, beats*shots), dtype=torch.cfloat)

    # Calculate signals for all tissues
    for ii in range(n_tissues):
        # Set m0 to 1 as the signals will be normalized anyway
        s[ii] = calculate_signal_fisp_epg(t1[ii], t2[ii], 1, beats, shots, fa, tr, ph, prep, ti, t2te, te, inv_eff, delta_B1)

    # Normalize signals
    s = torch.nn.functional.normalize(s, dim=1)

    # Calculate orthogonality matrix
    d = torch.norm(torch.eye(len(t1)) - s @ torch.t(torch.conj(s)))

    return d