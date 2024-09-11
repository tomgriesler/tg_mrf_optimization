"""
NOTE: The following code has mainly been implemented to show the equivalence of the EPG and isochromat approaches. It works, but be aware that I did not put much thought into making it efficient, using it leads to very lenghthy computation times. Use of the corresponding EPG implementation is strongly recommended. 

NOTE: In this script, I use the term 'signal' to refer to a complex valued 1xN array where the real part is the magnetization in x- and the imaginary part the magnetization in y-direction. The term 'm_trans' or 'transversal magnetization' is a real valued 2xN array/tensor where the first row is the magnetization in x- and the second row the magnetization in y-direction. 

Tom Griesler, 05/24
tomgr@umich.edu
"""

import torch
import numpy as np
from typing import Union, List

from tools import to_tensor


# create zero and one tensors
zero = torch.tensor(0, dtype=torch.double)
one = torch.tensor(1, dtype=torch.double)


def q_iso(alpha: torch.tensor, phi: torch.tensor=torch.tensor(np.pi/2)):
    """
    Calculate Bloch excitation matrix with flip angle alpha and phase phi.

    Args: 
        alpha (tensor): flip angle in rad
        
    Keyword Args: 
        phi (tensor): phase in rad

    Returns: 
        (tensor): excitation matrix
    """
    
    sinphi   = torch.sin(phi)
    cosphi   = torch.cos(phi)
    sinalpha = torch.sin(alpha)
    cosalpha = torch.cos(alpha)

    M1 = torch.stack([torch.stack([cosphi, sinphi, zero]),
                      torch.stack([-sinphi, cosphi, zero]),
                      torch.stack([zero, zero, one])])
    
    M2 = torch.stack([torch.stack([one, zero, zero]),
                      torch.stack([zero, cosalpha, sinalpha]),
                      torch.stack([zero, -sinalpha, cosalpha])])

    M3 = torch.stack([torch.stack([cosphi, -sinphi, zero]),
                      torch.stack([sinphi, cosphi, zero]),
                      torch.stack([zero, zero, one])])

    return M1 @ M2 @ M3


def r_iso(t1: torch.tensor, t2: torch.tensor, t: torch.tensor):
    """
    Calculate relaxation matrix with time constants T1 and T2

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
                        torch.stack([zero, zero, E1])])


def dr_dt1_iso(t1: torch.tensor, t: torch.tensor):
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
                                                      dtype=torch.double)


def dr_dt2_iso(t2: torch.tensor, t: torch.tensor):
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
                                                      dtype=torch.double)


def b_iso(t1: torch.tensor, t: torch.tensor):
    """
    Calculate longitudinal relaxation term.

    Args:
        t1 (tensor): longitudinal relaxation time in ms
        t (tensor): time in ms

    Returns:
        (tensor): longitudinal relaxation term 
    """
    return (1-torch.exp(-t/t1)) * torch.tensor([[0], [0], [1]], dtype=torch.double)


def db_dt1_iso(t1: torch.tensor, t: torch.tensor):
    """
    Derivative of longitudinal relaxation term with respect to T1.

    Args:
        t1 (tensor): longitudinal relaxation time in ms
        t (tensor): time in ms

    Returns:
        (tensor): derivative of longitudinal relaxation term with respect to T1
    """
    return -t/t1**2 * torch.exp(-t/t1) * torch.tensor([[0], [0], [1]], dtype=torch.double)


def g(beta_r: torch.tensor):
    """
    Rotation matrix around z axis with angle beta_r. Used to simulate spin dephasing. 

    Args:
        beta_r (tensor): rotation angle in rad

    Returns:    
        (tensor): dephasing matrix
    """
    sinbeta = torch.sin(beta_r)
    cosbeta = torch.cos(beta_r)

    return torch.tensor([[cosbeta, sinbeta, 0],
                         [-sinbeta, cosbeta, 0],
                         [0, 0, 1]], 
                         dtype=torch.double)


def inversion_iso(inv_eff: torch.tensor):
    """
    Calculate inversion matrix. 

    Args: 
        inv_eff (tensor): inversion efficiency

    Returns: 
        (tensor): inversion matrix
    """

    return torch.stack([torch.stack([zero, zero, zero]),
                        torch.stack([zero, zero, zero]),
                        torch.stack([zero, zero, -inv_eff])]).type(torch.double)


def t2prep_iso(t2: torch.tensor, t2te: torch.tensor):
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
                        torch.stack([zero, zero, -t2te/t2])]).type(torch.double)


def dt2prep_dt2_iso(t2: torch.tensor, t2te: torch.tensor):
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
                        torch.stack([zero, zero, t2te/t2**2 * np.exp(-t2te/t2)])]).type(torch.double)


# Projection operator (project three dimensional magnetization to transverse plane)
P = torch.tensor([[1, 0, 0],
                  [0, 1, 0]], 
                  dtype=torch.double)


def calculate_signal_iso(t1: float, t2: float, m0: float, beats: int, shots: int, fa: Union[np.ndarray, torch.tensor], tr: Union[np.ndarray, torch.tensor], ph: Union[np.ndarray, torch.tensor], prep: List[int], ti: List[int], t2te: List[int], te: float, inv_eff: float=1., delta_B1: float=1., n_iso: int=200, deph: int=2):
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
        n_iso (int): number of isochromats
        deph (int): dephasing across voxel in multiples of pi

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

    # Calculate spin dephasing angle for every isochromat
    beta = torch.linspace(0, deph*np.pi, n_iso)

    # Calculate relaxation matrix, longitudinal relaxation term, and inversion matrix
    r_te = r_iso(t1, t2, te)
    b_te = b_iso(t1, te)
    inv = inversion_iso(inv_eff)

    # Initial magnetization
    m = torch.zeros(n_iso, 3, 1, dtype=torch.double)
    m[:, 2, :] = m0/n_iso

    # Initialize signal tensor
    signal = torch.empty(beats*shots, dtype=torch.cfloat)

    # Iterate over magnetization prepared blocks
    for ii in range(beats):

        # Inversion preparation
        if prep[ii] == 1:
            # potentially inefficient implementation, might be accelerated by using batch matrix multiplication
            for iso in range(n_iso):
                m[iso] = r_iso(t1, t2, ti[ii]) @ inv @ m[iso].clone() + m0/n_iso*b_iso(t1, ti[ii])

        # T2 preparation
        elif prep[ii] == 2:
            # potentially inefficient implementation, might be accelerated by using batch matrix multiplication
            for iso in range(n_iso):
                m[iso] = t2prep_iso(t2, t2te[ii]) @ m[iso].clone()

        # Iterate over excitations
        for jj in range(shots):
            
            # Index of current excitation
            n = ii*shots + jj
            
            # Excitation matrix
            q_n = q_iso(delta_B1*torch.deg2rad(fa[n]), torch.deg2rad(ph[n]))

            # Update magnetization (excitation, relaxation during TE) and calculate total transversal magnetization at TE
            m_trans = torch.zeros(2, 1)
            for iso in range(n_iso):
                m[iso] = r_te @ q_n @ m[iso].clone() + m0/n_iso*b_te
                m_trans = m_trans + P @ m[iso]

            # Calculate complex signal
            signal[n] = (m_trans[0] + 1j*m_trans[1]) * torch.exp(1j*torch.deg2rad(ph[n]))

            # Update magnetization (relaxation during TR-TE, gradient dephasing)
            for iso in range(n_iso):                
                m[iso] = g(beta[iso]) @ r_iso(t1, t2, tr[n]-te) @ m[iso].clone() + m0/n_iso*b_iso(t1, tr[n]-te)
            
    return signal


def calculate_crlb_sc_iso(t1, t2, m0, beats, shots, fa, tr, ph, prep, ti, t2te, te, inv_eff=1., delta_B1=1., n_iso=200, deph=2):
    """
    Function to calculate the CRLB matrix of a single component voxel for a given MRF sequence using Bloch/isochromat summation.

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
        n_iso (int): number of isochromats
        deph (int): dephasing across voxel in multiples of pi

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

    # Calculate spin dephasing angle for every isochromat
    beta = torch.linspace(0, deph*np.pi, n_iso)

    # Calculate relaxation matrix and inversion matrix
    r_te = r_iso(t1, t2, te)
    inv = inversion_iso(inv_eff)

    # Initial magnetization and derivatives
    m = torch.zeros(n_iso, 3, 1, dtype=torch.double)
    m[:, 2, :] = m0/n_iso

    dm_dt1 = torch.zeros(n_iso, 3, 1, dtype=torch.double)
    dm_dt2 = torch.zeros(n_iso, 3, 1, dtype=torch.double)
    dm_dm0 = torch.zeros(n_iso, 3, 1, dtype=torch.double)
    dm_dm0[:, 2, :] = 1/n_iso

    # Initialize Fisher Information Matrix
    fim = torch.zeros(3, 3, dtype=torch.double)

    # Iterate over magnetization prepared blocks
    for ii in range(beats):

        # Inversion preparation
        if prep[ii] == 1:

            r_ti = r_iso(t1, t2, ti[ii])

            for iso in range(n_iso):

                dm_dt1[iso] = dr_dt1_iso(t1, ti[ii]) @ inv @ m[iso].clone() + r_ti @ inv @ dm_dt1[iso].clone() + m0/n_iso*db_dt1_iso(t1, ti[ii])

                dm_dt2[iso] = dr_dt2_iso(t2, ti[ii]) @ inv @ m[iso].clone() + r_ti @ inv @ dm_dt2[iso].clone()

                dm_dm0[iso] = r_ti @ inv @ dm_dm0[iso].clone()

                m[iso] = r_ti @ inv @ m[iso].clone() + m0/n_iso*b_iso(t1, ti[ii])

        # T2 preparation
        elif prep[ii] == 2:
            
            t2prep_ii = t2prep_iso(t2, t2te[ii])
            
            for iso in range(n_iso):

                dm_dt1[iso] = t2prep_ii @ dm_dt1[iso].clone()

                dm_dt2[iso] = dt2prep_dt2_iso(t2, t2te[ii]) @ m[iso].clone() + t2prep_ii @ dm_dt2[iso].clone()

                dm_dm0[iso] = t2prep_ii @ dm_dm0[iso].clone()

                m[iso] = t2prep_ii @ m[iso].clone()
        
        # Iterate over excitations
        for jj in range(shots):

            # Index of current excitation
            n = ii*shots + jj

            # Excitation matrix
            q_n = q_iso(delta_B1*torch.deg2rad(fa[n]), torch.deg2rad(ph[n]))

            # Derivatives of transversal magnetization
            dm_trans_dt1 = torch.zeros(2, 1, dtype=torch.double)
            dm_trans_dt2 = torch.zeros(2, 1, dtype=torch.double)
            dm_trans_dm0 = torch.zeros(2, 1, dtype=torch.double)
            
            for iso in range(n_iso):

                dm_trans_dt1 = dm_trans_dt1 + P @ r_te @ q_n @ dm_dt1[iso].clone()

                dm_trans_dt2 = dm_trans_dt2 + P @ dr_dt2_iso(t2, te) @ q_n @ m[iso].clone() + P @ r_te @ q_n @ dm_dt2[iso].clone()

                dm_trans_dm0 = dm_trans_dm0 + P @ r_te @ q_n @ dm_dm0[iso].clone()

            # Calculate Jacobian
            J_n = torch.hstack([dm_trans_dt1, dm_trans_dt2, dm_trans_dm0])

            # Update FIM
            fim = fim + torch.t(J_n) @ J_n

            # Calculate new magnetization and derivatives
            r_tr = r_iso(t1, t2, tr[n])
            b_tr = b_iso(t1, tr[n])

            for iso in range(n_iso):

                g_temp = g(beta[iso])

                dm_dt1[iso] = g_temp @ dr_dt1_iso(t1, tr[n]) @ q_n @ m[iso].clone() + g_temp @ r_tr @ q_n @ dm_dt1[iso].clone() + m0/n_iso*db_dt1_iso(t1, tr[n])

                dm_dt2[iso] = g_temp @ dr_dt2_iso(t2, tr[n]) @ q_n @ m[iso].clone() + g_temp @ r_tr @ q_n @ dm_dt2[iso].clone()

                dm_dm0[iso] = g_temp @ r_tr @ q_n @ dm_dm0[iso].clone() + 1/n_iso*b_tr

                m[iso] = g_temp @ r_tr @ q_n @ m[iso].clone() + m0/n_iso*b_tr

    # Invert FIM
    v = torch.linalg.inv(fim)

    return v


def calculate_crlb_mc_iso(t1: List[float], t2: List[float], m0: float, ratio: float, beats: int, shots: int, fa: Union[np.ndarray, torch.tensor], tr: Union[np.ndarray, torch.tensor], ph: Union[np.ndarray, torch.tensor], prep: List[int], ti: List[int], t2te: List[int], te: float, inv_eff: float=1., delta_B1: float=1., n_iso: int=200, deph: int=2):
    """
    Function to calculate the CRLB of a two component voxel for a given MRF sequence using Bloch/isochromat summation.

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
        n_iso (int): number of isochromats
        deph (int): dephasing across voxel in multiples of pi

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

    # Calculate spin dephasing angle for every isochromat
    beta = torch.linspace(0, deph*np.pi, n_iso)

    # Calculate relaxation matrices and inversion matrix
    r_te_1 = r_iso(t1[0], t2[0], te)
    r_te_2 = r_iso(t1[1], t2[1], te)
    inv = inversion_iso(inv_eff)

    # Initial magnetizations and derivatives
    m_1 = torch.zeros(n_iso, 3, 1, dtype=torch.double)
    m_1[:, 2, :] = ratio*m0/n_iso

    m_2 = torch.zeros(n_iso, 3, 1, dtype=torch.double)
    m_2[:, 2, :] = (1-ratio)*m0/n_iso

    dm_1_dratio = torch.zeros(n_iso, 3, 1, dtype=torch.double)
    dm_1_dratio[:, 2, :] = m0/n_iso

    dm_2_dratio = torch.zeros(n_iso, 3, 1, dtype=torch.double)
    dm_2_dratio[:, 2, :] = -m0/n_iso

    # Initialize Fisher Information Matrix
    fim = torch.zeros(1, dtype=torch.double)
    
    # Iterate over magnetization prepared blocks
    for ii in range(beats):

        # Inversion preparation
        if prep[ii] == 1:

            r_ti_1 = r_iso(t1[0], t2[0], ti[ii])
            r_ti_2 = r_iso(t1[1], t2[1], ti[ii])
            b_ti_1 = b_iso(t1[0], ti[ii])
            b_ti_2 = b_iso(t1[1], ti[ii])
            
            for iso in range(n_iso):

                dm_1_dratio[iso] = r_ti_1 @ inv @ dm_1_dratio[iso].clone() + m0/n_iso*b_ti_1

                dm_2_dratio[iso] = r_ti_2 @ inv @ dm_2_dratio[iso].clone() - m0/n_iso*b_ti_2

                m_1[iso] = r_ti_1 @ inv @ m_1[iso].clone() + ratio*m0/n_iso*b_ti_1
                m_2[iso] = r_ti_2 @ inv @ m_2[iso].clone() + (1-ratio)*m0/n_iso*b_ti_2

        # T2 preparation
        elif prep[ii] == 2:

            t2prep_ii_1 = t2prep_iso(t2[0], t2te[ii])
            t2prep_ii_2 = t2prep_iso(t2[1], t2te[ii])

            for iso in range(n_iso):

                dm_1_dratio[iso] = t2prep_ii_1 @ dm_1_dratio[iso].clone()
                dm_2_dratio[iso] = t2prep_ii_2 @ dm_2_dratio[iso].clone()
                
                m_1[iso] = t2prep_ii_1 @ m_1[iso].clone()
                m_2[iso] = t2prep_ii_1 @ m_2[iso].clone()

        # Iterate over excitations
        for jj in range(shots):

            # Index of current excitation
            n = ii*shots + jj

            # Excitation matrix
            q_n = q_iso(delta_B1*torch.deg2rad(fa[n]), torch.deg2rad(ph[n]))

            # Derivative of transversal magnetization with respect to tissue ratio
            dm_trans_dratio = torch.zeros(2, 1, dtype=torch.double)

            for iso in range(n_iso):

                dm_trans_dratio = dm_trans_dratio + P @ r_te_1 @ q_n @ dm_1_dratio[iso].clone() + P @ r_te_2 @ q_n @ dm_2_dratio[iso].clone()

            # Update FIM
            fim = fim + torch.t(dm_trans_dratio) @ dm_trans_dratio

            # Calculate new magnetizations and derivatives
            r_tr_1 = r_iso(t1[0], t2[0], tr[n])
            r_tr_2 = r_iso(t1[1], t2[1], tr[n])
            b_tr_1 = b_iso(t1[0], tr[n])
            b_tr_2 = b_iso(t1[1], tr[n])

            for iso in range(n_iso):

                g_temp = g(beta[iso])

                dm_1_dratio[iso] = g_temp @ r_tr_1 @ q_n @ dm_1_dratio[iso].clone() + m0/n_iso*b_tr_1
                dm_2_dratio[iso] = g_temp @ r_tr_2 @ q_n @ dm_2_dratio[iso].clone() - m0/n_iso*b_tr_2

                m_1[iso] = g_temp @ r_tr_1 @ q_n @ m_1[iso].clone() + ratio*m0/n_iso*b_tr_1
                m_2[iso] = g_temp @ r_tr_2 @ q_n @ m_2[iso].clone() + (1-ratio)*m0/n_iso*b_tr_2

    return 1/fim
                       

def calculate_orth_iso(t1: List[float], t2: List[float], beats: int, shots: int, fa: Union[np.ndarray, torch.tensor], tr: Union[np.ndarray, torch.tensor], ph: Union[np.ndarray, torch.tensor], prep: List[int], ti: List[int], t2te: List[int], te: float, inv_eff: float=1., delta_B1: float=1., n_iso: int=200, deph: int=2):
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
        n_iso (int): number of isochromats
        deph (int): dephasing across voxel in multiples of pi

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
        s[ii] = calculate_signal_iso(t1[ii], t2[ii], 1, beats, shots, fa, tr, ph, prep, ti, t2te, te, inv_eff, delta_B1, n_iso, deph)

    # Normalize signals
    s = torch.nn.functional.normalize(s, dim=1)

    # Calculate orthogonality matrix
    d = torch.norm(torch.eye(len(t1)) - s @ torch.t(torch.conj(s)))

    return d