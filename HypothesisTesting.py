from typing import List
import numpy as np
from scipy.optimize import approx_fprime
from typing import List, Callable
from scipy.optimize import minimize
from Inference import *
import pandas as pd
from statsmodels.stats.multitest import multipletests
import os

def get_all_grad_logZs(n: List[int], θs: List[List[float]]) -> List[List[float]]:
    '''Compute gradient of log partition function for multiple parameter sets.
    
    Args:
        n: List of region sizes
        θs: List of parameter vectors (α1,...,αk,β)
        
    Returns:
        List of gradient vectors for each parameter set
    '''
    return [get_grad_logZ(n, θ) for θ in θs]

def comp_unmat_stat_mml(n: List[int], grad_logZ1s: List[List[float]], grad_logZ2s: List[List[float]]) -> float:
    '''Compute unpaired group comparison statistic for mean methylation level (MML).
    
    Args:
        n: List of region sizes
        grad_logZ1s: Gradients for group 1
        grad_logZ2s: Gradients for group 2
        
    Returns:
        Difference in mean MML between groups (rounded to 6 decimal places)
    '''
    mml1 = sum(comp_mml(n, grad_logZ) for grad_logZ in grad_logZ1s) / len(grad_logZ1s)
    mml2 = sum(comp_mml(n, grad_logZ) for grad_logZ in grad_logZ2s) / len(grad_logZ2s)
    return round(mml1 - mml2, 6)

def comp_unmat_perm_stat_mml(n: List[int], grad_logZ1s: List[List[float]], grad_logZ2s: List[List[float]], perm_ids: List[int]) -> float:
    '''Compute permuted unpaired MML statistic.
    
    Args:
        n: List of region sizes
        grad_logZ1s: Gradients for group 1
        grad_logZ2s: Gradients for group 2
        perm_ids: Indices for permutation
        
    Returns:
        Permuted MML difference statistic
    '''
    grad_logZ2sp = np.concatenate((grad_logZ1s, grad_logZ2s), axis=0)
    grad_logZ1sp = grad_logZ2sp[perm_ids]
    grad_logZ2sp = np.delete(grad_logZ2sp, perm_ids, axis=0)
    return comp_unmat_stat_mml(n, grad_logZ1sp, grad_logZ2sp)

def comp_unmat_stat_nme(n: List[int], θ1s: List[List[float]], θ2s: List[List[float]], grad_logZ1s: List[List[float]], grad_logZ2s: List[List[float]]) -> float:
    '''Compute unpaired group comparison statistic for normalized methylation entropy (NME).
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        grad_logZ1s: Gradients for group 1
        grad_logZ2s: Gradients for group 2
        
    Returns:
        Difference in mean NME between groups (rounded to 6 decimal places)
    '''
    nme1 = 0.0
    for s in range(len(θ1s)):
        nme1 += comp_nme(n, θ1s[s], grad_logZ1s[s])
    nme1 /= len(θ1s)

    nme2 = 0.0
    for s in range(len(θ2s)):
        nme2 += comp_nme(n, θ2s[s], grad_logZ2s[s])
    nme2 /= len(θ2s)

    return round(nme1 - nme2, 6)

def comp_unmat_perm_stat_nme(n: List[int], θ1s: List[List[float]], θ2s: List[List[float]], grad_logZ1s: List[List[float]], grad_logZ2s: List[List[float]], perm_ids: List[int]) -> float:
    '''Compute permuted unpaired NME statistic.
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        grad_logZ1s: Gradients for group 1
        grad_logZ2s: Gradients for group 2
        perm_ids: Indices for permutation
        
    Returns:
        Permuted NME difference statistic
    '''
    θ2sp = np.concatenate((θ1s, θ2s))
    θ1sp = θ2sp[perm_ids]
    θ2sp = np.delete(θ2sp, perm_ids, axis=0)

    grad_logZ2sp = np.concatenate((grad_logZ1s, grad_logZ2s))
    grad_logZ1sp = grad_logZ2sp[perm_ids]
    grad_logZ2sp = np.delete(grad_logZ2sp, perm_ids, axis=0)

    return comp_unmat_stat_nme(n, θ1sp, θ2sp, grad_logZ1sp, grad_logZ2sp)

def comp_unmat_stat_pdm(n: List[int], θ1s: List[List[float]], θ2s: List[List[float]], grad_logZ1s: List[List[float]], grad_logZ2s: List[List[float]]) -> float:
    '''Compute unpaired group comparison statistic for pairwise divergence measure (PDM).
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        grad_logZ1s: Gradients for group 1
        grad_logZ2s: Gradients for group 2
        
    Returns:
        Average pairwise divergence between groups (rounded to 6 decimal places)
    '''
    cmds = []
    for s1 in range(len(θ1s)):
        for s2 in range(len(θ2s)):
            cmds.append(comp_cmd(n, θ1s[s1], θ2s[s2], grad_logZ1s[s1], grad_logZ2s[s2]))
    return round(sum(cmds) / (len(θ1s) * len(θ2s)), 6)

def comp_unmat_perm_stat_pdm(n: List[int], θ1s: List[List[float]], θ2s: List[List[float]], grad_logZ1s: List[List[float]], grad_logZ2s: List[List[float]], perm_ids: List[int]) -> float:
    '''Compute permuted unpaired PDM statistic.
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        grad_logZ1s: Gradients for group 1
        grad_logZ2s: Gradients for group 2
        perm_ids: Indices for permutation
        
    Returns:
        Permuted pairwise divergence statistic
    '''
    θ2sp = np.concatenate((θ1s, θ2s))
    θ1sp = θ2sp[perm_ids]
    θ2sp = np.delete(θ2sp, perm_ids, axis=0)

    grad_logZ2sp = np.concatenate((grad_logZ1s, grad_logZ2s))
    grad_logZ1sp = grad_logZ2sp[perm_ids]
    grad_logZ2sp = np.delete(grad_logZ2sp, perm_ids, axis=0)

    return comp_unmat_stat_pdm(n, θ1sp, θ2sp, grad_logZ1sp, grad_logZ2sp)

def comp_unmat_stat_pdr(n: List[int], θ1s: List[List[float]], θ2s: List[List[float]]) -> float:
    '''Compute unpaired group comparison statistic for partially disordered ratio (PDR).
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        
    Returns:
        Difference in mean PDR between groups (rounded to 6 decimal places)
    '''
    pdr1 = sum(comp_pdr(n, θ) for θ in θ1s) / len(θ1s)
    pdr2 = sum(comp_pdr(n, θ) for θ in θ2s) / len(θ2s)
    return round(pdr1 - pdr2, 6)

def comp_unmat_perm_stat_pdr(n: List[int], θ1s: List[List[float]], θ2s: List[List[float]], perm_ids: List[int]) -> float:
    '''Compute permuted unpaired PDR statistic.
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        perm_ids: Indices for permutation
        
    Returns:
        Permuted PDR difference statistic
    '''
    θ2sp = np.concatenate((θ1s, θ2s), axis=0)
    θ1sp = θ2sp[perm_ids]
    θ2sp = np.delete(θ2sp, perm_ids, axis=0)
    return comp_unmat_stat_pdr(n, θ1sp, θ2sp)

def comp_unmat_stat_chalm(n: List[int], θ1s: List[List[float]], θ2s: List[List[float]]) -> float:
    '''Compute unpaired group comparison statistic for completely homogeneous allelic likelihood (CHALM).
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        
    Returns:
        Difference in mean CHALM between groups (rounded to 6 decimal places)
    '''
    chalm1 = sum(comp_chalm(n, θ) for θ in θ1s) / len(θ1s)
    chalm2 = sum(comp_chalm(n, θ) for θ in θ2s) / len(θ2s)
    return round(chalm1 - chalm2, 6)

def comp_unmat_perm_stat_chalm(n: List[int], θ1s: List[List[float]], θ2s: List[List[float]], perm_ids: List[int]) -> float:
    '''Compute permuted unpaired CHALM statistic.
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        perm_ids: Indices for permutation
        
    Returns:
        Permuted CHALM difference statistic
    '''
    θ2sp = np.concatenate((θ1s, θ2s), axis=0)
    θ1sp = θ2sp[perm_ids]
    θ2sp = np.delete(θ2sp, perm_ids, axis=0)
    return comp_unmat_stat_chalm(n, θ1sp, θ2sp)

def comp_unmat_stat_mcr(n: List[int], θ1s: List[List[float]], θ2s: List[List[float]], grad_logZ1s: List[List[float]], grad_logZ2s: List[List[float]]) -> float:
    '''Compute unpaired group comparison statistic for methylation correlation ratio (MCR).
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        grad_logZ1s: Gradients for group 1
        grad_logZ2s: Gradients for group 2
        
    Returns:
        Difference in mean MCR between groups
    '''
    tchalm = comp_unmat_stat_chalm(n, θ1s, θ2s)
    tmml = comp_unmat_stat_mml(n, grad_logZ1s, grad_logZ2s)
    return tchalm - tmml

def comp_unmat_perm_stat_mcr(n: List[int], θ1s: List[List[float]], θ2s: List[List[float]], grad_logZ1s: List[List[float]], grad_logZ2s: List[List[float]], perm_ids: List[int]) -> float:
    '''Compute permuted unpaired MCR statistic.
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        grad_logZ1s: Gradients for group 1
        grad_logZ2s: Gradients for group 2
        perm_ids: Indices for permutation
        
    Returns:
        Permuted MCR difference statistic
    '''
    ptchalm = comp_unmat_perm_stat_chalm(n, θ1s, θ2s, perm_ids)
    ptmml = comp_unmat_perm_stat_mml(n, grad_logZ1s, grad_logZ2s, perm_ids)
    return ptchalm - ptmml

def unmat_tests(n, θ1s, θ2s, Lmax=1000):
    '''Perform unpaired group comparison tests with permutation-based p-values.
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        Lmax: Maximum number of permutations (default: 1000)
        
    Returns:
        Tuple of tuples containing (statistic, p-value) pairs for:
        - MML (mean methylation level)
        - NME (normalized methylation entropy)
        - PDM (pairwise divergence measure)
        - PDR (partially disordered ratio)
        - CHALM (completely homogeneous allelic likelihood)
        - MCR (methylation correlation ratio)
    '''
    # Compute gradients
    grad_logZ1s = get_all_grad_logZs(n, θ1s)
    grad_logZ2s = get_all_grad_logZs(n, θ2s)

    # Compute observed statistics
    tmml_obs = comp_unmat_stat_mml(n, grad_logZ1s, grad_logZ2s)
    tnme_obs = comp_unmat_stat_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
    tpdm_obs = comp_unmat_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
    tpdr_obs = comp_unmat_stat_pdr(n, θ1s, θ2s)
    tchalm_obs = comp_unmat_stat_chalm(n, θ1s, θ2s)
    tmcr_obs = comp_unmat_stat_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)

    # Compute number of possible permutations
    L = comb(len(θ1s) + len(θ2s), len(θ1s))
    exact = L < Lmax

    # Create iterable object with all combinations
    comb_iter = combinations(range(len(θ1s) + len(θ2s)), len(θ1s))

    # Get group label combinations to use
    if exact:
        comb_iter_used = list(comb_iter)
    else:
        ind_subset = np.random.choice(L, Lmax, replace=False)
        comb_iter_used = [comb for idx, comb in enumerate(comb_iter) if idx in ind_subset]

    # Compute permuted statistics
    tmml_perms = [comp_unmat_perm_stat_mml(n, grad_logZ1s, grad_logZ2s, list(x)) for x in comb_iter_used]
    tnme_perms = [comp_unmat_perm_stat_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, list(x)) for x in comb_iter_used]
    tpdm_perms = [comp_unmat_perm_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, list(x)) for x in comb_iter_used]
    tpdr_perms = [comp_unmat_perm_stat_pdr(n, θ1s, θ2s, list(x)) for x in comb_iter_used]
    tchalm_perms = [comp_unmat_perm_stat_chalm(n, θ1s, θ2s, list(x)) for x in comb_iter_used]
    tmcr_perms = [comp_unmat_perm_stat_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, list(x)) for x in comb_iter_used]

    # Compute p-values
    if exact:
        tmml_pval = np.sum(np.abs(tmml_perms) >= abs(tmml_obs)) / len(tmml_perms)
        tnme_pval = np.sum(np.abs(tnme_perms) >= abs(tnme_obs)) / len(tnme_perms)
        tpdm_pval = np.sum(np.array(tpdm_perms) >= tpdm_obs) / len(tpdm_perms)
        tpdr_pval = np.sum(np.abs(tpdr_perms) >= abs(tpdr_obs)) / len(tpdr_perms)
        tchalm_pval = np.sum(np.abs(tchalm_perms) >= abs(tchalm_obs)) / len(tchalm_perms)
        tmcr_pval = np.sum(np.abs(tmcr_perms) >= abs(tmcr_obs)) / len(tmcr_perms)
    else:
        tmml_pval = (1.0 + np.sum(np.abs(tmml_perms) >= abs(tmml_obs))) / (1.0 + len(tmml_perms))
        tnme_pval = (1.0 + np.sum(np.abs(tnme_perms) >= abs(tnme_obs))) / (1.0 + len(tnme_perms))
        tpdm_pval = (1.0 + np.sum(np.array(tpdm_perms) >= tpdm_obs)) / (1.0 + len(tpdm_perms))
        tpdr_pval = (1.0 + np.sum(np.abs(tpdr_perms) >= abs(tpdr_obs))) / (1.0 + len(tpdr_perms))
        tchalm_pval = (1.0 + np.sum(np.abs(tchalm_perms) >= abs(tchalm_obs))) / (1.0 + len(tchalm_perms))
        tmcr_pval = (1.0 + np.sum(np.abs(tmcr_perms) >= abs(tmcr_obs))) / (1.0 + len(tmcr_perms))

    return ((tmml_obs, tmml_pval.item()), 
            (tnme_obs, tnme_pval.item()), 
            (tpdm_obs, tpdm_pval.item()), 
            (tpdr_obs, tpdr_pval.item()),
            (tchalm_obs, tchalm_pval.item()), 
            (tmcr_obs, tmcr_pval.item()))

# Paired comparison functions...

def comp_mat_diff_mml(n, grad_logZ1s, grad_logZ2s):
    '''Compute paired differences in mean methylation level (MML).
    
    Args:
        n: List of region sizes
        grad_logZ1s: Gradients for group 1
        grad_logZ2s: Gradients for group 2
        
    Returns:
        List of paired MML differences
    '''
    diffs = []
    for grad1, grad2 in zip(grad_logZ1s, grad_logZ2s):
        mml_diff = comp_mml(n, grad1) - comp_mml(n, grad2)
        diffs.append(mml_diff)
    return diffs

def comp_mat_diff_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s):
    '''Compute paired differences in normalized methylation entropy (NME).
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        grad_logZ1s: Gradients for group 1
        grad_logZ2s: Gradients for group 2
        
    Returns:
        List of paired NME differences (rounded to 6 decimal places)
    '''
    diffs = []
    for θ1, θ2, grad1, grad2 in zip(θ1s, θ2s, grad_logZ1s, grad_logZ2s):
        nme_diff = comp_nme(n, θ1, grad1) - comp_nme(n, θ2, grad2)
        diffs.append(round(nme_diff, 6))
    return diffs

def comp_mat_diff_pdr(n, θ1s, θ2s):
    '''Compute paired differences in partially disordered ratio (PDR).
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        
    Returns:
        List of paired PDR differences (rounded to 6 decimal places)
    '''
    diffs = []
    for θ1, θ2 in zip(θ1s, θ2s):
        pdr_diff = comp_pdr(n, θ1) - comp_pdr(n, θ2)
        diffs.append(round(pdr_diff, 6))
    return diffs

def comp_mat_diff_chalm(n, θ1s, θ2s):
    '''Compute paired differences in completely homogeneous allelic likelihood (CHALM).
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        
    Returns:
        List of paired CHALM differences (rounded to 6 decimal places)
    '''
    diffs = []
    for θ1, θ2 in zip(θ1s, θ2s):
        chalm_diff = comp_chalm(n, θ1) - comp_chalm(n, θ2)
        diffs.append(round(chalm_diff, 6))
    return diffs

def comp_mat_diff_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s):
    '''Compute paired differences in methylation correlation ratio (MCR).
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        grad_logZ1s: Gradients for group 1
        grad_logZ2s: Gradients for group 2
        
    Returns:
        List of paired MCR differences (rounded to 6 decimal places)
    '''
    diffs = []
    for θ1, θ2, grad1, grad2 in zip(θ1s, θ2s, grad_logZ1s, grad_logZ2s):
        mcr_diff = comp_mcr(n, θ1, grad1) - comp_mcr(n, θ2, grad2)
        diffs.append(round(mcr_diff, 6))
    return diffs

def comp_mat_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s):
    '''Compute paired pairwise divergence measure (PDM).
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        grad_logZ1s: Gradients for group 1
        grad_logZ2s: Gradients for group 2
        
    Returns:
        Average pairwise divergence between paired samples
    '''
    cmds = 0.0
    for θ1, θ2, grad1, grad2 in zip(θ1s, θ2s, grad_logZ1s, grad_logZ2s):
        cmds += comp_cmd(n, θ1, θ2, grad1, grad2)
    return cmds / len(θ1s)

def comp_mat_j_stat(diffs, j):
    '''Compute permuted statistic for paired comparison.
    
    Args:
        diffs: List of paired differences
        j: Permutation index
        
    Returns:
        Permuted statistic value
    '''
    diffs_perm = 0.0
    for i in range(len(diffs)):
        diffs_perm += diffs[i] if (j & 1) else -diffs[i]
        j >>= 1
    return diffs_perm / len(diffs)

def comp_mat_perm_stats(diffs, js):
    '''Compute multiple permuted statistics for paired comparison.
    
    Args:
        diffs: List of paired differences
        js: List of permutation indices
        
    Returns:
        List of permuted statistics
    '''
    return [comp_mat_j_stat(diffs, j) for j in js]

def mat_tests(n, θ1s, θ2s, Lmax=1000):
    '''Perform paired group comparison tests with permutation-based p-values.
    
    Args:
        n: List of region sizes
        θ1s: Parameter vectors for group 1
        θ2s: Parameter vectors for group 2
        Lmax: Maximum number of permutations (default: 1000)
        
    Returns:
        Tuple of tuples containing (statistic, p-value) pairs for:
        - MML (mean methylation level)
        - NME (normalized methylation entropy)
        - PDM (pairwise divergence measure)
        - PDR (partially disordered ratio)
        - CHALM (completely homogeneous allelic likelihood)
        - MCR (methylation correlation ratio)
    '''
    grad_logZ1s = get_all_grad_logZs(n, θ1s)
    grad_logZ2s = get_all_grad_logZs(n, θ2s)

    # Compute paired differences
    mml_diffs = comp_mat_diff_mml(n, grad_logZ1s, grad_logZ2s)
    nme_diffs = comp_mat_diff_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
    pdr_diffs = comp_mat_diff_pdr(n, θ1s, θ2s)
    chalm_diffs = comp_mat_diff_chalm(n, θ1s, θ2s)
    mcr_diffs = comp_mat_diff_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
    tpdm_obs = comp_mat_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
    
    # Determine permutation strategy
    exact = 2**len(θ1s) < Lmax
    js = list(range(2**len(θ1s))) if exact else [np.random.randint(0, 2**len(θ1s) - 1) for _ in range(Lmax)]

    # Compute permuted statistics
    tmml_perms = comp_mat_perm_stats(mml_diffs, js)
    tnme_perms = comp_mat_perm_stats(nme_diffs, js)
    tpdr_perms = comp_mat_perm_stats(pdr_diffs, js)
    tchalm_perms = comp_mat_perm_stats(chalm_diffs, js)
    tmcr_perms = comp_mat_perm_stats(mcr_diffs, js)

    # Compute observed statistics
    tmml_obs = sum(mml_diffs) / len(θ1s)
    tnme_obs = sum(nme_diffs) / len(θ2s)
    tpdr_obs = sum(pdr_diffs) / len(θ1s)
    tchalm_obs = sum(chalm_diffs) / len(θ1s)
    tmcr_obs = sum(mcr_diffs) / len(θ1s)

    # Compute p-values
    if exact:
        tmml_pval = sum(np.abs(tmml_perms) >= np.abs(tmml_obs)) / len(tmml_perms)
        tnme_pval = sum(np.abs(tnme_perms) >= np.abs(tnme_obs)) / len(tnme_perms)
        tpdr_pval = sum(np.abs(tpdr_perms) >= np.abs(tpdr_obs)) / len(tpdr_perms)
        tchalm_pval = sum(np.abs(tchalm_perms) >= np.abs(tchalm_obs)) / len(tchalm_perms)
        tmcr_pval = sum(np.abs(tmcr_perms) >= np.abs(tmcr_obs)) / len(tmcr_perms)
    else:
        tmml_pval = (1.0 + sum(np.abs(tmml_perms) >= np.abs(tmml_obs))) / (1.0 + len(tmml_perms))
        tnme_pval = (1.0 + sum(np.abs(tnme_perms) >= np.abs(tnme_obs))) / (1.0 + len(tnme_perms))
        tpdr_pval = (1.0 + sum(np.abs(tpdr_perms) >= np.abs(tpdr_obs))) / (1.0 + len(tpdr_perms))
        tchalm_pval = (1.0 + sum(np.abs(tchalm_perms) >= np.abs(tchalm_obs))) / (1.0 + len(tchalm_perms))
        tmcr_pval = (1.0 + sum(np.abs(tmcr_perms) >= np.abs(tmcr_obs))) / (1.0 + len(tmcr_perms))

    return ((tmml_obs, tmml_pval.item()),
            (tnme_obs, tnme_pval.item()),
            (tpdm_obs, np.nan),  # PDM p-value not computed in this implementation
            (tpdr_obs, tpdr_pval.item()),
            (tchalm_obs, tchalm_pval.item()),
            (tmcr_obs, tmcr_pval.item()))