import numpy as np
from Inference import *
from HypothesisTesting import *
from scipy.stats import rv_discrete
from scipy.special import expit
from numpy.random import choice
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

def gen_x_mc(n, θ):
    '''Generate methylation states using Markov Chain Monte Carlo.
    
    Args:
        n: List of region sizes
        θ: Parameter vector (α1,...,αk,β)
        
    Returns:
        List of generated methylation states (-1, 0, 1)
    '''
    alpha = np.array(θ[:-1])
    beta = θ[-1]

    # Create subregion IDs
    subid = np.concatenate([np.full(ni, i + 1, dtype=int) for i, ni in enumerate(n)])

    x = []
    # Generate first site observation
    initial_prob = float(comp_lkhd(np.insert(np.zeros(sum(n) - 1, dtype=int), 0, 1), n, alpha, beta))
    x.append(choice([-1, 1], p=[1 - initial_prob, initial_prob]))

    if sum(n) <= 1:
        return x

    # Generate remaining sites
    for i in range(1, sum(n)):
        expaux = np.exp(alpha[subid[i] - 1] + beta * x[i - 1])

        if i < sum(n) - 1:
            # Calculate parameters for conditional probability
            ap1p = 2.0 * beta + alpha[subid[i] - 1]  # x2=1
            ap1m = -2.0 * beta + alpha[subid[i] - 1]  # x2=-1
            
            unique_subid = np.unique(subid[i + 1:])
            n_miss = [np.sum(subid[i + 1:] == sr) for sr in unique_subid]
            
            # Calculate marginal probabilities
            gp = float(comp_g(n_miss, alpha[unique_subid - 1], beta, ap1p, alpha[-1]))  # x2=1
            gm = float(comp_g(n_miss, alpha[unique_subid - 1], beta, ap1m, alpha[-1]))    # x2=-1
            p = gp * expaux / (gp * expaux + gm / expaux)
        else:
            p = expaux / (expaux + 1.0 / expaux)

        x.append(choice([-1, 1], p=[1 - p, p]))

    return x

def unmatched_sim(m, n, θ1, θ2):
    '''Perform unmatched group comparison simulation.
    
    Args:
        m: Number of samples per group
        n: List of CpG site counts
        θ1: Parameters for group 1
        θ2: Parameters for group 2
        
    Returns:
        Tuple of test statistic results for:
        - MML (mean methylation level)
        - NME (normalized methylation entropy) 
        - PDM (pairwise divergence measure)
        - PDR (partially disordered ratio)
        - CHALM (completely homogeneous allelic likelihood)
        - MCR (methylation correlation ratio)
    '''
    tmml = []
    tnme = []
    tpdm = []
    tpdr = []
    tchalm = []
    tmcr = []
    c = 1
    g1 = []
    g2 = []
    
    while len(tmml) < 2:
        # Generate samples for both groups
        g1_samples = []
        for _ in range(m):
            g1_samples.append([gen_x_mc(n, θ1) for _ in range(np.random.randint(10, 31))])

        g2_samples = []
        for _ in range(m):
            g2_samples.append([gen_x_mc(n, θ2) for _ in range(np.random.randint(10, 31))])

        # Estimate parameters for each sample
        g1_θs = []
        for xobs in g1_samples:
            l = est_theta_sa(n, xobs)
            g1_θs.append(l)
            g1.append(l)

        g2_θs = []
        for xobs in g2_samples:
            l = est_theta_sa(n, xobs)
            g2_θs.append(l)
            g2.append(l)

        print("simulation:", c)
        
        # Perform hypothesis tests
        test_results = unmat_tests(n, g1_θs, g2_θs)
        tmml.append(test_results[0])
        tnme.append(test_results[1])
        tpdm.append(test_results[2])
        tpdr.append(test_results[3])
        tchalm.append(test_results[4])
        tmcr.append(test_results[5])
        c += 1

    return tmml, tnme, tpdm, tpdr, tchalm, tmcr

def matched_sim(m, n, θ1, θ2):
    '''Perform matched group comparison simulation.
    
    Args:
        m: Number of sample pairs
        n: List of CpG site counts
        θ1: Parameters for group 1
        θ2: Parameters for group 2
        
    Returns:
        Tuple of test statistic results (same as unmatched_sim)
    '''
    tmml = []
    tnme = []
    tpdm = []
    tpdr = []
    tchalm = []
    tmcr = []
    c = 1

    while len(tmml) < 2:
        # Generate matched samples
        g1_samples = []
        for _ in range(m):
            g1_samples.append([gen_x_mc(n, θ1) for _ in range(np.random.randint(10, 31))])

        g2_samples = []
        for _ in range(m):
            g2_samples.append([gen_x_mc(n, θ2) for _ in range(np.random.randint(10, 31))])

        # Estimate parameters
        g1_θs = []
        for xobs in g1_samples:
            g1_θs.append(est_theta_sa(n, xobs))

        g2_θs = []
        for xobs in g2_samples:
            g2_θs.append(est_theta_sa(n, xobs))

        print("simulation:", c)
        
        # Perform hypothesis tests
        test_results = mat_tests(n, g1_θs, g2_θs)
        tmml.append(test_results[0])
        tnme.append(test_results[1])
        tpdm.append(test_results[2])
        tpdr.append(test_results[3])
        tchalm.append(test_results[4])
        tmcr.append(test_results[5])
        c += 1

    return tmml, tnme, tpdm, tpdr, tchalm, tmcr

def unmat_pvalue_ecdf(m, n, θ1, θ2):
    '''Compute empirical CDF of p-values for unmatched tests.
    
    Args:
        m: Number of samples per group
        n: List of CpG site counts
        θ1: Parameters for group 1
        θ2: Parameters for group 2
        
    Returns:
        Tuple containing:
        - ECDF functions for each test statistic
        - Raw p-values for each test
    '''
    tmml_results, tnme_results, tpdm_results, tpdr_results, tchalm_results, tmcr_results = unmatched_sim(m, n, θ1, θ2)
    
    # Extract p-values
    tmml_pvals = [result[1] for result in tmml_results]
    tnme_pvals = [result[1] for result in tnme_results]
    tpdm_pvals = [result[1] for result in tpdm_results]
    tpdr_pvals = [result[1] for result in tpdr_results]
    tchalm_pvals = [result[1] for result in tchalm_results]
    tmcr_pvals = [result[1] for result in tmcr_results]
    
    # Sort p-values for ECDF
    sorted_tmml = np.sort(tmml_pvals)
    sorted_tnme = np.sort(tnme_pvals)
    sorted_tpdm = np.sort(tpdm_pvals)
    sorted_tpdr = np.sort(tpdr_pvals)
    sorted_tchalm = np.sort(tchalm_pvals)
    sorted_tmcr = np.sort(tmcr_pvals)
    
    # Define ECDF functions
    def ecdf_tmml(x):
        return np.searchsorted(sorted_tmml, x, side='right') / len(sorted_tmml)
    
    def ecdf_tnme(x):
        return np.searchsorted(sorted_tnme, x, side='right') / len(sorted_tnme)
    
    def ecdf_tpdm(x):
        return np.searchsorted(sorted_tpdm, x, side='right') / len(sorted_tpdm)

    def ecdf_tpdr(x):
        return np.searchsorted(sorted_tpdr, x, side='right') / len(sorted_tpdr)

    def ecdf_tchalm(x):
        return np.searchsorted(sorted_tchalm, x, side='right') / len(sorted_tchalm)

    def ecdf_tmcr(x):
        return np.searchsorted(sorted_tmcr, x, side='right') / len(sorted_tmcr)
    
    # Return ECDFs and raw p-values
    pvals = (tmml_pvals, tnme_pvals, tpdm_pvals, tpdr_pvals, tchalm_pvals, tmcr_pvals)
    
    return ecdf_tmml, ecdf_tnme, ecdf_tpdm, ecdf_tpdr, ecdf_tchalm, ecdf_tmcr, pvals

def mat_pvalue_ecdf(m, n, θ1, θ2):
    '''Compute empirical CDF of p-values for matched tests.
    
    Args:
        m: Number of sample pairs
        n: List of CpG site counts
        θ1: Parameters for group 1
        θ2: Parameters for group 2
        
    Returns:
        Tuple containing:
        - ECDF functions for each test statistic
        - Raw p-values for each test
    '''
    tmml_results, tnme_results, _, tpdr_results, tchalm_results, tmcr_results = matched_sim(m, n, θ1, θ2)
    
    # Extract p-values
    tmml_pvals = [result[1] for result in tmml_results]
    tnme_pvals = [result[1] for result in tnme_results]
    tpdr_pvals = [result[1] for result in tpdr_results]
    tchalm_pvals = [result[1] for result in tchalm_results]
    tmcr_pvals = [result[1] for result in tmcr_results]
    
    # Sort p-values for ECDF
    sorted_tmml = np.sort(tmml_pvals)
    sorted_tnme = np.sort(tnme_pvals)
    sorted_tpdr = np.sort(tpdr_pvals)
    sorted_tchalm = np.sort(tchalm_pvals)
    sorted_tmcr = np.sort(tmcr_pvals)
    
    # Define ECDF functions
    def ecdf_tmml(x):
        return np.searchsorted(sorted_tmml, x, side='right') / len(sorted_tmml)
    
    def ecdf_tnme(x):
        return np.searchsorted(sorted_tnme, x, side='right') / len(sorted_tnme)

    def ecdf_tpdr(x):
        return np.searchsorted(sorted_tpdr, x, side='right') / len(sorted_tpdr)

    def ecdf_tchalm(x):
        return np.searchsorted(sorted_tchalm, x, side='right') / len(sorted_tchalm)

    def ecdf_tmcr(x):
        return np.searchsorted(sorted_tmcr, x, side='right') / len(sorted_tmcr)
    
    # Return ECDFs and raw p-values
    pvals = (tmml_pvals, tnme_pvals, tpdr_pvals, tchalm_pvals, tmcr_pvals)
    
    return ecdf_tmml, ecdf_tnme, ecdf_tpdr, ecdf_tchalm, ecdf_tmcr, pvals

def comp_stat(n, xobs):
    '''Compare model statistics with empirical statistics.
    
    Args:
        n: List of region sizes
        xobs: Observation matrix
        
    Returns:
        Tuple of tuples containing (model_value, empirical_value) pairs for:
        - MML
        - NME
        - PDR
        - CHALM
        - MCR
    '''
    # Compute model statistics
    θ = est_theta_sa(n, xobs)
    grad_logZ = get_grad_logZ(n, θ)
    model_mml = comp_mml(n, grad_logZ)
    model_nme = comp_nme(n, θ, grad_logZ)
    model_pdr = comp_pdr(n, θ)
    model_chalm = comp_chalm(n, θ)
    model_mcr = model_chalm - model_mml

    # Compute empirical statistics
    xobs_array = np.array(xobs)
    total_reads = len(xobs_array)
    total_CpGs = xobs_array.size
    methy_CpGs = np.sum(xobs_array == 1)
    concordant_reads = np.all(xobs_array == xobs_array[:, [0]], axis=1)
    cpdr = np.sum(concordant_reads) / total_reads
    methy_read = np.any(xobs_array == 1, axis=1)
    methy_reads = xobs_array[methy_read]
    unmethy_GpGs = np.sum(methy_reads == -1)

    true_mml = np.round(methy_CpGs / total_CpGs, 8).item()

    _, unique_reads_counts = np.unique(xobs_array, axis=0, return_counts=True)
    p = unique_reads_counts / total_reads
    
    with np.errstate(divide='ignore'):  # Ignore log(0) warnings
        log_p = np.log2(p)
    log_p[p == 0] = 0  # Handle zero probabilities
    true_nme = np.round(-1.0/sum(n)*np.sum(p * log_p), 8).item()
    
    true_pdr = 1.0 - np.round(cpdr, 8).item()
    true_chalm = np.round(np.sum(methy_read) / total_reads, 8).item()
    true_mcr = true_chalm - true_mml

    return (model_mml, true_mml), (model_nme, true_nme), (model_pdr, true_pdr), (model_chalm, true_chalm), (model_mcr, true_mcr)

def plot_unmat(m, n, θ1, θ2, alpha=0.05, figsize=(18, 5)):
    '''Plot ECDFs for unmatched test statistics.
    
    Args:
        m: Number of samples per group
        n: List of CpG site counts
        θ1: Parameters for group 1
        θ2: Parameters for group 2
        alpha: Significance level
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    '''
    # Get ECDF functions and p-values
    ecdf_tmml, ecdf_tnme, ecdf_tpdm, ecdf_tpdr, ecdf_tchalm, ecdf_tmcr, pvals = unmat_pvalue_ecdf(m, n, θ1, θ2)
    tmml_pvals, tnme_pvals, tpdm_pvals, tpdr_pvals, tchalm_pvals, tmcr_pvals = pvals

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 6, figure=fig)
    
    # Statistics to plot
    statistics = [
        ("TMML", ecdf_tmml, tmml_pvals),
        ("TNME", ecdf_tnme, tnme_pvals),
        ("TPDM", ecdf_tpdm, tpdm_pvals),
        ("TPDR", ecdf_tpdr, tpdr_pvals),
        ("TCHALM", ecdf_tchalm, tchalm_pvals),
        ("TMCR", ecdf_tmcr, tmcr_pvals)
    ]
    
    # Compute cumulative probabilities at alpha
    pr_values = [ecdf(alpha) for _, ecdf, _ in statistics]
    
    # Create subplots
    axes = []
    for i, (title, ecdf_func, _) in enumerate(statistics):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        
        # Plot ECDF
        x = np.linspace(0, 1, 100)
        y = [ecdf_func(xi) for xi in x]
        ax.plot(x, y, 'b-', linewidth=2)
        
        # Mark significant points
        pr_val = pr_values[i]
        if pr_val >= 0.95:
            ax.scatter(0.5, 0.5, color='red', s=150, marker='$*$', zorder=5)
        
        # Add text annotation
        if pr_val < 0.01:
            text = f'Pr[p≤{alpha}] < 0.01'
        else:
            text = f'Pr[p≤{alpha}] = {pr_val:.2f}'
        
        ax.text(0.05, 0.95, text, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('P_value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        if i == 0:
            ax.set_ylabel('cumulative probability')
    
    # Add overall title
    param_str = f"m={m}, n={n}, θ1={θ1}, θ2={θ2}"
    plt.suptitle(f"ECDF Comparison for Parameters(unmatched): {param_str}", fontsize=14, y=1.05)

    plt.tight_layout()
    return fig

def plot_mat(m, n, θ1, θ2, alpha=0.05, figsize=(15, 4)):
    '''Plot ECDFs for matched test statistics.
    
    Args:
        m: Number of sample pairs
        n: List of CpG site counts
        θ1: Parameters for group 1
        θ2: Parameters for group 2
        alpha: Significance level
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    '''
    # Get ECDF functions and p-values
    ecdf_tmml, ecdf_tnme, ecdf_tpdr, ecdf_tchalm, ecdf_tmcr, pvals = mat_pvalue_ecdf(m, n, θ1, θ2)
    tmml_pvals, tnme_pvals, tpdr_pvals, tchalm_pvals, tmcr_pvals = pvals
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 5, figure=fig)
    
    # Statistics to plot
    statistics = [
        ("TMML", ecdf_tmml, tmml_pvals),
        ("TNME", ecdf_tnme, tnme_pvals),
        ("TPDR", ecdf_tpdr, tpdr_pvals),
        ("TCHALM", ecdf_tchalm, tchalm_pvals),
        ("TMCR", ecdf_tmcr, tmcr_pvals)
    ]
    
    # Compute cumulative probabilities at alpha
    pr_values = [ecdf(alpha) for _, ecdf, _ in statistics]
    
    # Create subplots
    axes = []
    for i, (title, ecdf_func, _) in enumerate(statistics):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        
        # Plot ECDF
        x = np.linspace(0, 1, 100)
        y = [ecdf_func(xi) for xi in x]
        ax.plot(x, y, 'b-', linewidth=2)
        
        # Mark significant points
        pr_val = pr_values[i]
        if pr_val >= 0.95:
            ax.scatter(0.5, 0.5, color='red', s=150, marker='$*$', zorder=5)
        
        # Add text annotation
        if pr_val < 0.01:
            text = f'Pr[p≤{alpha}] < 0.01'
        else:
            text = f'Pr[p≤{alpha}] = {pr_val:.2f}'
        
        ax.text(0.05, 0.95, text, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('P_value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        if i == 0:
            ax.set_ylabel('cumulative probability')
    
    # Add overall title
    param_str = f"m={m}, n={n}, θ1={θ1}, θ2={θ2}"
    plt.suptitle(f"ECDF Comparison for Parameters(matched): {param_str}", fontsize=14, y=1.05)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Command line interface
    func_name = sys.argv[1]
    m = int(sys.argv[2])
    n = [int(sys.argv[3])]  # Convert to list
    θ1 = list(map(float, sys.argv[4].split(',')))
    θ2 = list(map(float, sys.argv[5].split(',')))
    
    # Create output directory
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    param_str = f"m{m}_n{n[0]}_theta1_{θ1[0]}_{θ1[1]}_theta2_{θ2[0]}_{θ2[1]}"
    
    # Call appropriate plotting function
    if func_name == "plot_unmat":
        fig = plot_unmat(m, n, θ1, θ2)
        filename = f"{output_dir}/unmat_{param_str}.png"
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)  
    elif func_name == "plot_mat":
        fig = plot_mat(m, n, θ1, θ2)
        filename = f"{output_dir}/mat_{param_str}.png"
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)