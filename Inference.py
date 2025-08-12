from typing import List
import numpy as np
from scipy.optimize import approx_fprime
from typing import List, Callable
from scipy.optimize import minimize

def create_Ux(n: List[int], a: List[float], b: float):
    '''Create the potential energy function U(x) based on parameters n, a, and b.
    
    Args:
        n: List of region sizes
        a: List of alpha parameters for each region
        b: Beta parameter for interaction between adjacent sites
        
    Returns:
        A function Ux(x) that calculates the potential energy for a given configuration x
    '''
    def Ux(x: List[int]) -> float:
        '''Calculate the potential energy for configuration x.
        
        Args:
            x: List of site states (-1, 0, or 1)
            
        Returns:
            The potential energy value
        '''
        u = a[0] * sum(x[:n[0]]) + b * sum(x[i] * x[i+1] for i in range(len(x)-1))
        start_idx = n[0]
        
        for i in range(1, len(n)):
            end_idx = start_idx + n[i]
            u += a[i] * sum(x[start_idx:end_idx])
            start_idx = end_idx
        
        return -u
    
    return Ux

def get_W(n: int, a: float, b: float) -> np.ndarray:
    '''Calculate the W matrix raised to the (n-1)th power.
    
    Args:
        n: Power to raise the matrix (K in original paper)
        a: Alpha parameter
        b: Beta parameter
        
    Returns:
        The W matrix raised to the (n-1)th power
    '''
    if n == 1:
        return np.array([[1.0, 0.0], [0.0, 1.0]])

    exp_a = np.exp(a)
    exp_b = np.exp(b)
    cosh_a = 0.5 * (exp_a + 1.0 / exp_a)
    sinh_a = exp_a - cosh_a

    aux1 = exp_b * cosh_a
    aux2 = np.sqrt(1.0 + exp_b**4 * sinh_a**2)
    lambda1N = (aux1 - 1.0 / exp_b * aux2)**(n - 1)
    lambda2N = (aux1 + 1.0 / exp_b * aux2)**(n - 1)

    aux1 = -exp_b**2 * sinh_a
    e1 = np.array([aux1 - aux2, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.array([aux1 + aux2, 1.0])
    e2 /= np.linalg.norm(e2)

    return np.outer(e1, e1) * lambda1N + np.outer(e2, e2) * lambda2N

def get_V(a: List[float], b: float) -> np.ndarray:
    '''Calculate the V matrix for given alpha parameters and beta.
    
    Args:
        a: List of two alpha parameters [alpha1, alpha2]
        b: Beta parameter
        
    Returns:
        The V matrix
    '''
    exp_b = np.exp(b)
    exp_a_p = np.exp(0.5 * (a[1]+a[0]))
    exp_a_m = np.exp(0.5 * (a[1]-a[0]))

    return np.array([
        [exp_b / exp_a_p, exp_a_m / exp_b],
        [1.0 / (exp_b * exp_a_m), exp_b * exp_a_p]
    ])

def get_u(a: float) -> np.ndarray:
    '''Calculate the u vector for given alpha parameter.
    
    Args:
        a: Alpha parameter
        
    Returns:
        The u vector
    '''
    exp_aux = np.exp(a/2.0)
    return np.array([
        [1.0/exp_aux],
        [exp_aux]
    ])

def comp_Z(n: List[int], a: List[float], b: float):
    '''Calculate the partition function Z.
    
    Args:
        n: List of region sizes
        a: List of alpha parameters
        b: Beta parameter
        
    Returns:
        The partition function value (with minimum value 1e-100 for numerical stability)
    '''
    y = np.dot(get_u(a[0]).T, get_W(n[0], a[0], b))
    
    if len(n)>1:
        for i in range(1,len(n)):
            y = np.dot(y, np.dot(get_V([a[i-1], a[i]], b), get_W(n[i], a[i], b)))
    y = np.dot(y, get_u(a[-1]))

    return max(1e-100, y)

def get_grad_logZ(n: List, θhat: List) -> np.ndarray:
    '''Numerically compute the gradient of the log partition function.
    
    Args:
        n: List of region sizes
        θhat: Parameter vector (α1,...,αk,β)
        
    Returns:
        Gradient vector of log Z with respect to parameters
    '''
    def f(θ):
        return np.log(comp_Z(n, θ[:-1], θ[-1])).item()

    epsilon = np.sqrt(np.finfo(float).eps)
    grad_logZ = approx_fprime(θhat, f, epsilon)

    return grad_logZ

ETA_MAX_ABS = 5.0

def check_boundary(θhat: List[float]) -> bool:
    '''Check if any parameter is at or beyond the boundary.
    
    Args:
        θhat: Parameter vector
        
    Returns:
        True if any parameter is at/beyond boundary, False otherwise
    '''
    return np.any(np.isclose(np.abs(θhat), ETA_MAX_ABS, atol=5e-2)) or np.any(np.abs(θhat) > ETA_MAX_ABS)

def comp_g(r: List[int], a: List[float], b: float, ap1: float, ap2: float) -> float:
    '''Compute the scaling factor g.
    
    Args:
        r: List of region sizes
        a: List of alpha parameters
        b: Beta parameter
        ap1: Left boundary condition
        ap2: Right boundary condition
        
    Returns:
        The scaling factor value (with minimum value 1e-100 for numerical stability)
    '''
    if all(v == 0 for v in r):
        return 1.0

    y = np.dot(get_u(ap1).T, get_W(r[0],a[0],b))
    if len(r) > 1:
        for i in range(1,len(r)):
            y = np.dot(y, np.dot(get_V(a[i-1:i+1], b) , get_W(r[i], a[i], b)))

    y = np.dot(y, get_u(ap2))

    return max(1e-100, y)

def comp_lkhd(x: List[int], n: List[int], a: List[float], b: float) -> float:
    '''Compute the likelihood of configuration x.
    
    Args:
        x: Configuration vector
        n: List of region sizes
        a: List of alpha parameters
        b: Beta parameter
        
    Returns:
        The likelihood value
    '''
    if len(x) == 1:
        return 0.5 * np.exp(x[0] * a[0])/np.cosh(a[0])

    Z = comp_Z(n,a,b)
    Ux = create_Ux(n,a,b)(x)

    zerost = [i + 1 for i, val in enumerate(np.abs(np.array(x[1:])) - 
                                        np.abs(np.array(x[:-1]))) if val == -1]
    zeroend = [i + 1 for i, val in enumerate(np.abs(np.array(x[1:])) -
                                        np.abs(np.array(x[:-1]))) if val == 1] 
    
    if x[0] == 0:
        zerost.insert(0,0)
    if x[-1] == 0:
        zeroend.append(len(x))

    subid = np.concatenate([np.full(n[i], i + 1) for i in range(len(n))])

    sf = 1.0
    for i in range(len(zerost)):
        ap1 = a[0] if zerost[i] == 0 else 2.0 * x[zerost[i] - 1] * b + a[subid[zerost[i]] - 1]
        ap2 = a[-1] if zeroend[i] == sum(n) else 2.0 * x[zeroend[i]] * b + a[subid[zeroend[i] - 1] - 1]
        
        b_id = subid[zerost[i]:zeroend[i]]
        unique_ids, counts = np.unique(b_id, return_counts=True)
        n_miss = [counts[unique_ids == bid][0] for bid in unique_ids]

        sf *= comp_g(n_miss, [a[bid - 1] for bid in unique_ids], b, ap1, ap2)

    return np.exp(-Ux) * sf / Z

def create_Llkhd(n: List[int], xobs: List[List[int]]) -> Callable[[List[float]], float]:
    '''Create the negative log-likelihood function for given data.
    
    Args:
        n: List of region sizes
        xobs: List of observed configurations
        
    Returns:
        A function that computes negative log-likelihood for given parameters
    '''
    def Llkhd_fun(theta: List[float]) -> float:
        '''Compute negative log-likelihood for given parameters.
        
        Args:
            theta: Parameter vector (α1,...,αk,β)
            
        Returns:
            Negative log-likelihood value
        '''
        aux = 0.0
        a = theta[:-1]
        b = theta[-1]

        Ux = create_Ux(n, a, b)
        logZ = np.log(comp_Z(n, a, b))

        subid = np.concatenate([np.full(n[i], i + 1) for i in range(len(n))])
        
        for x in xobs:
            zerost = np.where(np.abs(x[1:]) - np.abs(x[:-1]) == -1)[0] + 1
            zeroend = np.where((np.abs(x[1:]) - np.abs(x[:-1])) == 1)[0] + 1
            
            if x[0] == 0:
                zerost = np.insert(zerost, 0, 0)
            if x[-1] == 0:
                zeroend = np.append(zeroend, len(x))

            for i in range(len(zerost)):
                ap1 = a[0] if zerost[i] == 0 else 2.0 * x[zerost[i] - 1] * b + a[subid[zerost[i]-1] - 1]
                ap2 = a[-1] if zeroend[i] == sum(n) else 2.0 * x[zeroend[i]] * b + a[subid[zeroend[i]-1]-1]
                
                b_id = subid[zerost[i]-1:zeroend[i]-1]
                n_miss = [np.count_nonzero(b_id == n_val) for n_val in np.unique(b_id)]
                unique_vals = np.unique(b_id)
                indices = [int(i) - 1 for i in unique_vals]
                a_sub = [a[idx] for idx in indices]
                aux += np.log(comp_g(n_miss, a_sub, b, ap1, ap2))

            aux += -Ux(x)

        return -aux + len(xobs) * logZ

    return Llkhd_fun

def est_alpha(xobs: List[List[int]]) -> List[float]:
    '''Estimate initial alpha parameters from observed data.
    
    Args:
        xobs: List of observed configurations
        
    Returns:
        Initial estimate of alpha parameters [alpha, 0.0]
    '''
    phat = sum(1 for x in xobs if x == [1]) / len(xobs)
    a = 0.5 * (np.log(phat) - np.log(1.0-phat))
    return [min(max(-ETA_MAX_ABS,a),ETA_MAX_ABS),0.0]

def est_theta_sa(n: List[int], xobs: List[List[int]]) -> List[float]:
    '''Estimate parameters using simulated annealing.
    
    Args:
        n: List of region sizes
        xobs: List of observed configurations
        
    Returns:
        Estimated parameter vector [α1,...,αk,β]
    '''
    from scipy.optimize import dual_annealing
    if sum(n) == 1:
        return est_alpha(xobs)

    L = create_Llkhd(n, xobs)
    bounds = [(-ETA_MAX_ABS, ETA_MAX_ABS)] * (len(n) + 1)

    result = dual_annealing(L, bounds=bounds)
    return result.x.tolist()

def comp_mml(n: List[int], grad_logZ) -> float:
    '''Compute the mean methylation level (MML).
    
    Args:
        n: List of region sizes
        grad_logZ: Gradient of log partition function
        
    Returns:
        Mean methylation level (rounded to 8 decimal places)
    '''
    return np.round(0.5 * (1.0 + 1.0 / sum(n) * sum(grad_logZ[0:len(n)])),8).item()

def comp_nme(n: List[int], θ: List[float], grad_logZ) -> float:
    '''Compute the normalized methylation entropy (NME).
    
    Args:
        n: List of region sizes
        θ: Parameter vector
        grad_logZ: Gradient of log partition function
        
    Returns:
        Normalized methylation entropy (absolute value, rounded to 8 decimal places)
    '''
    return np.abs(np.round(1.0 / (sum(n) * np.log(2))*(np.log(comp_Z(n, θ[:-1], θ[-1])) - np.dot(θ, grad_logZ)).item(),8)).item()

def comp_cmd(n: List[int], θ1: List[float], θ2: List[float], grad_logZ1, grad_logZ2) -> float: 
    '''Compute the contrast methylation divergence (CMD).
    
    Args:
        n: List of region sizes
        θ1: First parameter vector
        θ2: Second parameter vector
        grad_logZ1: Gradient for first parameters
        grad_logZ2: Gradient for second parameters
        
    Returns:
        Contrast methylation divergence value
    '''
    θγ = 0.5 * (np.array(θ1)+np.array(θ2))

    logZ1 = np.log(comp_Z(n, θ1[0:-1], θ1[-1]))
    logZ2 = np.log(comp_Z(n, θ2[0:-1], θ2[-1]))
    logZγ = np.log(comp_Z(n, θγ[0:-1], θγ[-1]))

    cmd = logZ1 + logZ2 - (np.dot(θ1, grad_logZ1) + np.dot(θ2, grad_logZ2))
    cmd /= 2.0 * logZγ - np.dot(θγ, (np.array(grad_logZ1) + np.array(grad_logZ2)))

    return 1.0 - cmd.item()

def comp_cmd_unnorm(n: List[int], θ1: List[float], θ2: List[float], grad_logZ1s, grad_logZ2s) -> float:
    '''Compute the unnormalized contrast methylation divergence.
    
    Args:
        n: List of region sizes
        θ1: First parameter vector
        θ2: Second parameter vector
        grad_logZ1s: Gradient for first parameters
        grad_logZ2s: Gradient for second parameters
        
    Returns:
        Unnormalized contrast methylation divergence value
    '''
    θγ = 0.5 * (np.array(θ1)+np.array(θ2))

    logZ1 = np.log(comp_Z(n, θ1[0:-1], θ1[-1]))
    logZ2 = np.log(comp_Z(n, θ2[0:-1], θ2[-1]))
    logZγ = np.log(comp_Z(n, θγ[0:-1], θγ[-1]))

    cmd = 2.0 * logZγ - np.dot(θγ, (np.array(grad_logZ1) + np.array(grad_logZ2))) - (logZ1 + logZ2 - (np.dot(θ1,grad_logZ1)+np.dot(θ2,grad_logZ2)))

    return cmd.item()

def comp_pdr(n: List[int], θ: List[float]) -> float: 
    '''Compute the partially disordered ratio (PDR).
    
    Args:
        n: List of region sizes
        θ: Parameter vector
        
    Returns:
        Partially disordered ratio (rounded to 8 decimal places)
    '''
    Z = comp_Z(n, θ[0:-1], θ[-1])
    pdr = 2.0 * np.exp((sum(n)-1)* θ[-1])* np.cosh(np.dot(θ[0:-1],n))
    pdr /= Z
    return 1.0 - np.round(pdr,8).item()

def comp_chalm(n: List[int], θ: List[float]) -> float: 
    '''Compute the completely homogeneous allelic likelihood (CHALM).
    
    Args:
        n: List of region sizes
        θ: Parameter vector
        
    Returns:
        Completely homogeneous allelic likelihood (rounded to 8 decimal places)
    '''
    Z = comp_Z(n, θ[0:-1], θ[-1])
    chalm = 1.0/np.exp(np.dot(θ[0:-1],n)-θ[-1]*(sum(n)-1))
    chalm /= Z
    return 1.0 - np.round(chalm,8).item()

def comp_mcr(n: List[int], θ: List[float], grad_logZ) -> float: 
    '''Compute the methylation correlation ratio (MCR).
    
    Args:
        n: List of region sizes
        θ: Parameter vector
        grad_logZ: Gradient of log partition function
        
    Returns:
        Methylation correlation ratio value
    '''
    chalm = comp_chalm(n,θ)
    mml = comp_mml(n,grad_logZ)
    return chalm-mml