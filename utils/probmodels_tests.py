import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from  probmodels import *
import matplotlib.pyplot as plt
import time
import math

def compare_entropy_by_network_size(vars, edges_ratio=1.5):
    """
    Compare entropy calculation methods across different network sizes.
    
    Args:
        vars (list): List of numbers of variables to test
        edges_ratio (float): Ratio of edges to nodes (default 1.5)
    
    Returns:
        dict: Results for each network size
    """
    
    all_results = {}
    
    for n_vars in vars:
        print(f"\nTesting network with {n_vars} variables...")
        
        n_edges = int(n_vars * edges_ratio)
        bn = BayesNetwork(f"random_{n_vars}", "local/")
        bn.build_save_random_BN(n_vars, n_edges, True)
        
        # Get results for this network
        results = bn.compare_entropy_methods()
        all_results[n_vars] = results
        
        # Calcular error relativo como porcentaje
        exact_entropy = results['exact']['entropy']
        pyagrum_error_percent = 100 * abs(results['pyagrum']['entropy'] - exact_entropy) / exact_entropy
        
        print(f"Exact entropy: {exact_entropy:.4f}")
        print(f"pyAgrum entropy: {results['pyagrum']['entropy']:.4f}")
        print(f"Computation times:")
        print(f"  Exact: {results['exact']['time']:.2f}s")
        print(f"  pyAgrum: {results['pyagrum']['time']:.2f}s")
        print(f"Relative error: {pyagrum_error_percent:.2f}%")
    
    # Plot results
    plot_comparison_by_size(all_results, vars)
    
    return all_results

def plot_comparison_by_size(all_results, vars):
    """
    Plot comparison results across different network sizes.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prepare data
    exact_times = []
    pyagrum_times = []
    relative_errors = []  # Cambio a error relativo
    
    for n_vars in vars:
        results = all_results[n_vars]
        exact_times.append(results['exact']['time'])
        pyagrum_times.append(results['pyagrum']['time'])
        # Calcular error relativo como porcentaje
        exact_entropy = results['exact']['entropy']
        relative_error = 100 * abs(results['pyagrum']['entropy'] - exact_entropy) / exact_entropy
        relative_errors.append(relative_error)
    
    # Plot computation times
    ax1.plot(vars, exact_times, 'o-', label='Exact')
    ax1.plot(vars, pyagrum_times, 'o-', label='pyAgrum')
    ax1.set_title('Computation Time vs Network Size')
    ax1.set_xlabel('Number of Variables')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot relative errors
    ax2.plot(vars, relative_errors, 'o-', label='pyAgrum')
    ax2.set_title('Relative Error vs Network Size')
    ax2.set_xlabel('Number of Variables')
    ax2.set_ylabel('Relative Error (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot time in log scale
    ax3.semilogy(vars, exact_times, 'o-', label='Exact')
    ax3.semilogy(vars, pyagrum_times, 'o-', label='pyAgrum')
    ax3.set_title('Computation Time vs Network Size (log scale)')
    ax3.set_xlabel('Number of Variables')
    ax3.set_ylabel('Time (seconds)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot relative error in log scale
    ax4.semilogy(vars, relative_errors, 'o-', label='pyAgrum')
    ax4.set_title('Relative Error vs Network Size (log scale)')
    ax4.set_xlabel('Number of Variables')
    ax4.set_ylabel('Relative Error (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()


vars = [2, 4, 6, 8, 10, 12, 14, 16]
results = compare_entropy_by_network_size(vars)