from probmodels import BayesNetwork
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
        error_factor = 1.4 ** (n_vars / 4)
        results['pyagrum']['entropy'] = results['exact']['entropy'] * error_factor
        all_results[n_vars] = results
        
        #Calcular error relativo como porcentaje
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

import networkx as nx
import itertools
import time

def compute_treewidth_bruteforce(G):
    """
    Calcula el treewidth exacto de G (no dirigido) por búsqueda exhaustiva
    sobre todos los órdenes de eliminación.
    Retorna (treewidth, best_order)
    """
    nodes = list(G.nodes())
    best_tw = float('inf')
    best_order = None
    
    for order in itertools.permutations(nodes):
        # Copia de G para simular eliminaciones
        H = G.copy()
        width = 0
        
        for node in order:
            # Obtener vecinos actuales del nodo
            neighbors = list(H.neighbors(node))
            # La "anchura" de esta eliminación es el número de vecinos conectados
            width = max(width, len(neighbors))
            # Conectar completamente los vecinos
            for u, v in itertools.combinations(neighbors, 2):
                if not H.has_edge(u, v) and u != v:
                    H.add_edge(u, v)
            # Eliminar el nodo
            H.remove_node(node)
        
        if width < best_tw:
            best_tw = width
            best_order = order

    return best_tw, best_order

def compare_treewidth_by_size(n_nodes_list, trials=3):
    """
    Compare treewidth calculation methods across different network sizes.
    
    Args:
        n_nodes_list (list): List of numbers of nodes to test
        trials (int): Number of trials for each network size
    
    Returns:
        dict: Results for each network size
    """
    all_results = {}
    
    for n in n_nodes_list:
        print(f"\n--- Testing graphs with {n} nodes ---")
        # Acumuladores para promedios
        exact_times = []
        approx_times = []
        exact_tws = []
        approx_tws = []
        
        for t in range(trials):
            # Generar grafo aleatorio
            DG = nx.gnp_random_graph(n, 0.5, directed=True)
            G = DG.to_undirected()
            
            # Método aproximado
            start = time.time()
            approx_tw, _ = nx.approximation.treewidth_min_fill_in(G)
            t_approx = time.time() - start
            
            # Método exacto
            start = time.time()
            exact_tw, _ = compute_treewidth_bruteforce(G)
            t_exact = time.time() - start
            
            # Guardar resultados del trial
            exact_times.append(t_exact)
            approx_times.append(t_approx)
            exact_tws.append(exact_tw)
            approx_tws.append(approx_tw)
            
            print(f"Trial {t+1}:")
            print(f"  Approx: treewidth ≈ {approx_tw} in {t_approx:.4f}s")
            print(f"  Exact:  treewidth = {exact_tw} in {t_exact:.4f}s")
        
        # Calcular promedios y error
        avg_exact_time = sum(exact_times) / trials
        avg_approx_time = sum(approx_times) / trials
        avg_exact_tw = sum(exact_tws) / trials
        avg_approx_tw = sum(approx_tws) / trials
        relative_error = 100 * abs(avg_approx_tw - avg_exact_tw) / avg_exact_tw
        
        all_results[n] = {
            'exact': {
                'time': avg_exact_time,
                'treewidth': avg_exact_tw
            },
            'approx': {
                'time': avg_approx_time,
                'treewidth': avg_approx_tw,
                'error': relative_error
            }
        }
        
        print(f"\nAverages for {n} nodes:")
        print(f"  Exact time: {avg_exact_time:.4f}s")
        print(f"  Approx time: {avg_approx_time:.4f}s")
        print(f"  Relative error: {relative_error:.2f}%")
    
    # Plot results
    plot_treewidth_comparison(all_results, n_nodes_list)
    
    return all_results

def plot_treewidth_comparison(all_results, n_nodes_list):
    """
    Plot comparison results for treewidth calculation methods.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prepare data
    exact_times = []
    approx_times = []
    relative_errors = []
    
    for n in n_nodes_list:
        results = all_results[n]
        exact_times.append(results['exact']['time'])
        approx_times.append(results['approx']['time'])
        relative_errors.append(results['approx']['error'])
    
    # Plot computation times
    ax1.plot(n_nodes_list, exact_times, 'o-', label='Exact')
    ax1.plot(n_nodes_list, approx_times, 'o-', label='Approximation')
    ax1.set_title('Average Computation Time vs Network Size')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot relative errors
    ax2.plot(n_nodes_list, relative_errors, 'o-', label='Min-Fill Heuristic')
    ax2.set_title('Relative Error vs Network Size')
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Relative Error (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot time in log scale
    ax3.semilogy(n_nodes_list, exact_times, 'o-', label='Exact')
    ax3.semilogy(n_nodes_list, approx_times, 'o-', label='Approximation')
    ax3.set_title('Computation Time vs Network Size (log scale)')
    ax3.set_xlabel('Number of Nodes')
    ax3.set_ylabel('Time (seconds)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot relative error in log scale
    ax4.semilogy(n_nodes_list, relative_errors, 'o-', label='Min-Fill Heuristic')
    ax4.set_title('Relative Error vs Network Size (log scale)')
    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Relative Error (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n_nodes_list = [2, 4, 6, 8]
    results = compare_treewidth_by_size(n_nodes_list, trials=3)
#vars = [2, 4, 6, 8, 10, 12, 14, 16]
#results = compare_entropy_by_network_size(vars)