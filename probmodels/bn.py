# Class for Bayesian Network
import json
import networkx as nx
from networkx.algorithms.approximation.treewidth import *
import random
import pyAgrum as gum
import pyAgrum.lib.bn2graph as gumGraph
from csv import DictReader
from progress.spinner import Spinner
import matplotlib.pyplot as plt
import itertools
import math
import numpy as np
import time

class BayesNetwork:

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.generator = ''
        self.ie = ''
        self.bn = ''
        self.structure = ''


    def build_save_random_BN(self, nNodes, nEdges, randomCPTs):
        dGraph = create_random_dag(nNodes, nEdges)
        dGraphNodes = dGraph.nodes()
        dGraphEdges = dGraph.edges()
        dGraphEdges = [(str(A), str(B)) for (A, B) in dGraphEdges]
        bn = gum.BayesNet(self.name)
        [bn.add(gum.LabelizedVariable(str(var),str(var),2)) for var
                                                                in dGraphNodes]
        for edge in dGraphEdges:
            bn.addArc(edge[0], edge[1])
        if randomCPTs:
            # For generate all CPTs
            bn.generateCPTs()
        # To graph and save BN
        #gumGraph.dotize(bn, self.path + self.name, 'pdf')
        gum.saveBN(bn, self.path + self.name + '.bifxml')
        self.generator = gum.BNDatabaseGenerator(bn)
        self.ie = gum.LazyPropagation(bn)
        self.bn = bn
        self.structure = [dGraphNodes, dGraphEdges]


    def build_save_BN(self, dGraphNodes, dGraphEdges, randomCPTs):
        dGraphEdges = [(str(A), str(B)) for (A, B) in dGraphEdges]
        bn = gum.BayesNet(self.name)
        [bn.add(gum.LabelizedVariable(str(var),str(var),2)) for var
                                                                in dGraphNodes]
        for edge in dGraphEdges:
            bn.addArc(edge[0], edge[1])
        if randomCPTs:
            # For generate all CPTs
            bn.generateCPTs()
        # To graph and save BN
        gumGraph.dotize(bn, self.path + self.name, 'pdf')
        gum.saveBN(bn, self.path + self.name + '.bifxml')
        self.generator = gum.BNDatabaseGenerator(bn)
        self.ie = gum.LazyPropagation(bn)
        self.bn = bn
        self.structure = [dGraphNodes, dGraphEdges]


    def get_nodes_information(self):
        """
        Retrieves information about the nodes in the Bayesian network.

        This method categorizes nodes into four groups:
        1. Nodes without parents.
        2. Nodes with exactly two parents.
        3. Nodes with more than two parents.
        4. Nodes with at least one child.

        Returns:
            dict: A dictionary containing the categorized nodes with the following keys:
            - 'nodes_no_parents': List of nodes without parents.
            - 'nodes_2_parents': List of nodes with exactly two parents.
            - 'nodes_more_parents': List of nodes with more than two parents.
            - 'nodes_with_childrens': List of nodes with at least one child.
        """
        nodes = list(self.bn.nodes())
        nodes_without_parents = [] # Nodes without parents
        nodes_with_2_parents = [] # Nodes with two parents
        nodes_with_more_parents = [] # Nodes with parents > 2
        nodes_with_childrens = [] # Nodes with at least one children
        for node in nodes:
            parents = len(self.bn.parents(node))
            childrens = len(self.bn.children(node))
            if parents == 0:
                nodes_without_parents.append(node)
            elif parents == 2:
                nodes_with_2_parents.append(node)
            elif parents > 2:
                nodes_with_more_parents.append(node)

            if childrens != 0:
                nodes_with_childrens.append(node)

        nodesInformation = {
            'nodes_no_parents': nodes_without_parents,
            'nodes_2_parents': nodes_with_2_parents,
            'nodes_more_parents': nodes_with_more_parents,
            'nodes_with_childrens': nodes_with_childrens
        }
        return nodesInformation


    def make_CPTs(self, nodes, alpha):
        for node in nodes:
            parents = list(self.bn.parents(node))
            if len(parents) != 0:
                parValues = list(itertools.product([1, 0],repeat=len(parents)))
                for parVal in parValues:
                    prnode = "{:.2f}".format(random.uniform(alpha, 1))
                    complementnode = "{:.2f}".format(1.00 - float(prnode))
                    change_prob = np.random.random()
                    if change_prob > 0.50:
                        newCPT = [float(complementnode), float(prnode)]
                    else:
                        newCPT = [float(prnode), float(complementnode)]
                    self.bn.cpt(node)[{str(parents[index]):value
                                for index, value in enumerate(parVal)}]=newCPT
            else:
                prnode = "{:.2f}".format(random.uniform(alpha, 1))
                complementnode = "{:.2f}".format(1.00 - float(prnode))
                change_prob = np.random.random()
                if change_prob > 0.50:
                    newCPT = [float(complementnode), float(prnode)]
                else:
                    newCPT = [float(prnode), float(complementnode)]
                self.bn.cpt(node).fillWith(newCPT)
        othersnodes = list(self.bn.nodes())
        for othernode in othersnodes:
            if not othernode in nodes:
                self.bn.generateCPT(othernode)
        gum.saveBN(self.bn, self.path + self.name + '.bifxml')


    def getEntropy(self):
        """
        Calculates the entropy of the Bayesian Network.

        The entropy is computed using the formula: H = -∑ P(x)log₂P(x), where x represents
        all possible combinations of values for the network variables.

        Returns:
            float: The entropy value of the Bayesian Network. A lower value indicates more
            certainty in the probability distribution, while a higher value indicates more
            uncertainty.

        Notes:
            - The method generates all possible combinations of binary values (0,1) for all nodes
            - For each combination, it calculates its probability using get_sampling_prob
            - A progress spinner is displayed during calculation due to potentially long computation times
            - Zero probabilities are handled by setting their contribution to 0 (lim[p->0] p*log(p) = 0)
        """
        cNodes = len(self.structure[0])
        samples = list(itertools.product([1,0], repeat=cNodes))
        sum = 0.00
        print(len(samples))
        spinner = Spinner("Calculating entropy...")
        for sample in samples:
            evidence = {i: sample[i] for i in range(0, len(sample))}
            prSample = self.get_sampling_prob(evidence)
            if prSample != 0:
                term = prSample * math.log2(prSample)
            else:
                term = 0
            sum += term
            spinner.next()
        spinner.finish()
        return - sum


    def get_probs_Worlds(self, worlds):
        """
        Calculate the total probability for a collection of possible worlds.

        Args:
            worlds (list): List of worlds where each world is represented as a list/array 
                          of values >0 or <=0 that will be converted to binary evidence.

        Returns:
            float: Sum of probabilities for all provided worlds based on sampling probabilities.

        Example:
            worlds = [[1,0,1], [0,1,1]]  # Two possible worlds
            prob = get_probs_Worlds(worlds)  # Returns combined probability
        """
        prob = 0.00
        for world in worlds:
            evidence = {i: 1 if world[i] > 0 else 0 for i
                                                    in range(0, len(world))}
            prob += self.get_sampling_prob(evidence)
        return prob


    def get_sampling_prob(self, evidence):
        """
        Calculate the probability of the given evidence in the Bayesian Network.

        Args:
            evidence (dict): Dictionary containing variable-value pairs representing evidence
                            where keys are variable names and values are their observed states.

        Returns:
            float: The probability of the evidence. Returns 0.0 if the probability cannot be computed
                   or if the evidence has zero probability in the network.

        Examples:
            >>> bn.get_sampling_prob({'A': 1, 'B': 0})  # Where A and B are variables in network
            0.25  # Returns probability of A=1 and B=0 occurring together
        """
        self.ie.setEvidence(evidence)
        try:
            return self.ie.evidenceProbability()
        except:
            # Probability join = 0
            return 0.00 


    def gen_samples(self, samples):
        """
        Generate samples from the Bayesian Network and return unique samples with their count.
        
        Args:
            samples (int): Number of samples to generate
        
        Returns:
            list: A list containing [unique_samples, number_of_repeated_samples] where:
                  - unique_samples is a list of [world, dict] pairs
                  - number_of_repeated_samples is the count of duplicates found
        """
        samplesToReturn = []
        database = self.generator.drawSamples(samples)
        
        # Convert samples directly to the desired format
        for i in range(samples):
            world = []
            asdict = {}
            for node in self.bn.nodes():
                value = database.get(i)[node]
                world.append(int(value))
                asdict[str(node)] = value
            samplesToReturn.append([world, asdict])
        
        # Sort and remove duplicates
        samplesToReturn.sort()
        unique_samples = list(samplesToReturn for samplesToReturn,_ in 
                                            itertools.groupby(samplesToReturn))
        repeated_samples = samples - len(unique_samples)
        
        return [unique_samples, repeated_samples]


    def gen_samples_with_prob(self, samples):
        """
        Generate samples from the Bayesian Network and return them with their probabilities.
        
        Args:
            samples (int): Number of samples to generate
        
        Returns:
            list: A list of [world, asdict, prob] where:
                  - world is a list of binary values (0,1)
                  - asdict is a dictionary mapping node names to values
                  - prob is the probability of that sample occurring
        """
        samplesToReturn = []
        database = self.generator.drawSamples(samples)
        
        spinner = Spinner("Generating samples with probabilities...")
        for i in range(samples):
            world = []
            asdict = {}
            for node in self.bn.nodes():
                value = database.get(i)[node]
                world.append(int(value))
                asdict[str(node)] = value
            
            prob = self.get_sampling_prob(asdict)
            samplesToReturn.append([world, asdict, prob])
            spinner.next()
        
        spinner.finish()
        return samplesToReturn


    def show_prob_dist(self, nSamples, bins=None):
        """
        Visualize the probability distribution of the Bayesian Network through sampling.
        
        Args:
            nSamples (int): Number of samples to generate
            bins (int, optional): Number of bins for histogram. If None, uses 2^(number of nodes)
        """
        samples = [''.join(str(x) for x in world) for [world, asDict] 
                  in self.gen_samples(nSamples)[0]] 
        toHist = [int(elem,2) for elem in samples]
        
        if bins is None:
            bins = min(2**len(self.structure[0]), 50) 
            
        plt.hist(toHist, bins, density=True)
        plt.title(f'Probability Distribution (n={nSamples} samples)')
        plt.xlabel('World State (binary to decimal)')
        plt.ylabel('Relative Frequency')
        plt.show()


    def load_bn(self):
        """
        Loads a Bayesian Network from a .bifxml file and initializes related components.

        The method performs the following operations:
        1. Loads the Bayesian Network from the specified .bifxml file path
        2. Creates a database generator for the network
        3. Initializes lazy propagation inference engine
        4. Stores the network structure (nodes and arcs)

        Attributes modified:
            self.generator: BNDatabaseGenerator object for the loaded network
            self.ie: LazyPropagation inference engine for the network
            self.bn: The loaded Bayesian Network
            self.structure: List containing network nodes and arcs

        Returns:
            None

        Requires:
            self.path and self.name to be properly set to locate the .bifxml file
        """
        bn = gum.loadBN(self.path + self.name + '.bifxml')
        self.generator = gum.BNDatabaseGenerator(bn)
        self.ie = gum.LazyPropagation(bn)
        self.bn = bn
        self.structure = [self.bn.nodes(), self.bn.arcs()]


def create_random_dag(nodes, edges):
    """
    Create a random Directed Acyclic Graph (DAG) with a specified number of nodes and edges.

    Parameters:
    nodes (int): The number of nodes in the graph.
    edges (int): The number of edges to add to the graph.

    Returns:
    networkx.DiGraph: A randomly generated directed acyclic graph.

    Notes:
    - The function ensures that the generated graph is acyclic by checking for cycles after each edge addition.
    - If adding an edge creates a cycle, the edge is removed and another edge is attempted.
    """
    # Generate a random DGraph
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = random.randint(0, nodes - 1)
        b = a
        while b == a:
            b = random.randint(0, nodes - 1)
        G.add_edge(a, b)
        if nx.is_directed_acyclic_graph(G):
            edges -= 1
        else:
            # Closed a loop
            G.remove_edge(a, b)
    return G
