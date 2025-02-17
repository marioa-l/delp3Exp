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
        #print("CPTS adapted")

    def getEntropy(self):
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
        prob = 0.00
        for world in worlds:
            evidence = {i: 1 if world[i] > 0 else 0 for i
                                                    in range(0, len(world))}
            prob += self.get_sampling_prob(evidence)
        return prob

    def get_sampling_prob(self, evidence):
        self.ie.setEvidence(evidence)
        try:
            return self.ie.evidenceProbability()
        except:
            # Proability join = 0
            return 0.00 

    def gen_samples(self, samples):
        time_as_key = time.strftime("%Y%m%d-%H%M%S")
        self.generator.drawSamples(samples)
        self.generator.toCSV(self.path + self.name + 'samples' + str(samples) + time_as_key + '.csv')
        samplesToReturn = []
        # Load the csv and return samples as list
        with open(self.path + self.name + 'samples' + str(samples) + time_as_key + '.csv', 'r') as read_obj:
            csv_dict_reader = DictReader(read_obj)
            for row in csv_dict_reader:
                asdict = dict(row)
                world = [int(value) for value in list(asdict.values())]
                samplesToReturn.append([world, asdict])
        samplesToReturn.sort()
        unique_samples = list(samplesToReturn for samplesToReturn,_ in 
                                            itertools.groupby(samplesToReturn))
        repeated_samples = samples - len(unique_samples)
        return [unique_samples, repeated_samples]

    def gen_samples_with_prob(self, samples):
        self.generator.drawSamples(samples)
        self.generator.toCSV(self.path + 'samples.csv')
        samplesToReturn = []
        # Load the csv and return samples as list
        with open(self.path + 'samples.csv', 'r') as read_obj:
            csv_dict_reader = DictReader(read_obj)
            spinner = Spinner("Loading samples...")
            for row in csv_dict_reader:
                asdict = dict(row)
                world = [int(value) for value in list(asdict.values())]
                prob = self.get_sampling_prob(asdict)
                samplesToReturn.append([world, asdict, prob])
                spinner.next()
            spinner.finish()
        return samplesToReturn

    def show_prob_dist(self, nSamples):
        samples = [''.join(str(x) for x in world) for [world, asDict]
                                                in self.gen_samples(nSamples)]
        toHist=[int(elem,2) for elem in samples]
        plt.hist(toHist, 10)
        plt.show()

    def load_bn(self):
        bn = gum.loadBN(self.path + self.name + '.bifxml')
        self.generator = gum.BNDatabaseGenerator(bn)
        self.ie = gum.LazyPropagation(bn)
        self.bn = bn
        self.structure = [self.bn.nodes(), self.bn.arcs()]

def create_random_dag(nodes, edges):
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
    #asUndGraph = nx.Graph(G)
    #treewidht_h1 = treewidth_min_degree(asUndGraph)
    #treewidht_h2 = treewidth_min_fill_in(asUndGraph)
    #print(treewidht_h1)
    #print(treewidht_h2)
    return G
