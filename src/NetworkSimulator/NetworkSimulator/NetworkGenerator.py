import numpy as np
import networkx as nx
import json
import os
import pickle
import random as rnd
import queue
import matplotlib.pyplot as plt
import time

from .utilz import Utilz as utl

class NetworkGenerator:

    def __init__(self):

        self.component_count = {}
        self.graph = nx.Graph()
    
    def one_to_one(self,node_1,node_2,link=None):
        """Connect two nodes in the graph with a link that has a specified label.
        If the arguments for node_1 and node_2 are found to be the same as the 
        name of nodes already in the graph, then those found will be used.
        Otherwise, two new nodes (with unique numerical identifiers) will be added to 
        the graph e.g. if node_1 and node_2 = 'Node' then 'node_0' and 'node_1' will
        be added to the graph.

        Arguments:
        
        node_1/node_2 -- name of the node type/name of a node in the graph
        link -- name of the link type to connect the two nodes
        """
        if not self._node_is_in_graph(node_1):
            node_1 = self.add_node(node_1)
            
        if not self._node_is_in_graph(node_2):
            node_2 = self.add_node(node_2)

        self.add_link(node_1,node_2,label=link)
    
    def all_to_one(self,node_1,all_nodes,link=None):
        """Apply the 'one_to_one' method to a list of nodes, where
        node_1 is the node to which all nodes in all_nodes will be connected.
        If all_nodes is not a list of node names that are already in the graph, 
        then it should be a list like (for 3 'cpu' and 4 'gpu'):
        all_nodes = ['cpu']*3 + ['gpu']*4

        Arguments:

        node_1 -- name of node to which all other nodes will be connected
        all_nodes -- list of node names to be connected to node_1
        link -- name of the link type to connect all node-node pairs.
        """
        if not self._node_is_in_graph(node_1):
            node_1 = self.add_node(node_1)

        for node in all_nodes:
            self.one_to_one(node_1,node,link)
    
    def all_to_all(self,all_nodes,link=None):
        """Establish all-to-all connectivity amongst a set of nodes.
        all_nodes should be treated the same as in all_to_one.

        Arguments:

        all_nodes -- list of node names to be inter-connected.
        link -- name of the link type to connect all node-node pairs.
        """
        node_labels = []

        for node in all_nodes:
            if not self._node_is_in_graph(node):
                node_labels.append(self.add_node(node))
        
        if not node_labels:
            node_labels = all_nodes
                
        for node in node_labels:
            nodes = node_labels.copy()
            nodes.remove(node)
            self.all_to_one(node,nodes,link)

    def add_node(self,label):
        """Add a node to the graph.

        Arguments:
        
        label -- name of node type to be added.
        """
        self._increment_component_count(label)
        node_id = '{}_{}'.format(label,self.component_count[label])
        self.graph.add_node(node_id)

        return node_id
    
    def add_link(self,node_1,node_2,label=None):
        """Add a link to the graph.

        Arguments:
        
        label -- name of link type to be added.
        """
        self._increment_component_count(label)
        link_id = '{}_{}'.format(label,self.component_count[label])
        if label is None:
            self.graph.add_edge(node_1,node_2)
        else:
            self.graph.add_edge(node_1,node_2,label=link_id)

    def nodes_in_graph_of_type(self,label):

        all_nodes = list(self.graph.nodes)
        return [item for item in all_nodes if item.split('_')[0] == label]

    def show_graph(self,with_labels=False):
        
        try:
            nx.draw(self.graph,self.pos,with_labels=with_labels)
            plt.show()
        except:
            nx.draw(self.graph,with_labels=with_labels)
            plt.show()
    
    def _increment_component_count(self,label):
        """Keep record of types of components in graph and 
        how many of each
        """
        if label in self.component_count.keys():
            self.component_count[label] += 1
        else:
            self.component_count[label] = 0
    
    def _node_is_in_graph(self,label_id):
        """Check if node with particular name is in the graph.
        """
        if label_id in list(self.graph.nodes):
            return True
        else:
            return False
    
    def _link_is_in_graph(self,label_id):
        """Check if link with particular name is in the graph.
        """
        edges = []

        for edge in self.graph.edges:
            edges.append(self.graph[edge[0]][edge[1]]['label'])

        if label_id in edges:
            return True
        else:
            return False