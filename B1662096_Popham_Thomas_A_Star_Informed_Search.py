# Core Python Libraries
import os
import sys
import math
import random
import gzip
from queue import PriorityQueue
from timeit import timeit

# Anaconda/Pip Python Libraries
from IPython import display
import numpy as np
import pandas as pd
import scipy as sci
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns


# Informed Search Python Libraries
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout


# Visualise the graph with the graph and position nodes
def drawGraph(graph, pos_nodes, edge_cost):
        nx.draw_networkx_nodes(graph, pos_nodes)
        nx.draw_networkx_edges(graph, pos_nodes)
        nx.draw_networkx_labels(graph, pos_nodes)
        nx.draw_networkx_edge_labels(graph, pos_nodes, edge_cost)
        plt.show()




# Define the first graph for this pathfinding solution
# Create a new graph
searchGraph1 = nx.Graph()
# Define a list of nodes with weights attached to the edges between each node
searchGraph1.add_edges_from([
    ('A', 'B', {"weight": 2}),
    ('A', 'C', {"weight": 4}),
    ('A', 'D', {"weight": 5}),
    ('G', 'F', {"weight": 3}),
    ('B', 'C', {"weight": 4}),
    ('F', 'M', {"weight": 4}),
    ('C', 'G', {"weight": 2}),
    ('B', 'C', {"weight": 2}),
    ('C', 'D', {"weight": 2}),
    ('D', 'H', {"weight": 2}),
    ('G', 'H', {"weight": 2}),
    ('J', 'K', {"weight": 2}),
    ('H', 'I', {"weight": 1}),
    ('E', 'J', {"weight": 2}),
    ('H', 'F', {"weight": 2}),
    ('E', 'G', {"weight": 2}),
    ('K', 'I', {"weight": 3}),
    ('K', 'L', {"weight": 4}),
    ('L', 'N', {"weight": 2}),
    ('M', 'G', {"weight": 2}),
    ('N', 'O', {"weight": 2}),
    ('P', 'N', {"weight": 3}),
    ('O', 'M', {"weight": 2}),
    ('M', 'N', {"weight": 3})
])

# Check the number of nodes and edges
print("Search Graph 1: {}".format(searchGraph1))

print("\n")

nxPosNodes1 = nx.spring_layout(searchGraph1)

# Check the position of the nodes
print("Node position for searchGraph1: {}".format(nxPosNodes1))

graphEdgeCostLabel1 = nx.get_edge_attributes(searchGraph1, 'weight')

print("Cost of edges: {}".format(graphEdgeCostLabel1))

drawGraph(searchGraph1, nxPosNodes1, graphEdgeCostLabel1)




# Define the second graph for this pathfinding solution

# Create the directed graph using the nodes and weighted edges in a list
searchGraph2 = nx.Graph()
searchGraph2.add_edges_from([
    ('A', 'B', {"weight": 4}),
    ('A', 'C', {"weight": 2}),
    ('B', 'F', {"weight": 3}),
    ('F', 'G', {"weight": 5}),
    ('B', 'C', {"weight": 6}),
    ('C', 'G', {"weight": 2}),
    ('B', 'C', {"weight": 7}),
    ('C', 'D', {"weight": 9}),
    ('D', 'H', {"weight": 4}),
    ('G', 'H', {"weight": 1}),
    ('J', 'K', {"weight": 8}),
    ('H', 'I', {"weight": 2}),
    ('E', 'J', {"weight": 4}),
    ('H', 'F', {"weight": 3}),
    ('E', 'G', {"weight": 6}),
    ('K', 'I', {"weight": 4}),
    ('K', 'L', {"weight": 5}),
    ('L', 'N', {"weight": 2}),
    ('M', 'G', {"weight": 7}),
    ('N', 'O', {"weight": 1}),
    ('P', 'N', {"weight": 8}),
    ('O', 'M', {"weight": 3}),
    ('M', 'N', {"weight": 2})
])

# Check the number of nodes and edges
print("Search Graph 2: {}".format(searchGraph2))

print("\n")

nxPosNodes2 = nx.spring_layout(searchGraph2)

# Check the position of the nodes
print("Node position for searchGraph2: {}".format(nxPosNodes2))

graphEdgeCostLabel2 = nx.get_edge_attributes(searchGraph2, 'weight')

print("Cost of edges: {}".format(graphEdgeCostLabel2))

drawGraph(searchGraph2, nxPosNodes2, graphEdgeCostLabel2)



# Evaluate the shortest distance between two nodes in the graph
def evaluateAStar(graph, node_pos, current_node, goal_node):

  # Initialise the costs (weights) and estimated costs from the euclideanDistance method
  total_cost = dict()
  # Set the cost for the starting node
  total_cost[current_node] = 0

  priority_cost = dict()

  # Define a set that tracks the nodes that will be visited in the graph
  # Add the starting node to the new_nodes set
  new_nodes = set()
  new_nodes.add(current_node)

  node_path = dict()
  node_path[current_node] = current_node

  # Define a dict that records the previous nodes
  previous_nodes = set()

  # While the 'new' priority queue is not empty
  while not len(new_nodes) < 0:

        previous_nodes.add(current_node)

        # If the current node is equal to the goal, which is P, end the search method and return the path
        if current_node == goal_node:
          return returnPath(current_node, previous_nodes)

        first_node = []
        end_node = []

        # for node in graph.nodes:
        #   print(graph.nodes)
        #   first_node.append(node[0])
        #   end_node.append(node[1])

        node_path = set(first_node).union(set(end_node))

        for neighbours in graph.neighbors(current_node):
          # Search through the neighbouring nodes of the neighbours
          for next_node in neighbours:
            if next_node not in previous_nodes:
              previous_nodes.add(next_node)
            print("Next node: {}".format(next_node))
            costs = nx.get_edge_attributes(graph, "weight")
            print(costs)
            new_cost = total_cost[current_node] + costs[current_node, next_node]
            print("The cost between " + str(current_node) + " and " + str(next_node) + " = " + str(new_cost))
            if next_node not in total_cost or new_cost < total_cost[next_node]:
              # The current node in the graph is visited and is recorded as such
              previous_nodes.add(current_node)
              print("Previous Nodes", previous_nodes)
              # Add the new costs (as a value) to the total gathered cost of the next node (as a key)
              total_cost[next_node] = new_cost
              print("Minimum cost from start to " + str(next_node) + " has been found")
              # Add the next node in the graph
              new_nodes.add(next_node)
              print("New nodes in open list: {}".format(new_nodes))
            # Get the maximum cost of the current node
            total_cost[current_node] = 999999
            # Get the lowest cost based on the maximum cost and the current node
            lowest_cost_node = min(total_cost, key=total_cost.get)
            # If the node has a cost that is not the lowest cost, run the pathfinding method again
            if lowest_cost_node not in previous_nodes:
              evaluateAStar(graph, total_cost, lowest_cost_node, goal_node)
            # Define an empty dictionary that records the path with the least cost
            least_cost_path = dict()
            for node in node_path:
              least_cost_path[node] = '  '


# Show the graph as an external window
drawGraph(searchGraph1, nxPosNodes1, graphEdgeCostLabel1)

# Get the starting node from the input
start_node = input("Enter the starting node: ")

# Get the goal node from the input
goal_node = input("Enter the goal node: ")

# Run the evaluation method with the first graph and the necessary parameters
evaluateAStar(searchGraph1, nxPosNodes1, start_node, goal_node)


# # Show the graph as an external window
# drawGraph(searchGraph2, nxPosNodes2, graphEdgeCostLabel2)

# # Get the starting node from the input
# start_node = input("Enter the starting node: ")

# # Get the goal node from the input
# goal_node = input("Enter the goal node: ")

# # Run the evaluation method with the second graph and the necessary parameters
# evaluateAStar(searchGraph2, nxPosNodes2, start_node, goal_node)



