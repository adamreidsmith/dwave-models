from collections import defaultdict
from itertools import product
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from dwave.system import DWaveSampler, EmbeddingComposite

# Set the environment variable for your DWave API token
with open('./dwave_api_token.txt', 'r') as f:
    os.environ['DWAVE_API_TOKEN'] = f.readline()

# Define the graph representing the map

# Here we define lables for each of the graph nodes
provinces = {
    0: 'Yukon',
    1: 'British Columbia',
    2: 'Northwest Territories',
    3: 'Alberta',
    4: 'Nunavut',
    5: 'Saskatchewan',
    6: 'Manitoba',
    7: 'Ontario',
    8: 'Quebec',
    9: 'Newfoundland and Labrador',
    10: 'New Brunswick',
    11: 'Nova Scotia',
    12: 'Prince Edward Island',
}

# An edge connects regions that share a border
edges = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 5),
    (4, 5),
    (4, 6),
    (4, 8),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (8, 10),
    (8, 12),
    (9, 11),
    (9, 12),
    (10, 11),
    (10, 12),
    (11, 12),
]
canada_graph = nx.Graph()
canada_graph.add_edges_from(edges)

# Set the number of colours to colour the graph
n_colours = 3
n_regions = len(canada_graph.nodes)

# H_single is the Hamiltonian encoding the constraint that each region must have exactly one colour.
# H_adj is the Hamiltonian encoding the constraint that neighbouring regions cannot have the same colour.
# A and B are the constants of proportionality between H_single and H_adj.
# We select A > B so that it is never favourable to break the single colour constraint in favour of the
# adjacency constraint.
A, B = 3, 1  # H = A * H_single + B * H_adj

# Our Ising model has n_regions * n_colours variables, s_(i,j), each in {-1, 1}.  The linear coefficients
# are the coefficients in front of the s_(i,j) terms in the Hamiltonian, and the quadratic coefficients
# are the coefficients in front of the s_(i,j) * s_(k,l) terms.

# Linear coefficients come from H_single.
# Note that
# $\sum_{i=1}^N (n - 2 + \sum_{k=1}^n s_{i,k})^2$
# $= C + \sum_{i=1}^N \sum_{k=1}^n 2(n - 2)s_{i,k} + \sum_{i=1}^N \sum_{k=1}^n \sum_{j=k+1}^n 2s_{i,k}s_{i,j}$
# where $C$ is a constant that can be ignored.
linear_coefficients = {indices: 2 * (n_colours - 2) * A for indices in product(range(n_regions), range(n_colours))}

quadratic_coefficients = defaultdict(lambda: 0)
# Add the contributions to the quadratic coefficients from H_single
for i in range(n_regions):
    for k in range(n_colours):
        for j in range(k + 1, n_colours):
            quadratic_coefficients[((i, k), (i, j))] += 2 * A

# Add the contributions to the quadratic coefficients from H_adj
for i, j in canada_graph.edges:
    for k in range(n_colours):
        quadratic_coefficients[((i, k), (j, k))] += 1 * B

# Instantiate the DWave sampler and sample from the Ising model
sampler = DWaveSampler(solver={'qpu': True})
sampler_embedded = EmbeddingComposite(sampler)
response = sampler_embedded.sample_ising(linear_coefficients, quadratic_coefficients, num_reads=500)

# Get the lowest energy solution
best_solution = response.first.sample

# Check that we have distinct colour for each region
for i in range(n_regions):
    s = 0
    for k in range(n_colours):
        s += best_solution[(i, k)]
    assert s == -n_colours + 2, f'Failed to find distinct colour for region {i}'
print('Each region has a distinct colour!')

# Get the colours associated to each region
region_colours = [np.argmax([best_solution[(i, k)] for k in range(n_colours)]) for i in range(n_regions)]

# Check that adjacent regions don't have the same colour
for i, j in canada_graph.edges:
    assert (
        region_colours[i] != region_colours[j]
    ), f'{provinces[i]} and {provinces[j]} share a border but have the same colour!'
print('No neighbouring regions have the same colour!')

# Plot the graph of the solution
colour_strings = [
    'blue',
    'green',
    'red',
    'yellow',
    'cyan',
    'magenta',
    'black',
    'white',
    'orange',
    'purple',
    'pink',
    'brown',
    'grey',
    'olive',
    'teal',
    'hotpink',
]
colour_map = []
for node in canada_graph:
    colour_map.append(colour_strings[region_colours[node]])

nx.draw(canada_graph, node_color=colour_map, with_labels=True, labels=provinces)
plt.show()
