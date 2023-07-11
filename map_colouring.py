from collections import defaultdict
import os
import numpy as np
import networkx as nx
from dwave.system import DWaveSampler, EmbeddingComposite

# Set the environment variable for your DWave API token
os.environ['DWAVE_API_TOKEN'] = 'YOUR_API_TOKEN'

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
graph = nx.Graph()
graph.add_edges_from(edges)

n_colours = 4
n_regions = len(graph.nodes)

linear_coefficients = {i: 2 * (n_colours - 2) for i in range(n_regions * n_colours)}

quadratic_coefficients = defaultdict(lambda: 0)
A, B = 5, 1  # H = A * H_single + B * H_adj

# Add the coefficients from H_single
for i in range(n_regions):
    for k in range(n_colours):
        for j in range(k + 1, n_colours):
            quadratic_coefficients[(n_colours * i + k, n_colours * i + j)] += 2 * A

# Add the coeddicients from H_adj
for i, j in graph.edges:
    for k in range(n_colours):
        quadratic_coefficients[n_colours * i + k, n_colours * j + k] += 1


# sampler = DWaveSampler(solver={'qpu': True})
# sampler_embedded = EmbeddingComposite(sampler)
# response = sampler_embedded.sample_ising(linear_coefficients, quadratic_coefficients, num_reads=100)

# best_solution = response.first.sample
# best_energy = response.first.energy

# print(best_solution)
# print(best_energy)

best_solution = {
    0: -1,
    1: 1,
    2: -1,
    3: 1,
    4: 1,
    5: -1,
    6: 1,
    7: -1,
    8: 1,
    9: 1,
    10: -1,
    11: -1,
    12: -1,
    13: -1,
    14: 1,
    15: 1,
    16: -1,
    17: -1,
    18: 1,
    19: 1,
    20: -1,
    21: 1,
    22: 1,
    23: -1,
    24: 1,
    25: 1,
    26: -1,
    27: -1,
    28: -1,
    29: -1,
    30: 1,
    31: 1,
    32: 1,
    33: 1,
    34: -1,
    35: -1,
    36: -1,
    37: -1,
    38: 1,
    39: 1,
    40: -1,
    41: -1,
    42: 1,
    43: 1,
    44: 1,
    45: 1,
    46: -1,
    47: -1,
    48: -1,
    49: -1,
    50: 1,
    51: 1,
}
best_energy = -304.0
