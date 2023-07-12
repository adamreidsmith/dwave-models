from collections import defaultdict
from itertools import product
import os
import random
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

from dwave.system import DWaveSampler, EmbeddingComposite

# Set the environment variable for your DWave API token
with open('./dwave_api_token.txt', 'r') as f:
    os.environ['DWAVE_API_TOKEN'] = f.readline()

SEED = 11223344550
random.seed(SEED)

A, B = 100, 1
NUM_READS = 500

# Number of random points in the city of Calgary on which we will compute the solution of the TSP.
# Due to the limited connectivity of the chimera graph of the D-Wave QPU, this must be kept fairly low in order
# to find a valid embedding.
n_points = 7

# Load the city boundary data
city = gpd.read_file('CityBoundary/geo_export_829045c7-570b-4230-a4fd-8e88e4b92774.shp')

# Get the bounds of the city
minx, miny, maxx, maxy = city.geometry.total_bounds

# Get a graph of the road network in Calgary
print('Obtaining graph of the city of Calgary from OSM...')
calgary_graph = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type='drive')

# Generate n_points random points within the city of Calgary
points = set()
while len(points) < n_points:
    point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
    if not any(city.geometry.contains(point)):
        continue
    # Replace the random point with the nearest graph node
    nearest_node = ox.nearest_nodes(calgary_graph, point.x, point.y)
    if nearest_node not in points:
        points.add(nearest_node)
points = list(points)

# Compute the shortest paths between each pair of points
weighted_edges = []
weight_dict = {}
shortest_path_dict = {}
for start_node, end_node in tqdm(
    product(points, points), total=n_points**2, desc='Computing network shortest paths'
):
    if start_node == end_node:
        continue

    # Find the shortest path and its length in km
    shortest_path = nx.shortest_path(calgary_graph, start_node, end_node, weight='length')
    shortest_path_length = sum(
        calgary_graph.edges[shortest_path[i], shortest_path[i + 1], 0]['length'] for i in range(len(shortest_path) - 1)
    )
    # Convert the length to km
    shortest_path_length /= 1000

    weighted_edges.append((start_node, end_node, shortest_path_length))
    weight_dict[(start_node, end_node)] = shortest_path_length
    shortest_path_dict[(start_node, end_node)] = shortest_path

# Create a graph representing the order locations and the shortest paths between them
tsp_graph = nx.DiGraph()
tsp_graph.add_weighted_edges_from(weighted_edges)

# Define the dicitonary to hold our QUBO coefficients
QUBO_matrix = defaultdict(lambda: 0)

# The first part of the Hamiltonian enforces that each vertex only appears once in a cycle
for v in range(n_points):
    for i in range(n_points):
        QUBO_matrix[((v, i), (v, i))] -= A
        for j in range(i + 1, n_points):
            QUBO_matrix[((v, i), (v, j))] += 2 * A

# The second enforces that there must be an ith node in the cycle for each i
for i in range(n_points):
    for v in range(n_points):
        QUBO_matrix[((v, i), (v, i))] -= A
        for w in range(v + 1, n_points):
            QUBO_matrix[((v, i), (w, i))] += 2 * A

# The third enforces that two vertices cannot appear consecutively in the cycle if there is no edge connecting them
node_list = list(tsp_graph.nodes)
edge_set = set(tsp_graph.edges)
for u, v in product(range(n_points), range(n_points)):
    if (node_list[u], node_list[v]) in edge_set:
        continue
    for i in range(n_points):
        QUBO_matrix[((u, i), (v, (i + 1) % n_points))] += A

# The last adds the TSP constraint, i.e. that the sum of the weights along the path is minimized
for u, v in product(range(n_points), range(n_points)):
    if (node_list[u], node_list[v]) in edge_set:
        for i in range(n_points):
            QUBO_matrix[((u, i), (v, (i + 1) % n_points))] += B * weight_dict[(node_list[u], node_list[v])]

# Instantiate the DWave sampler and sample from the QUBO model
sampler = DWaveSampler(solver={'qpu': True})
sampler_embedded = EmbeddingComposite(sampler)
response = sampler_embedded.sample_qubo(QUBO_matrix, num_reads=NUM_READS)

solution = response.first.sample

# Check that each vertex is visited exactly once
for v in range(n_points):
    s = 0
    for i in range(n_points):
        s += solution[(v, i)]
    assert s == 1, f'Node {v} visited more or less than one time!'

# Check that each step has exactly one associated vertex
for i in range(n_points):
    s = 0
    for v in range(n_points):
        s += solution[(v, i)]
    assert s == 1, f'Step {i} is not associated to exaclty one node!'
print('The solution represents a valid path!')

# Get the nodes associated with the TSP route
path = [-1] * n_points
for (v, i), x in solution.items():
    if x == 1:
        path[i] = v

# Rotate the path such that we start at node 0
zero_index = path.index(0)
path = path[zero_index:] + path[:zero_index]

# Replace the node indices by their labels in the tsp_graph
path = [node_list[node] for node in path]

print(f'Solution: {path}')
print(
    f'Path length: {sum(tsp_graph.edges[path[i], path[(i + 1) % n_points]]["weight"] for i in range(n_points)):.3f} km'
)

# Check if the solution provided by networkx agrees with our solution
if n_points < 8:
    print('Computing networkx solution...')
    nx_solution = nx.algorithms.approximation.traveling_salesman_problem(tsp_graph, weight='weight', nodes=points)[:-1]
    for i in range(n_points):
        if path[i:] + path[:i] == nx_solution:
            print('Solution agrees with networkx!')
            break
    else:
        fail_str = (
            f'Solution does not agree with networkx! Try adjusting A (currently set to {A}) and\n'
            + f'B (currently set to {B}). A lower A/B ratio will increase the importance of minimizing\n'
            + f'the total path length, but setting A/B too low will break the other constraints! Also,\n'
            + f'increasing NUM_READS (currently set to {NUM_READS}) will increase the probability of\n'
            + f'finding a minimum energy solution.'
        )
        print(fail_str)
        nx_path_length = sum(
            tsp_graph.edges[nx_solution[i], nx_solution[(i + 1) % n_points]]["weight"] for i in range(n_points)
        )
        print(f'Networkx path length: {nx_path_length:.3f} km')

# Plot the solution
route = [shortest_path_dict[(path[i], path[(i + 1) % n_points])] for i in range(n_points)]
route_colors = ['red'] * n_points
if n_points < 8:
    route = [shortest_path_dict[(nx_solution[i], nx_solution[(i + 1) % n_points])] for i in range(n_points)] + route
    route_colors = (['green'] * n_points) + route_colors
ox.plot_graph_routes(calgary_graph, route, route_colors=route_colors, show=False, close=False)

legend_handles = [mlines.Line2D([], [], color='red', marker='_', markersize=15, label='Quantum solution')]
if n_points < 8:
    legend_handles.append(mlines.Line2D([], [], color='green', marker='_', markersize=15, label='networkx solution'))
plt.legend(handles=legend_handles, loc='upper right')
plt.show()
