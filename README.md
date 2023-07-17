# QUBO/Ising Formulations

This repository implements QUBO formulations of select problems and obtains solutions using a D-Wave quantum annealer, accessed through the D-Wave Ocean SDK.

## Map Colouring
The map colouring problem is the problem of colouring regions on a map such that no two adjacent regions have the same colour.  It is an NP-hard problem that which can be formulated as an Ising model, and as such, can be solved on a quantum annealer. [map_colouring.py](./map_colouring.py) utilizes the D-Wave Ocean SDK to solve the map colouring problem for the provinces of Canada.

## Travelling Salesman
The travelling salesman problem (TSP) asks the following question: "Given a list of destinations and a path between each pair of destinations, what is the shortest possible route that visits each destination exactly once and returns to its starting point?".  This is an NP-complete problem that has several applications such as vehicle routing, scheduling, and PCB design.

[travelling_salesman.py](./travelling_salesman.py) provides an example of the solution of the TSP for randomly chosen destinations in the city of Calgary.  Random nodes from the graph of streets obtained from OpenStreetMap are selected, and the shortest path between each pair is computed (this is done classically, although there is a [QUBO formulation](https://ieeexplore.ieee.org/document/9186612) of the shortest path problem amenable to quantum annealing).  A QUBO formulation of the TSP is then created and solved on a D-Wave quantum annealer.