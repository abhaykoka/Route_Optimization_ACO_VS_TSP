import numpy as np
import pandas as pd
import time
import random

# Load the CSV file containing city distances
file_path = 'route_app/static/Cities Dataset - Route Optimization.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Extract all cities and initialize a distance matrix for Floyd-Warshall
cities = pd.concat([df['Origin'], df['Destination']]).unique()
n = len(cities)
city_indices = {city: idx for idx, city in enumerate(cities)}
distance_matrix = np.full((n, n), np.inf)

# Populate the distance matrix
for _, row in df.iterrows():
    i, j = city_indices[row['Origin']], city_indices[row['Destination']]
    distance_matrix[i, j] = row['Distance']
    distance_matrix[j, i] = row['Distance']  # Since it's undirected

# Set the diagonal to 0 (distance from a city to itself is 0)
np.fill_diagonal(distance_matrix, 0)

# Implement Floyd-Warshall algorithm to compute shortest paths
def floyd_warshall_with_paths(matrix):
    n = matrix.shape[0]
    dist = matrix.copy()
    next_node = np.full((n, n), -1)

    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] < np.inf:
                next_node[i, j] = j

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    next_node[i, j] = next_node[i, k]

    return dist, next_node

# Reconstruct the full path using the next_node matrix
def reconstruct_path(u, v, next_node):
    if next_node[u, v] == -1:
        return []  # No path exists
    path = [u]
    while u != v:
        u = next_node[u, v]
        path.append(u)
    return path

# Get shortest path distance and next_node matrix
shortest_path_matrix, next_node_matrix = floyd_warshall_with_paths(distance_matrix)

# Get distance between two cities
def get_distance(city1, city2):
    i, j = city_indices[city1], city_indices[city2]
    return shortest_path_matrix[i, j]

# Get the full path between two cities
def get_full_path(city1, city2):
    i, j = city_indices[city1], city_indices[city2]
    path_indices = reconstruct_path(i, j, next_node_matrix)
    return [cities[idx] for idx in path_indices]

# Ant Colony Optimization (ACO) algorithm
class AntColony:
    def __init__(self, cities, city_indices, distance_matrix, n_ants=10, n_iterations=100, alpha=1, beta=2, evaporation_rate=0.5):
        self.cities = cities
        self.city_indices = city_indices
        self.distance_matrix = distance_matrix
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta  # Distance heuristic importance
        self.evaporation_rate = evaporation_rate
        self.pheromone = np.ones(distance_matrix.shape)

    def run(self):
        best_route = None
        best_distance = float('inf')
        best_full_sequence = []

        for iteration in range(self.n_iterations):
            all_routes = []
            for _ in range(self.n_ants):
                route, full_sequence = self.construct_route()
                distance = self.full_sequence_distance(full_sequence)
                all_routes.append((route, distance, full_sequence))
                if distance < best_distance:
                    best_route = route
                    best_distance = distance
                    best_full_sequence = full_sequence

            self.update_pheromone(all_routes)

        return best_route, best_distance, best_full_sequence

    def construct_route(self):
        route = [random.choice(self.cities)]  # Start from a random city
        full_sequence = [route[0]]  # To track the sequence of all cities visited
        while len(route) < len(self.cities):
            current_city = route[-1]
            probabilities = self.calculate_probabilities(current_city, route)
            next_city = random.choices(self.cities, probabilities)[0]
            route.append(next_city)

            # Add intermediate cities to the sequence
            intermediate_path = get_full_path(current_city, next_city)
            full_sequence.extend(intermediate_path[1:])  # Avoid duplication

        return route, full_sequence

    def calculate_probabilities(self, current_city, visited):
        probabilities = []
        current_idx = self.city_indices[current_city]
        for city in self.cities:
            if city not in visited:
                next_idx = self.city_indices[city]
                pheromone = self.pheromone[current_idx, next_idx]
                distance = self.distance_matrix[current_idx, next_idx]
                probabilities.append((pheromone ** self.alpha) * ((1 / distance) ** self.beta))
            else:
                probabilities.append(0)
        total = sum(probabilities)
        return [p / total for p in probabilities]

    def full_sequence_distance(self, full_sequence):
        total_distance = 0
        for i in range(len(full_sequence) - 1):
            total_distance += get_distance(full_sequence[i], full_sequence[i + 1])
        return total_distance

    def update_pheromone(self, all_routes):
        self.pheromone *= (1 - self.evaporation_rate)  # Evaporate pheromone
        for route, distance, _ in all_routes:
            for i in range(len(route) - 1):
                self.pheromone[self.city_indices[route[i]], self.city_indices[route[i + 1]]] += 1 / distance

def find_Optimal_path(selected_cities):
# User input for cities to be traveled


# Start measuring ACO execution time
    start_time = time.time()

    # Initialize and run the ACO algorithm
    aco = AntColony(cities=selected_cities, city_indices=city_indices, distance_matrix=shortest_path_matrix)
    optimal_route, total_distance, full_sequence = aco.run()

    # End time after ACO finishes
    end_time = time.time()

    # Calculate and print execution time (in milliseconds)
    execution_time_ms = (end_time - start_time) * 1000
    formatted_time=format(execution_time_ms, ".4f")
    # Output the results
    print("\nOptimal Route:", " -> ".join(optimal_route))
    print("Total Distance:", total_distance)
    print("Full Sequence of Cities Traversed:", " -> ".join(full_sequence))
    print(f"\nTime Complexity (ACO): O(n^2 * iterations)")
    print(f"ACO Execution Time: {execution_time_ms:.2f} ms")
    
    return " -> ".join(optimal_route), " -> ".join(full_sequence), total_distance, formatted_time