import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

# Load the dataset
file_path = 'route_app/static/Cities Dataset - Route Optimization.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Create a distance matrix using Floyd-Warshall for all-pairs shortest path
def create_distance_matrix(data):
    cities = pd.concat([data['Origin'], data['Destination']]).unique()
    n_cities = len(cities)
    distance_matrix = np.full((n_cities, n_cities), np.inf)

    city_to_index = {city: index for index, city in enumerate(cities)}

    for _, row in data.iterrows():
        origin_idx = city_to_index[row['Origin']]
        destination_idx = city_to_index[row['Destination']]
        distance = row['Distance']

        # Since it's undirected, distance is the same both ways
        distance_matrix[origin_idx, destination_idx] = distance
        distance_matrix[destination_idx, origin_idx] = distance

    # Set diagonal to 0 (distance to itself)
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix, city_to_index, cities

# Ant Colony Optimization (ACO) Class
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

        for iteration in range(self.n_iterations):
            all_routes = []
            for _ in range(self.n_ants):
                route = self.construct_route()
                distance = self.calculate_distance(route)
                all_routes.append((route, distance))
                if distance < best_distance:
                    best_route = route
                    best_distance = distance
            self.update_pheromone(all_routes)
        return best_route, best_distance

    def construct_route(self):
        route = [random.choice(self.cities)]  # Start from a random city
        while len(route) < len(self.cities):
            current_city = route[-1]
            next_city = random.choice([city for city in self.cities if city not in route])
            route.append(next_city)
        return route

    def calculate_distance(self, route):
        return sum(self.distance_matrix[self.city_indices[route[i]], self.city_indices[route[i + 1]]]
                   for i in range(len(route) - 1))

    def update_pheromone(self, all_routes):
        self.pheromone *= (1 - self.evaporation_rate)  # Evaporate pheromone
        for route, distance in all_routes:
            for i in range(len(route) - 1):
                self.pheromone[self.city_indices[route[i]], self.city_indices[route[i + 1]]] += 1 / distance

# Main function to run ACO and measure time complexity
def Aco_Graph(num_cities):
    # Load the dataset and create the distance matrix
    
    distance_matrix, city_to_index, cities = create_distance_matrix(data)

    # Ask the user for the number of cities to test   

    # Testing multiple runs and plotting time complexity
    num_cities_list = range(2, num_cities + 1)  # Start with 2 cities up to user input
    aco_times = []

    for n in num_cities_list:
        selected_cities = [cities[i] for i in range(n)]  # Select the first n cities

        # Create a reduced distance matrix for the selected cities
        city_indices = [city_to_index[city] for city in selected_cities]
        reduced_matrix = distance_matrix[np.ix_(city_indices, city_indices)]

        # Initialize the ACO algorithm
        aco = AntColony(cities=selected_cities, city_indices=city_to_index, distance_matrix=reduced_matrix)

        # Measure ACO execution time
        start_time = time.time()
        optimal_route, total_distance = aco.run()
        end_time = time.time()

        # Calculate execution time in milliseconds
        execution_time_ms = (end_time - start_time) * 1000
        aco_times.append(execution_time_ms)

    # Plotting the ACO execution time
    plt.figure(figsize=(10, 6))
    plt.plot(num_cities_list, aco_times, label='ACO Execution Time', marker='o', color='b')
    plt.title('ACO Execution Time vs Number of Cities')
    plt.xlabel('Number of Cities')
    plt.ylabel('Execution Time (ms)')
    plt.xticks(num_cities_list)
    plt.grid(True)
    plt.legend()
    output_path = 'route_app/static/media/Aco_plot.png'
    plt.savefig(output_path)
    plt.close()
    return output_path[10:]



