import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'route_app/static/Cities Dataset - Route Optimization.csv'

data = pd.read_csv(file_path)

# Create a distance matrix using Floyd-Warshall for all-pairs shortest path
def create_distance_matrix(data):
    cities = pd.concat([data['Origin'], data['Destination']]).unique()
    n_cities = len(cities)
    distance_matrix = np.full((n_cities, n_cities), np.inf)
    next_city = np.full((n_cities, n_cities), -1)

    city_to_index = {city: index for index, city in enumerate(cities)}

    for _, row in data.iterrows():
        origin_idx = city_to_index[row['Origin']]
        destination_idx = city_to_index[row['Destination']]
        distance = row['Distance']

        # Since it's undirected, distance is the same both ways
        distance_matrix[origin_idx, destination_idx] = distance
        distance_matrix[destination_idx, origin_idx] = distance
        next_city[origin_idx, destination_idx] = destination_idx
        next_city[destination_idx, origin_idx] = origin_idx

    # Set diagonal to 0 (distance to itself)
    np.fill_diagonal(distance_matrix, 0)

    # Floyd-Warshall algorithm to handle cases with no direct connections
    for k in range(n_cities):
        for i in range(n_cities):
            for j in range(n_cities):
                if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]:
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
                    next_city[i][j] = next_city[i][k]

    return distance_matrix, next_city, city_to_index, cities

# Function to reconstruct the full path between two cities using backtracking
def reconstruct_path(start, end, next_city):
    if next_city[start, end] == -1:
        return []  # No path exists
    path = [start]
    while start != end:
        start = next_city[start, end]
        path.append(start)
    return path

# Optimal TSP using iterative approach with backtracking to reconstruct path
def optimal_tsp_scipy(distance_matrix, next_city, global_indices):
    n_cities = len(distance_matrix)
    all_cities = set(range(n_cities))

    def find_optimal_tour(current_city, visited_cities):
        if visited_cities == all_cities:
            return [current_city], distance_matrix[current_city][0]  # Return to the start

        min_tour = None
        min_distance = np.inf

        for city in all_cities - visited_cities:
            tour, distance = find_optimal_tour(city, visited_cities | {city})
            total_distance = distance + distance_matrix[current_city][city]
            if total_distance < min_distance:
                min_distance = total_distance
                min_tour = [current_city] + tour

        return min_tour, min_distance

    optimal_tour, total_distance = find_optimal_tour(0, {0})
    optimal_tour.append(0)  # Add start city to the end for a complete cycle

    # Reconstruct the full path using the next_city matrix
    full_tour = []
    for i in range(len(optimal_tour) - 1):
        path_segment = reconstruct_path(global_indices[optimal_tour[i]], global_indices[optimal_tour[i + 1]], next_city)
        full_tour.extend(path_segment if i == 0 else path_segment[1:])

    return full_tour, total_distance

# Main program with time complexity calculation
def plot_exec_time(num_cities):
    distance_matrix, next_city, city_to_index, cities = create_distance_matrix(data)

    # Ask user for the number of cities to test

    num_cities_list = range(1, num_cities + 1)  # Testing with 1 to user-defined number of cities
    opt_times = []

    for n in num_cities_list:
        selected_cities = []

        for i in range(n):
            city_name = cities[i]  # Select the first n cities from the dataset
            selected_cities.append(city_name)

        # Get the indices of the selected cities
        city_indices = [city_to_index[city] for city in selected_cities]

        # Create a reduced distance matrix for the selected cities
        reduced_matrix = distance_matrix[np.ix_(city_indices, city_indices)]

        # Optimal TSP
        start_time_opt = time.time()
        tour_opt, total_distance_opt = optimal_tsp_scipy(reduced_matrix, next_city, city_indices)
        end_time_opt = time.time()

        opt_times.append((end_time_opt - start_time_opt) * 1000)  # Convert to milliseconds

    # Plotting the results
    plt.figure(figsize=(10, 4))
    plt.plot(num_cities_list, opt_times, label='Optimal TSP', marker='o')
    plt.title('Execution Time vs Number of Cities for Optimal TSP')
    plt.xlabel('Number of Cities')
    plt.ylabel('Execution Time (ms)')
    plt.xticks(num_cities_list)
    plt.grid()
    plt.legend()
    output_path = 'route_app/static/media/opt_tsp_plot.png'
    plt.savefig(output_path)
    plt.close()
    return output_path[10:]


