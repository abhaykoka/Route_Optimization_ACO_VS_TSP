import pandas as pd
import numpy as np
import time

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
            return [current_city], 0  # No need to return to start city, distance is zero

        min_tour = None
        min_distance = np.inf

        for city in all_cities - visited_cities:
            tour, distance = find_optimal_tour(city, visited_cities | {city})
            total_distance = distance + distance_matrix[current_city][city]
            if total_distance < min_distance:
                min_distance = total_distance
                min_tour = [current_city] + tour

        return min_tour, min_distance

    optimal_tour, total_distance = find_optimal_tour(0, {0})  # Start from the first city

    # Reconstruct the full path using the next_city matrix, including all cities in the path
    full_tour = []
    for i in range(len(optimal_tour) - 1):
        path_segment = reconstruct_path(global_indices[optimal_tour[i]], global_indices[optimal_tour[i + 1]], next_city)
        full_tour.extend(path_segment if i == 0 else path_segment[1:])

    # Filtered tour to include only input cities
    filtered_tour = [city for city in full_tour if city in global_indices]

    return full_tour, filtered_tour, total_distance

# Main program with time complexity calculation
def find_optimal_route(selected_cities):
    distance_matrix, next_city, city_to_index, cities = create_distance_matrix(data)

    # Get user input for cities to be traveled
   

    # Get the indices of the selected cities
    city_indices = [city_to_index[city] for city in selected_cities]

    # Create a reduced distance matrix for the selected cities
    reduced_matrix = distance_matrix[np.ix_(city_indices, city_indices)]

    # Optimal TSP
    start_time_opt = time.time()
    full_tour, filtered_tour, total_distance_opt = optimal_tsp_scipy(reduced_matrix, next_city, city_indices)
    end_time_opt = time.time()

    # Convert indices back to city names for display
    full_tour_cities = [cities[i] for i in full_tour]
    filtered_tour_cities = [cities[i] for i in filtered_tour]

    # Display results
    print("\nOptimal TSP:")
    optimal_route = " -> ".join(filtered_tour_cities)
    full_sequence = " -> ".join(full_tour_cities)
    execution_time= end_time_opt - start_time_opt
    formatted_time = format(execution_time, ".4f")
    print("Optimal Route:", " -> ".join(filtered_tour_cities))
    print("\n")
    print("Full Sequence of Cities Traversed:", " -> ".join(full_tour_cities))    
    print("Total Distance:", total_distance_opt)
    print(f"Time Complexity (Optimal TSP): O(n!), Execution Time: {end_time_opt - start_time_opt:.4f} seconds")
    return optimal_route, full_sequence, total_distance_opt, formatted_time


