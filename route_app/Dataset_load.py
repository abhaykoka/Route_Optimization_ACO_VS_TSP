import pandas as pd
import numpy as np
# Load data and create the distance matrix
def load_data(file_path):
    """
    Loads the CSV dataset containing origin, destination, and distance.
    Creates a distance matrix to be used in the optimization algorithm.
    """
    data = pd.read_csv(file_path)
    cities = pd.concat([data['Origin'], data['Destination']]).unique()
    n_cities = len(cities)
    distance_matrix = np.full((n_cities, n_cities), np.inf)
    
    city_to_index = {city: index for index, city in enumerate(cities)}

    for _, row in data.iterrows():
        origin_idx = city_to_index[row['Origin']]
        destination_idx = city_to_index[row['Destination']]
        distance = row['Distance']
        # Since it's an undirected graph, distances are symmetric
        distance_matrix[origin_idx, destination_idx] = distance
        distance_matrix[destination_idx, origin_idx] = distance
    
    # Set the diagonal to 0 (distance from a city to itself)
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix, city_to_index, cities

