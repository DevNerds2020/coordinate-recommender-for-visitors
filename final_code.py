import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, OPTICS, Birch
from numba import njit, prange
import itertools
import math
import flask


app = flask.Flask(__name__)


def convert_list_to_df(clusters):
    df = pd.DataFrame(columns=['Location ID', 'Latitude', 'Longitude', 'Weight'])
    for i, cluster in enumerate(clusters):
        for location in cluster:
            df = df.append({'Location ID': location[0], 'Latitude': location[1], 'Longitude': location[2], 'Weight': location[3]}, ignore_index=True)
    return df

def cluster_data(data, min_cluster_size, max_cluster_size):
    cluster_sizes = range(min_cluster_size, max_cluster_size)
    models = []
    for size in cluster_sizes:
        model = AgglomerativeClustering(n_clusters=size, affinity='euclidean', linkage='ward').fit(data)
        models.append(model)
    return models


def calculate_cluster_weight(model, data):
    cluster_count = len(set(model.labels_))
    cluster_weights = []
    for i in range(cluster_count):
        cluster = data[model.labels_ == i]
        cluster_weight = cluster['Weight'].mean()
        cluster_weights.append(cluster_weight)
    return cluster_weights

def calculate_cluster_distance(model, data):
    cluster_count = len(set(model.labels_))
    cluster_distances = []
    for i in range(cluster_count):
        cluster = data[model.labels_ == i]
        cluster_distance = cluster['Latitude'].max() - cluster['Latitude'].min() + cluster['Longitude'].max() - cluster['Longitude'].min()
        cluster_distances.append(cluster_distance)
    return cluster_distances

def calculate_cluster_score(model, data):
    cluster_weights = calculate_cluster_weight(model, data)
    cluster_distances = calculate_cluster_distance(model, data)
    score = 0
    for i in range(len(cluster_weights)):
        for j in range(i + 1, len(cluster_weights)):
            score += abs(cluster_weights[i] - cluster_weights[j]) / abs(cluster_distances[i] - cluster_distances[j]) * 10
    return score

def calculate_best_model(models, data):
    scores = []
    for model in models:
        score = calculate_cluster_score(model, data)
        scores.append(score)
    best_model_index = scores.index(max(scores))
    return models[best_model_index]

def calculate_cluster_size(model, data):
    cluster_count = len(set(model.labels_))
    cluster_sizes = []
    for i in range(cluster_count):
        cluster = data[model.labels_ == i]
        cluster_size = cluster.shape[0]
        cluster_sizes.append(cluster_size)
    return cluster_sizes

def distance_between_points(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2
    radius = 6371  # km

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    sin_dlat = np.sin(dlat / 2)
    sin_dlon = np.sin(dlon / 2)
    cos_lat1 = np.cos(np.radians(lat1))
    cos_lat2 = np.cos(np.radians(lat2))

    a = sin_dlat * sin_dlat + cos_lat1 * cos_lat2 * sin_dlon * sin_dlon
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = radius * c

    return d

def calculate_shortest_path(locations):
    shortest_distance = math.inf
    shortest_path = None
    for path in itertools.permutations(locations):
        distance = 0
        for i in range(len(path) - 1):
            distance += distance_between_points(path[i], path[i + 1])
            if distance > shortest_distance:
                break  # Exit early if the distance exceeds the current shortest distance
        if distance < shortest_distance:
            shortest_distance = distance
            shortest_path = path
    return shortest_path, shortest_distance


def print_cluster_locations(model, data):
    cluster_count = len(set(model.labels_))
    for i in range(cluster_count):
        cluster = data[model.labels_ == i]
        print(cluster)

def return_locations_in_target_cluster(model, data, target_cluster):
    cluster_count = len(set(model.labels_))
    for i in range(cluster_count):
        cluster = data[model.labels_ == i]
        if i == target_cluster:
            return cluster

def return_locations_list_in_target_cluster(model, data, target_cluster):
    # return [[lat, lon], [lat, lon], ...]
    cluster_count = len(set(model.labels_))
    for i in range(cluster_count):
        cluster = data[model.labels_ == i]
        if i == target_cluster:
            return cluster[['Latitude', 'Longitude']].values.tolist()

def haversine_distance(coord1, coord2):
    """
    Calculate the Haversine distance (in meters) between two sets of latitude and longitude coordinates.
    """
    # Earth radius in meters
    earth_radius = 6371000

    # Convert latitude and longitude to radians
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1_rad = lat1 * (3.141592653589793 / 180)
    lon1_rad = lon1 * (3.141592653589793 / 180)
    lat2_rad = lat2 * (3.141592653589793 / 180)
    lon2_rad = lon2 * (3.141592653589793 / 180)

    # Haversine formula
    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad
    a = (pow(math.sin(d_lat / 2), 2) +
         math.cos(lat1_rad) * math.cos(lat2_rad) * pow(math.sin(d_lon / 2), 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c

    return distance

def nearest_neighbor_algorithm(start, locations):
    """
    Calculate the optimized path from the start point to the end point using latitude and longitude coordinates.

    Parameters:
    start (tuple): Latitude and longitude of the start point.
    locations (list): List of latitude and longitude coordinates for all points.

    Returns:
    list: The optimized path as a list of latitude and longitude coordinates.
    """
    # Create a copy of the list of locations to keep track of visited nodes
    unvisited = list(locations)
    path = [start]

    # Find the nearest neighbor for each point and add it to the path
    while unvisited:
        current = path[-1]
        nearest_dist = float('inf')
        nearest_neighbor = None

        for neighbor in unvisited:
            dist = haversine_distance(current, neighbor)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_neighbor = neighbor

        path.append(nearest_neighbor)
        unvisited.remove(nearest_neighbor)

    return path


def haversine_distance(coord1, coord2):
    """
    Calculate the Haversine distance (in meters) between two sets of latitude and longitude coordinates.
    """
    # Earth radius in meters
    earth_radius = 6371000

    # Convert latitude and longitude to radians
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1_rad = lat1 * (3.141592653589793 / 180)
    lon1_rad = lon1 * (3.141592653589793 / 180)
    lat2_rad = lat2 * (3.141592653589793 / 180)
    lon2_rad = lon2 * (3.141592653589793 / 180)

    # Haversine formula
    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad
    a = (pow(math.sin(d_lat / 2), 2) +
         math.cos(lat1_rad) * math.cos(lat2_rad) * pow(math.sin(d_lon / 2), 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c

    return distance

def held_karp_algorithm(start, locations):
    """
    Calculate the most optimized path from the start point to the end point using latitude and longitude coordinates.

    Parameters:
    start (tuple): Latitude and longitude of the start point.
    locations (list): List of latitude and longitude coordinates for all points.

    Returns:
    list: The optimized path as a list of latitude and longitude coordinates.
    """
    num_points = len(locations)
    all_points = [start] + locations

    # Initialize the memoization table for dynamic programming
    memo = {}

    # Helper function for memoization
    def dp_mask(mask, pos):
        if mask == (1 << num_points) - 1:
            return haversine_distance(all_points[pos], start)

        if (mask, pos) in memo:
            return memo[(mask, pos)]

        min_distance = float('inf')

        for next_pos in range(1, num_points + 1):
            if not (mask & (1 << next_pos)):
                new_mask = mask | (1 << next_pos)
                distance = haversine_distance(all_points[pos], all_points[next_pos]) + dp_mask(new_mask, next_pos)
                min_distance = min(min_distance, distance)

        memo[(mask, pos)] = min_distance
        return min_distance

    # Calculate the optimal distance
    optimal_distance = dp_mask(1, 0)

    # Reconstruct the path based on the memoization table
    path = [start]
    mask = 1

    for i in range(1, num_points + 1):
        for next_pos in range(1, num_points + 1):
            if not (mask & (1 << next_pos)):
                if dp_mask(mask, 0) == haversine_distance(all_points[0], all_points[next_pos]) + dp_mask(mask | (1 << next_pos), next_pos):
                    path.append(all_points[next_pos])
                    mask |= (1 << next_pos)
                    break

    return path, optimal_distance
# app.route('/cluster_data', methods=['POST'])
# def main_func