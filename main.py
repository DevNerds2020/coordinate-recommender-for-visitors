from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import requests
import json
import time

class RouteOptimizer:
    def __init__(self, coordinates, min_number_of_routes, max_number_of_routes):
        self.coordinates = coordinates
        self.max_number_of_routes = max_number_of_routes
        self.min_number_of_routes = min_number_of_routes

    def optimize(self):
        models = self.cluster_data()
        best_model = self.calculate_best_model(models)
        return best_model

    def cluster_data(self):
        cluster_sizes = range(self.min_number_of_routes, self.max_number_of_routes)
        models = []
        for size in cluster_sizes:
            model = AgglomerativeClustering(
                n_clusters=size, affinity="euclidean", linkage="ward"
            ).fit(self.coordinates)
            models.append(model)
        return models

    def calculate_distance_batch(self, points):
        time.sleep(1)
        if len(points) < 2:
            return 0

        requests_string = "http://router.project-osrm.org/route/v1/car/"

        for index, (lat, lon) in enumerate(points):
            requests_string += f"{lon},{lat}"
            if index < len(points) - 1:
                requests_string += ";"

        requests_string += "?overview=false"

        r = requests.get(requests_string)
        routes = json.loads(r.content)
        
        return routes.get("routes")[0]["distance"]
    
    def calculate_cluster_score(self, model):
        cluster_labels = model.labels_
        clusters = {}
        for index, cluster_label in enumerate(cluster_labels):
            if cluster_label not in clusters:
                clusters[cluster_label] = []
            clusters[cluster_label].append(self.coordinates[index])

        total_distance = 0
        for cluster in clusters.values():
            total_distance += self.calculate_distance_batch(cluster)
        return total_distance

    def calculate_best_model(self, models):
        scores = []
        for model in models:
            score = self.calculate_cluster_score(model)
            scores.append(score)
        best_model_index = scores.index(min(scores))
        return models[best_model_index]


