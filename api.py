from flask import Flask, request, jsonify
from sklearn.cluster import KMeans

app = Flask(__name__)

@app.route('/cluster', methods=['POST'])
def cluster_and_assign():
    # Receive JSON data
    data = request.get_json()

    # Extract relevant features for clustering
    points = data.get('Points', [])
    distributor_data = data.get('Distributors', [])

    features = [[float(point['Lat']), float(point['Long']), point.get('DistributeWeight', 0)] for point in points]

    # Number of clusters (you may adjust this based on your data)
    K = 3

    # Perform clustering
    kmeans = KMeans(n_clusters=K, random_state=42)
    clusters = kmeans.fit_predict(features)

    # Assign clusters to distributors
    distributor_features = [
        [d['TotalWeight'], d['TotalVolume'], d['MaxNum'], d['MinNum']] for d in distributor_data
    ]

    distributor_clusters = kmeans.predict(distributor_features)

    # Create a dictionary to store distributor assignments
    distributor_assignments = {i: [] for i in range(K)}

    # Assign points to clusters
    for i, cluster_id in enumerate(clusters):
        distributor_assignments[cluster_id].append(points[i])

    # Filter points based on distributor constraints
    final_assignments = {}
    for distributor_id, distributor_cluster in enumerate(distributor_clusters):
        cluster_points = distributor_assignments[distributor_cluster]

        # Consider constraints (e.g., max and min number, total weight, total volume)
        max_num = distributor_data[distributor_id]['MaxNum']
        min_num = distributor_data[distributor_id]['MinNum']
        total_weight = distributor_data[distributor_id]['TotalWeight']
        total_volume = distributor_data[distributor_id]['TotalVolume']

        filtered_cluster_points = []

        for point in cluster_points:
            # Add point only if it satisfies the constraints
            if min_num <= len(filtered_cluster_points) <= max_num and \
               point['Weight'] <= total_weight and \
               point['Volumne'] <= total_volume:
                filtered_cluster_points.append(point)

        final_assignments[f'Distributor_{distributor_id + 1}'] = filtered_cluster_points

    return jsonify(final_assignments)

if __name__ == '__main__':
    app.run(debug=True)
