from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
from main import *
app = Flask(__name__)

@app.route('/cluster', methods=['POST'])
def cluster_and_assign(data):
    data = request.get_json()
    coordinates = pd.DataFrame(data)
    min_number_of_routes = 2
    max_number_of_routes = 10
    route_optimizer = RouteOptimizer(coordinates, min_number_of_routes, max_number_of_routes)
    best_model = route_optimizer.optimize()
    return jsonify(best_model.labels_.tolist())
    
if __name__ == '__main__':
    app.run(debug=True)
