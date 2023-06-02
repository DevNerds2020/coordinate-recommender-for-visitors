import csv
import random
import math

def generate_sample_dataset(num_locations, num_visitors):
    # Generate random coordinates within Tehran city limits
    min_lat, max_lat = 35.6, 35.9
    min_lon, max_lon = 51.2, 51.6
    locations = []
    for i in range(1, num_locations+1):
        lat = round(random.uniform(min_lat, max_lat), 6)
        lon = round(random.uniform(min_lon, max_lon), 6)
        locations.append((i, lat, lon))

    # Generate random money spent for each location
    money_spent = [random.randint(50, 200) for _ in range(num_locations)]

    # Cluster the locations
    clusters = [[] for _ in range(num_visitors)]
    for i, loc in enumerate(locations):
        cluster_id = i % num_visitors
        clusters[cluster_id].append(loc + (money_spent[i],))

    return clusters

def save_dataset_to_csv(clusters):
    with open('sample_dataset.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Location ID', 'Latitude', 'Longitude', 'Money Spent'])
        for i, cluster in enumerate(clusters):
            for location in cluster:
                writer.writerow([location[0], location[1], location[2], location[3]])

# Set the number of locations and visitors
num_locations = 500
num_visitors = 25

# Generate the sample dataset
clusters = generate_sample_dataset(num_locations, num_visitors)

# Save the dataset to a CSV file
save_dataset_to_csv(clusters)
