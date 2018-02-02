from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd


class GeoCoordSimulator:

    def __init__(self, geo_coord_df):
        self.geo_coord_df = geo_coord_df
        self.clusters = None

    def calculate_clusters(self, eps=0.001, min_samples=10):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(self.geo_coord_df)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        unique_labels = set(labels)

        cluster_results = pd.DataFrame(columns=['min_lat', 'max_lat', 'min_long', 'max_long'])
        clusters_len = np.array([])
        for k in unique_labels:

            class_member_mask = (labels == k)
            all_clust_members = self.geo_coord_df[class_member_mask]

            cluster_results = cluster_results.append({
                'min_lat': min(all_clust_members['Latitude']),
                'max_lat': max(all_clust_members['Latitude']),
                'min_long': min(all_clust_members['Longitude']),
                'max_long': max(all_clust_members['Longitude'])
            }, ignore_index=True)
            clusters_len = np.append(clusters_len, len(all_clust_members))

        clust_prob = clusters_len / len(self.geo_coord_df)

        cluster_results['unique_labels'] = unique_labels
        cluster_results['probabilities'] = clust_prob
        cluster_results['clusters_len'] = clusters_len
        cluster_results.index = cluster_results['unique_labels']
        self.clusters = cluster_results

        return cluster_results

    def get_random_coordinates(self, n=10):
        if self.clusters is None:
            raise ValueError("You must call calculate_clusters method in order to generate random coordinates")

        choiced_clusters = np.random.choice(self.clusters['unique_labels'], n, p=self.clusters['probabilities'])
        simulated_lat_long = np.zeros((n, 2))
        for c, r in zip(choiced_clusters, range(0, n)):
            row = self.clusters.iloc[c]
            simulated_lat_long[r, 0] = np.random.uniform(row['min_lat'], row['max_lat'], 1)
            simulated_lat_long[r, 1] = np.random.uniform(row['min_long'], row['max_long'], 1)

        return simulated_lat_long
