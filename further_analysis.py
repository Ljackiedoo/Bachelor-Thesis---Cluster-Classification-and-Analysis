import numpy as np
import pandas as pd

# Load CSV files
before_df = pd.read_csv('/home/samuel_data/nihao_uhd_2.79e12_zoom_6_rerun_stars_all_snapshots_chemodynamical_tagging_S=auto/star_cluster_analysis_3/all_cluster_metrics.csv')
#after_df = pd.read_csv('/home/samuel_data/nihao_uhd_2.79e12_zoom_6_rerun_stars_all_snapshots_dynamical_tagging_S=auto/star_cluster_analysis_2/cluster_formation_metrics_1.5_gyr.csv')

# Convert cluster indices to sets
#before_clusters = set(before_df['cluster_idx'])
#after_clusters = set(after_df['cluster_idx'])

# Find lost and gained clusters
# lost_clusters = before_clusters - after_clusters
# gained_clusters = after_clusters - before_clusters

# # Output results
# print(f"Lost clusters ({len(lost_clusters)}):", sorted(lost_clusters))
# print(f"Gained clusters ({len(gained_clusters)}):", sorted(gained_clusters))

star_formation_clusters = before_df[before_df['burst_cluster']]
print(f"Number of star formation clusters: {len(star_formation_clusters)}")