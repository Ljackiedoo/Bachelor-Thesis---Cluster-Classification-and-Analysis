import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy import stats
import pynbody as pb
import logging

def starClusterAnalysis(snapshot_file_paths, working_directory_path, snapshots, snapshot_conversion_factor):
    # Create output directories
    analysis_dir = f"{working_directory_path}star_cluster_analysis/"
    os.makedirs(analysis_dir, exist_ok=True)

    star_data_dir = f"{analysis_dir}star_data/"
    os.makedirs(star_data_dir, exist_ok=True)

    logging.info("Starting star cluster analysis...")
    
    # Load fuzzy cluster data
    print(f"Loading fuzzy cluster data from {working_directory_path}...")
    logging.info(f"Loading fuzzy cluster data from {working_directory_path}...")

    clustered_ids = np.load(f"{working_directory_path}clusteredIDs.npy")
    print(f"clustered_ids shape: {clustered_ids.shape}")
    ordering = np.load(f"{working_directory_path}ordering.npy")
    print(f"ordering shape: {ordering.shape}")
    fuzzy_clusters = np.load(f"{working_directory_path}fuzzyClusters.npy")
    print(f"fuzzy_clusters shape: {fuzzy_clusters.shape}")

    

    n_snapshots = len(snapshot_file_paths)
    n_clusters = len(fuzzy_clusters)
    print(f"Processing {n_snapshots} snapshots for {n_clusters} fuzzy clusters")
    logging.info(f"Processing {n_snapshots} snapshots for {n_clusters} fuzzy clusters")

    #------ load and save all star data for each snapshot from simulation (ids, ages, masses, positions)----------
    if not len(os.listdir(star_data_dir)) == n_snapshots:
        save_star_data(snapshot_file_paths, star_data_dir)

    #------ identify burst clusters from fuzzy clusters and save their metrics ----------
    if not os.path.exists(f"{analysis_dir}all_cluster_metrics.npy"):
        get_cluster_data(snapshot_file_paths, working_directory_path, analysis_dir, star_data_dir, n_snapshots, n_clusters, ordering, fuzzy_clusters, clustered_ids, snapshots, snapshot_conversion_factor)
    if not os.path.exists(f"{analysis_dir}cluster_masses.npy"):
        get_cluster_masses(snapshot_file_paths, working_directory_path, analysis_dir, star_data_dir, n_snapshots, n_clusters, ordering, fuzzy_clusters, clustered_ids, snapshots)

    cluster_metrics = np.load(f"{analysis_dir}all_cluster_metrics.npy", allow_pickle=True).item()
    cluster_masses = np.load(f"{analysis_dir}cluster_masses.npy", allow_pickle=True).item()
    
    #------Compare no. of burst clusters to no. of non-burst clusters----------
    age_distribution_analysis_dir = f"{analysis_dir}age_distribution_analysis/"
    os.makedirs(age_distribution_analysis_dir, exist_ok=True)
    #star_forming_vs_non_star_forming_age_distribution(age_distribution_analysis_dir, cluster_metrics)


    #------make plots for all fuzzy clusters---------
    fuzzy_cluster_analysis_dir = f"{analysis_dir}fuzzy_cluster_analysis/"
    os.makedirs(fuzzy_cluster_analysis_dir, exist_ok=True)

    fuzzy_cluster_analysis(analysis_dir, working_directory_path, n_snapshots, fuzzy_cluster_analysis_dir, star_data_dir, ordering, fuzzy_clusters, cluster_metrics, cluster_masses, snapshots, snapshot_conversion_factor)

    #------make plots for all burst clusters----------
    burst_cluster_analysis_dir = f"{analysis_dir}burst_cluster_analysis/"
    os.makedirs(burst_cluster_analysis_dir, exist_ok=True)

    #burst_cluster_analysis(analysis_dir, burst_cluster_analysis_dir, working_directory_path, n_snapshots, ordering, fuzzy_clusters, star_data_dir, cluster_metrics, cluster_masses, snapshots, snapshot_conversion_factor)

def calculate_cluster_metrics(star_data, member_ids, cluster_idx, snap_idx, cluster_start_snapshot, cluster_end_snapshot, snapshot_conversion_factor, is_burst):
    """Calculate metrics for a cluster at a specific snapshot."""

    logging.info(f"Calculating metrics for cluster {cluster_idx} at snapshot {snap_idx}...")

    is_member = np.isin(star_data[:, 0], member_ids)
    member_star_data = star_data[is_member]

    star_ages_in_cluster = member_star_data[:, 1]

    t25, t75 = np.percentile(star_ages_in_cluster, [25, 75])
    t25_t75 = t75 - t25

    median_age = np.median(star_ages_in_cluster)
    std_age = np.std(star_ages_in_cluster)

    star_positions = member_star_data[:, 3:6]
    star_irons = member_star_data[:, 6]
    star_oxygens = member_star_data[:, 7]

    radial_distances = np.linalg.norm(star_positions[:, :2], axis=1)
    z_distances = np.abs(star_positions[:, 2])
    median_radial = np.median(radial_distances)
    median_z = np.median(z_distances)
    median_iron = np.median(star_irons)
    median_oxygen = np.median(star_oxygens)

    cluster_structure_age = (snap_idx - cluster_start_snapshot) * snapshot_conversion_factor

    metrics = {
        'cluster_idx': cluster_idx,
        'cluster_start_snapshot': cluster_start_snapshot,
        'cluster_end_snapshot': cluster_end_snapshot,
        'star_count': len(star_ages_in_cluster),
        'median_age': median_age,
        'std_age': std_age,
        'median_radial_distance': median_radial,
        'median_z_distance': median_z,
        'median_iron': median_iron,
        'median_oxygen': median_oxygen,
        't25_t75': t25_t75,
        'burst_snapshot': snap_idx,
        'cluster_structure_age': cluster_structure_age,
        'burst_cluster': is_burst
    }       
    return metrics

def get_cluster_data(snapshot_file_paths, working_directory_path, analysis_dir, star_data_dir, n_snapshots, n_clusters, ordering, fuzzy_clusters, clustered_ids, available_snapshots, snapshot_conversion_factor, threshold_fraction = 0.2):
    """
    Load and save all data for the clusters in csv and numpy format."""
    logging.info("Starting cluster data extraction...")
    # ---- CSV file setup ----
    csv_filepath = os.path.join(analysis_dir, "all_cluster_metrics.csv")
    header_columns = ['cluster_idx', 
                      'cluster_start_snapshot', 
                      'cluster_end_snapshot', 
                      'star_count', 
                      'median_age', 
                      'std_age', 
                      'median_radial_distance', 
                      'median_z_distance', 
                      'median_iron', 
                      'median_oxygen', 
                      't25_t75', 
                      'burst_snapshot', 
                      'cluster_structure_age', 
                      'burst_cluster']
    processed_cluster_ids = set()

    # Check if file exists and is non-empty
    if os.path.exists(csv_filepath) and os.path.getsize(csv_filepath) > 0:
        try:
            df_existing = pd.read_csv(csv_filepath)
            if 'cluster_idx' in df_existing.columns and not df_existing.empty:
                processed_cluster_ids = set(df_existing['cluster_idx'].dropna().astype(int).unique())
                print(f"Resuming. Found {len(processed_cluster_ids)} already processed cluster(s) in {csv_filepath}")
                logging.info(f"Resuming. Found {len(processed_cluster_ids)} already processed cluster(s) in {csv_filepath}")
            else:
                print(f"File {csv_filepath} exists but is empty or missing 'cluster_idx' column.")
                logging.warning(f"File {csv_filepath} exists but is empty or missing 'cluster_idx' column.")
        except pd.errors.EmptyDataError:
            print(f"File {csv_filepath} exists but is unreadable as CSV.")
            logging.error(f"File {csv_filepath} exists but is unreadable as CSV.")
    else:
        # File doesn't exist or is empty – write header
        pd.DataFrame(columns=header_columns).to_csv(csv_filepath, index=False)
        print(f"Header written to {csv_filepath}")
        logging.info(f"Header written to {csv_filepath}")

    clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")
    # Pre-calculate snapshot ranges for each fuzzycat cluster
    snapshot_indices_per_cluster = [set(int(filename.split('_')[0]) for filename in clusterFileNames[ordering[start_idx:end_idx]]) for start_idx, end_idx in fuzzy_clusters]
    fuzzy_cluster_snapshot_ranges = np.array([[min(snapshots), max(snapshots)] for snapshots in snapshot_indices_per_cluster])

    min_snapshot = np.min(fuzzy_cluster_snapshot_ranges[:, 0])
    max_snapshot = np.max(fuzzy_cluster_snapshot_ranges[:, 1])

    logging.info("Preloading all star data for available snapshots...")
    all_star_data = {
        snap_idx: np.load(f"{star_data_dir}star_data_{snap_idx:03d}.npy") 
        for snap_idx in range(min_snapshot, max_snapshot + 1) if snap_idx in available_snapshots
    }

    all_metrics = []

    for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):
        if cluster_idx in processed_cluster_ids:
            print(f"Cluster {cluster_idx} already processed and in CSV. Skipping.")
            logging.info(f"Cluster {cluster_idx} already processed and in CSV. Skipping.")
            continue

        cluster_start_snapshot, cluster_end_snapshot = fuzzy_cluster_snapshot_ranges[cluster_idx]
        snapshot_check_end = cluster_start_snapshot + int(threshold_fraction * 1/snapshot_conversion_factor) + 1

        for snap_idx in range(cluster_start_snapshot, snapshot_check_end):
            if snap_idx not in available_snapshots:
                continue
            if snap_idx >= n_snapshots:
                continue

            star_data = all_star_data.get(snap_idx)

            member_ids = get_member_ids_for_fuzzycat_cluster(cluster_idx, ordering, fuzzy_clusters, working_directory_path, snap_idx)
            
            if len(member_ids)==0:
                continue
            is_member = np.isin(star_data[:, 0], member_ids)
            star_ages_in_cluster = star_data[is_member, 1]
            if len(star_ages_in_cluster) <= 20:
                continue
            t25, t75 = np.percentile(star_ages_in_cluster, [25, 75])
            t25_t75 = t75 - t25
            if t25_t75 < threshold_fraction and len(star_ages_in_cluster) > 20:
            
                metrics = calculate_cluster_metrics(star_data, member_ids, cluster_idx, snap_idx, cluster_start_snapshot, cluster_end_snapshot, snapshot_conversion_factor, is_burst=True)
                all_metrics.append(metrics)
                print(f"Found burst for cluster {cluster_idx} at snapshot {snap_idx}. Recording metrics.")
                logging.info(f"Found burst for cluster {cluster_idx} at snapshot {snap_idx}. Recording metrics.") 
                break
        else:
            # If no burst was found, record metrics at the start snapshot
            print(f"No burst found for cluster {cluster_idx} in this run. Recording metrics at start snapshot {cluster_start_snapshot}.")
            logging.info(f"No burst found for cluster {cluster_idx} in this run. Recording metrics at start snapshot {cluster_start_snapshot}.")
            snap_idx = cluster_start_snapshot
            star_data = all_star_data.get(snap_idx)
            member_ids = get_member_ids_for_fuzzycat_cluster(cluster_idx, ordering, fuzzy_clusters, working_directory_path, snap_idx)

            if len(member_ids) == 0:
                continue

            metrics = calculate_cluster_metrics(star_data, member_ids, cluster_idx, snap_idx, cluster_start_snapshot, cluster_end_snapshot, snapshot_conversion_factor, is_burst=False)
            all_metrics.append(metrics)
        
        df_row = pd.DataFrame([metrics])
        df_row.to_csv(csv_filepath, mode='a', header=False, index=False)

    # Save all metrics to a numpy file
    all_cluster_metrics = pd.read_csv(csv_filepath).set_index('cluster_idx').to_dict(orient='index')
    np.save(f"{analysis_dir}all_cluster_metrics.npy", all_cluster_metrics, allow_pickle=True)
    logging.info(f"All cluster metrics saved to {analysis_dir}all_cluster_metrics.npy")

def get_cluster_masses(snapshot_file_paths, working_directory_path, analysis_dir, star_data_dir, n_snapshots, n_clusters, ordering, fuzzy_clusters, clustered_ids, snapshots):
    """
    Load and save the masses of all clusters in a numpy file.
    This function calculates the total mass of each cluster at each snapshot and saves it in a dictionary and numpy file.
    """
    cluster_metrics = np.load(f"{analysis_dir}all_cluster_metrics.npy", allow_pickle=True).item()

    cluster_masses = {cluster_idx: {} for cluster_idx in cluster_metrics}
    snapshot_to_clusters = defaultdict(list)
    for cluster_idx, metrics in cluster_metrics.items():
        for snap_idx in range(metrics['cluster_start_snapshot'], metrics['cluster_end_snapshot'] + 1):
            if snap_idx < 0 or snap_idx >= n_snapshots or snap_idx not in snapshots:
                continue
            snapshot_to_clusters[snap_idx].append(cluster_idx)

    for snap_idx in snapshots:
        print(f"Processing cluster masses for snapshot {snap_idx}/{n_snapshots}")
        logging.info(f"Processing cluster masses for snapshot {snap_idx}/{n_snapshots}")
        star_data = np.load(f"{star_data_dir}star_data_{snap_idx:03d}.npy")
        star_mass_map = {int(row[0]): row[2] for row in star_data}

        for cluster_idx in snapshot_to_clusters[snap_idx]:
            member_ids = get_member_ids_for_fuzzycat_cluster(cluster_idx, ordering, fuzzy_clusters, working_directory_path, snap_idx)
            if len(member_ids) == 0:
                continue

            star_masses_in_cluster = [star_mass_map[star_id] for star_id in member_ids if star_id in star_mass_map]
            cluster_masses[cluster_idx][snap_idx] = np.array(star_masses_in_cluster)

    np.save(f"{analysis_dir}cluster_masses.npy", cluster_masses, allow_pickle=True)
    logging.info(f"Cluster masses saved to {analysis_dir}cluster_masses.npy")

def save_star_data(snapshot_file_paths, star_data_dir):
    """Load and save all star data for each snapshot from simulation (ids, ages, masses, positions, chemical abundances)"""

    print("Saving star data for each snapshot...")
    logging.info("Saving star data for each snapshot...")
    n_snapshots = len(snapshot_file_paths)

    #save star ages for faster access
    for snap_idx, snapshot_path in enumerate(snapshot_file_paths):
        if os.path.exists(f"{star_data_dir}star_data_{snap_idx:03d}.npy"):
            continue
        snapshot_name = os.path.basename(snapshot_path)
        print(f"Processing snapshot {snap_idx+1}/{n_snapshots}: {snapshot_name}")
        logging.info(f"Processing snapshot {snap_idx+1}/{n_snapshots}: {snapshot_name}")
        
        # Load the simulation data
        simulation = pb.load(snapshot_path)
        
        # Load the main halo and make it face-on
        main_halo = simulation.halos()[0]
        pb.analysis.angmom.faceon(main_halo)
        main_halo.physical_units()

        
        # Get particle data
        star_particles = main_halo.stars
        star_ids = star_particles['iord']
        star_ages = star_particles['age']
        star_masses = star_particles['mass']
        star_pos = star_particles['pos']
        star_iron = star_particles['FeMassFrac'] 
        star_oxygen = star_particles['OxMassFrac']

        # Create an array with the above parameters

        star_ages = star_ages.in_units('Gyr')  # Convert ages to Gyr
        star_masses = star_masses.in_units('Msol')  # Convert masses to Msun
        stars = np.column_stack((star_ids, star_ages, star_masses, star_pos, star_iron, star_oxygen))

        # Save ages for this snapshot
        np.save(f"{star_data_dir}star_data_{snap_idx:03d}.npy", stars)

def burst_cluster_analysis(analysis_dir, burst_cluster_analysis_dir, working_directory_path, n_snapshots, ordering, fuzzy_clusters, star_data_dir, cluster_metrics, cluster_masses, snapshots, snapshot_conversion_factor):
    """Analyze all burst clusters and make plots"""


    cluster_metrics = {k: v for k, v in cluster_metrics.items() if v['burst_cluster']}
    cluster_masses = {k: v for k, v in cluster_masses.items() if k in cluster_metrics}

    logging.info(f"Starting analysis for burst clusters... Found {len(cluster_metrics)} burst clusters.")

    # #------Cluster Mass function-----
    power_law_analysis_dir = f"{burst_cluster_analysis_dir}power_law_analysis/"
    os.makedirs(power_law_analysis_dir, exist_ok=True)
    power_law_analysis_few_clusters(power_law_analysis_dir, n_snapshots, cluster_masses)

    # #------Lifetime distribution of clusters-----
    lifetime_analysis_dir = f"{burst_cluster_analysis_dir}lifetime_analysis/"
    os.makedirs(lifetime_analysis_dir, exist_ok=True)
    lifetime_analysis(cluster_metrics, cluster_masses, lifetime_analysis_dir, star_analysis=True)
    
    # #------Position distribution of clusters-----
    spacial_distribution_analysis_dir = f"{burst_cluster_analysis_dir}spacial_distribution_analysis/"
    os.makedirs(spacial_distribution_analysis_dir, exist_ok=True)
    spacial_distribution_analysis(cluster_metrics, spacial_distribution_analysis_dir, star_forming=True)

    #------Metallicity distribution of clusters-----
    metallicity_distribution_analysis_dir = f"{burst_cluster_analysis_dir}metallicity_distribution_analysis/"
    os.makedirs(metallicity_distribution_analysis_dir, exist_ok=True)
    metallicity_distribution_analysis(cluster_metrics, metallicity_distribution_analysis_dir, star_analysis=True)

    #------Contamination and Loss Analysis of Burst Clusters-----
    contamination_analysis_dir = f"{burst_cluster_analysis_dir}contamination_analysis/"
    os.makedirs(contamination_analysis_dir, exist_ok=True)
    contamination_analysis(cluster_metrics, cluster_masses, contamination_analysis_dir, working_directory_path, ordering, fuzzy_clusters, star_data_dir, snapshots, snapshot_conversion_factor)
    


def fuzzy_cluster_analysis(analysis_dir, working_directory_path, n_snapshots, fuzzy_cluster_analysis_dir, star_data_dir, ordering, fuzzy_clusters, cluster_metrics, cluster_masses, snapshots, snapshot_conversion_factor):
    """Analyze all fuzzy clusters and make plots"""

    logging.info("Starting analysis for all fuzzy clusters...")

    # #------Cluster Mass function-----
    power_law_analysis_dir = f"{fuzzy_cluster_analysis_dir}power_law_analysis/"
    os.makedirs(power_law_analysis_dir, exist_ok=True)
    power_law_analysis_many_clusters(power_law_analysis_dir, snapshots, cluster_masses)

    # #------Lifetime distribution of clusters-----
    lifetime_analysis_dir = f"{fuzzy_cluster_analysis_dir}lifetime_analysis/"
    os.makedirs(lifetime_analysis_dir, exist_ok=True)
    lifetime_analysis(cluster_metrics, cluster_masses, lifetime_analysis_dir)

    # #------Position distribution of clusters-----
    spacial_distribution_analysis_dir = f"{fuzzy_cluster_analysis_dir}spacial_distribution_analysis/"
    os.makedirs(spacial_distribution_analysis_dir, exist_ok=True)
    spacial_distribution_analysis(cluster_metrics, spacial_distribution_analysis_dir)
    
    #------Metallicity distribution of clusters-----
    metallicity_distribution_analysis_dir = f"{fuzzy_cluster_analysis_dir}metallicity_distribution_analysis/"
    os.makedirs(metallicity_distribution_analysis_dir, exist_ok=True)
    metallicity_distribution_analysis(cluster_metrics, metallicity_distribution_analysis_dir)

    
def metallicity_distribution_analysis(cluster_data, dir, star_analysis=False):
    """
    Analyze the metallicity distribution of clusters.
    This function takes the mean metallicity of the clusters given and plots one plot for iron and one for oxygen.
    """
    # Extract metallicity data
    iron_metallicities = [data['median_iron'] for data in cluster_data.values()]
    oxygen_metallicities = [data['median_oxygen'] for data in cluster_data.values()]
    
    # Plot the metallicity distributions
    #sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((1, 6))
    
    # --- Plot 1: Iron Metallicity Distribution ---
    median_iron = np.median(iron_metallicities)
    ax1 = axes[0]
    if star_analysis:
        sns.histplot(data = iron_metallicities, ax=ax1, color='royalblue', bins=30)
        ax1.xaxis.set_major_formatter(formatter)
        ax1.xaxis.offsetText.set_fontsize(14)
    else: 
        sns.histplot(data = iron_metallicities, ax=ax1, color='royalblue', bins=30, log_scale=True)
        ax1.set_xscale('log')

    ax1.axvline(median_iron, color='red', linestyle='--', linewidth=2, label=f'Median: {median_iron:.2e}')
    #ax1.set_title("Iron Metallicity Distribution of Clusters")
    ax1.set_xlabel("Mean Iron Mass Fraction", fontsize=18)
    ax1.set_ylabel("Cluster Count", fontsize=18)
    ax1.legend(fontsize=14)
    ax1.grid(True, which="both", linestyle='--', alpha=0.7)


    # --- Plot 2: Oxygen Metallicity Distribution ---
    median_oxygen = np.median(oxygen_metallicities)
    ax2 = axes[1]
    if star_analysis:
        sns.histplot(data=oxygen_metallicities, ax=ax2, color='seagreen', bins=30)
        ax2.xaxis.set_major_formatter(formatter)
        ax2.xaxis.offsetText.set_fontsize(14)
    else:
        sns.histplot(data=oxygen_metallicities, ax=ax2, color='seagreen', bins=30, log_scale=True)
        ax2.set_xscale('log')
    ax2.axvline(median_oxygen, color='red', linestyle='--', linewidth=2, label=f'Median: {median_oxygen:.2e}')
    #ax2.set_title("Oxygen Metallicity Distribution of Clusters")
    ax2.set_xlabel("Mean Oxygen Mass Fraction", fontsize=18)
    ax2.set_ylabel("Cluster Count", fontsize=18)
    ax2.legend(fontsize=14)
    ax2.grid(True, which="both", linestyle='--', alpha=0.7)


    plt.setp(ax1.get_xticklabels(), fontsize=12)
    plt.setp(ax1.get_yticklabels(), fontsize=12)
    plt.setp(ax2.get_xticklabels(), fontsize=12)
    plt.setp(ax2.get_yticklabels(), fontsize=12)

    # Add a main title for the entire figure
    #fig.suptitle("Metallicity Distribution of Star Clusters", fontsize=20, y=1.02)
    # Adjust layout to prevent titles/labels from overlapping
    plt.tight_layout()
    plt.savefig(f"{dir}cluster_metallicity_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster metallicity distribution plot to {dir}cluster_metallicity_distribution.png")

def contamination_analysis(burst_clusters, cluster_masses, contamination_analysis_dir, workingDirectoryPath, ordering, fuzzy_clusters, star_data_dir, snapshots, snapshot_conversion_factor):
    """Analyze the contamination and loss of burst clusters over time. Calculate both contamination and loss fractions for each cluster at each snapshot."""
    logging.info("Starting contamination analysis for burst clusters...")
    birth_age_range = .100  # in Gyr, the age range for a star to be considered native to the cluster's formation events
    if not os.path.exists(f"{contamination_analysis_dir}burst_clusters_contamination_data.npy"):
        contamination_data = {
            cluster_idx: {
                'snapshots': [],
                'contamination_fractions': [],
                'loss_fractions': [],
            }
            for cluster_idx in burst_clusters.keys()
        }
        native_member_ids_per_cluster = defaultdict(set)

        for snap_idx in snapshots:
            print(f"Processing snapshot {snap_idx} for contamination analysis...")
            logging.info(f"Processing snapshot {snap_idx} for contamination analysis...")
            # Load star data for the current snapshot
            star_data = np.load(f"{star_data_dir}star_data_{snap_idx:03d}.npy")

            star_id_to_age_map = {row[0]: row[1] for row in star_data}

            for cluster_idx, metrics in burst_clusters.items():
                start_snapshot = metrics['cluster_start_snapshot']
                end_snapshot = metrics['cluster_end_snapshot']
                if not (start_snapshot <= snap_idx <= end_snapshot):
                    continue

                native_member_ids = native_member_ids_per_cluster[cluster_idx]
                current_member_ids_list = get_member_ids_for_fuzzycat_cluster(
                    cluster_idx, ordering, fuzzy_clusters, workingDirectoryPath, snap_idx
                )
                current_member_ids = set(current_member_ids_list)

                for star_id in current_member_ids:
                    if star_id in native_member_ids:
                        continue  # Skip if already counted as native
                    
                    if star_id_to_age_map.get(star_id, float('inf')) <= birth_age_range:
                        native_member_ids.add(star_id)

                if not current_member_ids:
                    contamination_fraction = np.nan
                    loss_fraction = np.nan
                else:
                    intersection_count = len(native_member_ids.intersection(current_member_ids))
                    loss_fraction = 1 - intersection_count / len(native_member_ids) if native_member_ids else 0
                    contamination_fraction = 1 - intersection_count / len(current_member_ids)

                if contamination_fraction == 1.0 and loss_fraction == 0.0:
                    print(f"Cluster {cluster_idx} formed before simulation start, no in-situ information available")
                    logging.info(f"Cluster {cluster_idx} formed before simulation start, no in-situ information available")
                    contamination_fraction = np.nan
                    loss_fraction = np.nan

                # Append the calculated fractions to the lists
                contamination_data[cluster_idx]['snapshots'].append(snap_idx)
                contamination_data[cluster_idx]['contamination_fractions'].append(contamination_fraction)
                contamination_data[cluster_idx]['loss_fractions'].append(loss_fraction)

        # Save the final data structure to a file
        print("\nSaving contamination data...")
        np.save(f"{contamination_analysis_dir}burst_clusters_contamination_data.npy", contamination_data, allow_pickle=True)
        logging.info(f"Contamination data saved to {contamination_analysis_dir}burst_clusters_contamination_data.npy")

    
    contamination_data = np.load(f"{contamination_analysis_dir}burst_clusters_contamination_data.npy", allow_pickle=True).item()
    dir = f"{contamination_analysis_dir}contamination/"
    os.makedirs(dir, exist_ok=True)

    # Plot loss vs contamination fractions for all clusters in their last snapshot
    plt.figure(figsize=(8, 8))
    all_contamination_fractions = []
    all_loss_fractions = []
    for cluster_idx, data in contamination_data.items():
        snapshots = np.array(data['snapshots'])
        contamination_fractions = np.array(data['contamination_fractions'])
        loss_fractions = np.array(data['loss_fractions'])
        
        # Append the last contamination and loss fractions
        all_contamination_fractions.append(contamination_fractions[-1])
        all_loss_fractions.append(loss_fractions[-1])

    all_contamination_fractions = np.array(all_contamination_fractions)
    all_loss_fractions = np.array(all_loss_fractions)
    plot_cont = all_contamination_fractions.copy().astype(float)
    plot_loss = all_loss_fractions.copy().astype(float)
    
    plt.scatter(plot_loss,plot_cont, alpha=0.8, s=70)
    plt.title("Contamination vs Loss Fractions at Last Snapshot of Each Cluster")
    plt.xlabel("Loss Fraction")
    plt.ylabel("Contamination Fraction")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = f"{contamination_analysis_dir}contamination_vs_loss_fractions_last_snapshot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved contamination vs loss fractions plot to {output_path}")


    # Make seperate histograms for contamination and loss fractions for all values in the last 230 Myr of the simulation
    plt.figure(figsize=(12, 6))
    age_threshold = 230  # in Myr, the age threshold for the last 230 Myr of the simulation
    # Convert age threshold from Myr to snapshots (for our simulation)
    age_threshold_snapshots = int(age_threshold * 1/snapshot_conversion_factor) 
    all_contamination_fractions = []
    all_loss_fractions = []
    for snap_idx in range(200 - age_threshold_snapshots, 201):
        for cluster_idx, data in contamination_data.items():
            if snap_idx not in data['snapshots']:
                continue

            idx = data['snapshots'].index(snap_idx)
            all_contamination_fractions.append(data['contamination_fractions'][idx])
            all_loss_fractions.append(data['loss_fractions'][idx])
    all_contamination_fractions = np.array(all_contamination_fractions)
    all_loss_fractions = np.array(all_loss_fractions)
    no_info_mask = (all_loss_fractions == 1) & (all_contamination_fractions == 0)
    plot_cont = all_contamination_fractions.copy().astype(float)
    plot_loss = all_loss_fractions.copy().astype(float)
    # plot_cont[no_info_mask] = np.nan  # Set no info points to NaN for better plotting
    # plot_loss[no_info_mask] = np.nan
    
    # Plot histograms for contamination and loss fractions
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(plot_cont, binwidth=0.05, ax=ax[0], color='royalblue', kde=True, binrange=(0, 1))
    sns.histplot(plot_loss, binwidth=0.05, ax=ax[1], color='seagreen', kde=True, binrange=(0, 1))
    #ax[0].set_title("Distribution of Contamination Fractions in Last 230 Myr")
    ax[0].set_xlabel("Contamination Fraction", fontsize=18)
    ax[0].set_ylabel("Count", fontsize=18)
    #ax[1].set_title("Distribution of Loss Fractions in Last 230 Myr")
    ax[1].set_xlabel("Loss Fraction", fontsize=18)
    ax[1].set_ylabel("Count", fontsize=18)
    ax[0].set_xlim(0, 1)
    ax[1].set_xlim(0, 1)
    ax[0].tick_params(axis='x', labelsize=14)
    ax[0].tick_params(axis='y', labelsize=14)
    ax[1].tick_params(axis='x', labelsize=14)
    ax[1].tick_params(axis='y', labelsize=14)

    ax[0].grid(True, linestyle='--', alpha=0.7)
    ax[1].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = f"{contamination_analysis_dir}contamination_loss_distribution_last_230_Myr_2.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved contamination/loss distribution plot to {output_path}")

def lifetime_analysis(clusters, masses, dir, star_analysis = False):
    """
    Analyze the observed lifetime of burst clusters.
    """
    # Initialize a list to store cluster lifetimes
    cluster_lifetimes = []
    for cluster_idx, metrics in clusters.items():
        # Extract the start and end snapshots for the cluster
        start_snapshot = metrics['cluster_start_snapshot']
        end_snapshot = metrics['cluster_end_snapshot']
        
        # Calculate the lifetime in snapshots
        lifetime = end_snapshot - start_snapshot
        
        # Store the cluster index and its lifetime
        cluster_lifetimes.append((cluster_idx, lifetime))
            
    # Convert to a DataFrame for easier analysis
    lifetime_df = pd.DataFrame(cluster_lifetimes, columns=['Cluster Index', 'Lifetime (Snapshots)'])
    lifetime_df['Lifetime (Gyr)'] = lifetime_df['Lifetime (Snapshots)'] * 13.8/2000
    lifetime_df.set_index('Cluster Index', inplace=True)
    
    # Save the lifetime data to a CSV file
    lifetime_df.to_csv(f"{dir}cluster_lifetimes.csv")
    print(f"Saved cluster lifetimes to {dir}cluster_lifetimes.csv")

    median_lifetime = lifetime_df['Lifetime (Gyr)'].median()

    # Plot the distribution of cluster lifetimes
    plt.figure(figsize=(10, 6))
    sns.histplot(lifetime_df['Lifetime (Gyr)'], bins=30, color='plum', linewidth=0.5)
    plt.axvline(median_lifetime, color='red', linestyle='--', linewidth=2, label=f'Median Lifetime: {median_lifetime:.2f} Gyr')
    plt.title("Distribution of detected Cluster Lifetimes")
    plt.xlabel("Detected Lifetime (Gyr)", fontsize=18)
    plt.ylabel("Number of Clusters", fontsize=18)
    plt.xticks(np.arange(0.2, lifetime_df['Lifetime (Gyr)'].max() + 0.1, 0.1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{dir}cluster_lifetime_distribution.png", dpi=300)
    plt.close()
    print(f"Saved cluster lifetime distribution plot to {dir}cluster_lifetime_distribution.png")


    if star_analysis:
        #combine with spacial data
        clusters_spatial_data = {
            idx: {
                'median_radial_distance': data['median_radial_distance'],
                'median_vertical_distance': data['median_z_distance']
            } for idx, data in clusters.items()
        }
        # Convert to DataFrame for easier manipulation
        spatial_df = pd.DataFrame.from_dict(clusters_spatial_data, orient='index')
        spatial_df.index.name = 'Cluster Index'

        cluster_ids = []
        median_masses = []
        total_masses = []

        for cid, snaps in masses.items():
            # Use generator + np.concatenate for performance
            mass_arrays = [np.asarray(snap) for snap in snaps.values() if len(snap) > 0]
            if mass_arrays:
                all_masses = np.concatenate(mass_arrays)
                cluster_ids.append(cid)
                median_masses.append(np.median(all_masses))
                total_masses.append(np.sum(all_masses))

        mass_df = pd.DataFrame({
            "cluster_idx": cluster_ids,
            "median_mass": median_masses,
            "total_mass": total_masses
        })
        mass_df.set_index('cluster_idx', inplace=True)
        
        # Ensure the index matches the spatial DataFrame
        mass_df.index.name = 'Cluster Index'

        # Merge the spatial data with the mass data
        spatial_df = spatial_df.merge(mass_df, left_index=True, right_index=True)


        lifetimes = pd.read_csv(f"{dir}cluster_lifetimes.csv", index_col=0)
        # Merge the two DataFrames on the index (Cluster Index)
        combined_df = spatial_df.merge(lifetimes, left_index=True, right_index=True)

        median_radial_distance = np.median(combined_df['median_radial_distance'])
        max_radial_distance = np.max(combined_df['median_radial_distance'])
        median_vertical_distance = np.median(combined_df['median_vertical_distance'])
        max_vertical_distance = np.max(combined_df['median_vertical_distance'])


        y_ticks_linear = [0,1,2]  # in linear region
        y_ticks_log = [10, 20, 30, 40, 50,60,70]  # for the log region
        y_all_ticks = y_ticks_linear + y_ticks_log

        x_ticks_linear = list(range(0,51, 5))  # in linear region
        x_ticks_log = [100, 150, 200,250, 300]
        x_all_ticks = x_ticks_linear + x_ticks_log

        #plot the spatial distribution of cluster lifetimes
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=combined_df, x='median_radial_distance', y='median_vertical_distance', hue='Lifetime (Gyr)',  palette='viridis', size='Lifetime (Gyr)', sizes=(20, 200), legend='brief')
        #plt.title("Spatial Distribution of Cluster Lifetimes")
        plt.xlabel("Median Radial Distance (kpc)", fontsize=18)
        plt.ylabel("Median Vertical Distance |z| (kpc)", fontsize=18)
        plt.yscale('symlog', linthresh=2)
        plt.xscale('symlog', linthresh=50)
        plt.yticks(y_all_ticks, [str(tick) for tick in y_ticks_linear] + [str(tick) for tick in y_ticks_log], fontsize=12)
        plt.xticks(x_all_ticks, [str(tick) for tick in x_ticks_linear] + [str(tick) for tick in x_ticks_log], fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left', framealpha=0.5, title='Lifetime (Gyr)\nbrighter/larger = longer', fontsize=12, title_fontsize=14)
        plt.tight_layout()
        plt.xlim(0)
        plt.margins(x=0.01, y=0.01)  # Add a small margin to avoid cutting off points
        plt.savefig(f"{dir}spatial_distribution_cluster_lifetimes.png", dpi=300)
        plt.close()
        print(f"Saved spatial distribution of cluster lifetimes plot to {dir}spatial_distribution_cluster_lifetimes.png")


def get_member_ids_for_fuzzycat_cluster(cluster_idx, ordering, fuzzy_clusters, workingDirectoryPath, snap_idx):
    """
    Get the member ids for a given fuzzycat cluster index at a specific snapshot.
    """
    clusterFileNames = np.load(f"{workingDirectoryPath}clusterFileNames.npy")
    fuzzy_start_idx, fuzzy_end_idx = fuzzy_clusters[cluster_idx]
    astrolink_cluster_ids_in_fuzzycat_cluster = ordering[fuzzy_start_idx:fuzzy_end_idx]
    
    # Get AstroLink clusters for this snapshot that belong to the current FuzzyCat cluster
    cluster_filenames = [clusterFileNames[idx] for idx in astrolink_cluster_ids_in_fuzzycat_cluster if clusterFileNames[idx].startswith(f"{snap_idx:03d}")]
    
    all_clusters = []
    for cluster_filename in cluster_filenames:
        cluster = np.load(f"{workingDirectoryPath}Clusters_iord/{cluster_filename}")
        all_clusters.append(cluster)
        
    # If we found any clusters, combine their particle data
    if all_clusters:
        member_ids = np.concatenate(all_clusters)
        member_ids = np.unique(member_ids)
    else:
        member_ids = np.array([], dtype=int)
            
    return member_ids


def spacial_distribution_analysis(cluster_data, dir, star_forming=False):
    """
    Analyzes the spatial distribution of burst clusters in the simulation.
    At what radius from the center of the disk do the clusters form?
    How far out of the disk vertically are they?
    """
    # Plot the spatial distribution of clusters
    radial_distances = [data['median_radial_distance'] for data in cluster_data.values()]
    vertical_distances = [data['median_z_distance'] for data in cluster_data.values()]



    # Create a figure with two subplots, side by side
    # figsize controls the overall size of the figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ticks_z_linear = list(range(0,4))  # in linear region (you can adjust)
    ticks_z_log = [10,20,30]  # for the log region
    all_z_ticks = ticks_z_linear + ticks_z_log


    ax1 = axes[0]
    ax2 = axes[1]

    if star_forming:
        bins_lin = np.arange(0, 2.25, 0.25)
        ticks_z_linear = list(bins_lin)  # in linear region (you can adjust)
        bins_log = [10, 20, 30, 40, 50, 60, 70, 80, 90,100]  # for the log region
        ticks_z_log = [int(tick) for tick in bins_log]  # convert
        all_z_ticks = ticks_z_linear + ticks_z_log
        #all_bins = np.concatenate((bins_lin, bins_log))
        ax2.set_xscale('symlog', linthresh=2)
        ax2.set_xticks(all_z_ticks, [str(tick) for tick in ticks_z_linear] + [str(tick) for tick in ticks_z_log])

        ticks_r_linear = list(range(0,31, 5))  # in linear region (you can adjust)
        ticks_r_log = [100,200,300]  # for the log region
        all_r_ticks = ticks_r_linear + ticks_r_log
        ax1.set_xscale('symlog', linthresh=30)
        ax1.set_xticks(all_r_ticks, [str(tick) for tick in ticks_r_linear] + [str(tick) for tick in ticks_r_log])

    # --- Plot 1: Radial Distribution ---

    if star_forming:
        sns.histplot(data=radial_distances, ax=ax1, color='royalblue', binwidth=2)
    else:
        sns.histplot(data=radial_distances, ax=ax1, color='royalblue', binwidth=10)
    ax1.set_title("Radial Distribution of Clusters")
    ax1.set_xlabel("Median Radial Distance (kpc)")
    ax1.set_ylabel("Cluster Count")

    # Add a vertical line for the median of the distribution
    median_r = np.median(radial_distances)
    ax1.axvline(median_r, color='red', linestyle='--', linewidth=2, label=f'Median: {median_r:.2f} kpc')
    ax1.legend(fontsize=14)
    ax1.set_xlim(0, 300)  # Set x-axis limit


    # --- Plot 2: Vertical Distribution ---

    if star_forming:
        sns.histplot(data=vertical_distances, ax=ax2, color='seagreen', binwidth=0.25)
    else:
        sns.histplot(data=vertical_distances, ax=ax2, color='seagreen', binwidth=2.5)
    #ax2.set_title("Vertical Distribution of Clusters")
    ax2.set_xlabel("Median Vertical Distance |z| (kpc)", fontsize=18)
    ax2.set_ylabel("Cluster Count", fontsize=18)
    ax2.margins(0.0, 0.05)
    ax2.grid(True)
    ax2.set_xlim(0, 70)  # Set x-axis limit

    # Add a vertical line for the median of the distribution
    median_z = np.median(vertical_distances)
    ax2.axvline(median_z, color='red', linestyle='--', linewidth=2, label=f'Median: {median_z:.2f} kpc')
    ax2.legend()

    # Add a main title for the entire figure
    #fig.suptitle("Spatial Distribution of Star Clusters in the Galactic Disk", fontsize=20, y=1.02)

    # Adjust layout to prevent titles/labels from overlapping
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(f"{dir}cluster_spatial_distribution_count.png", dpi=300, bbox_inches='tight')


def power_law_analysis_few_clusters(power_law_analysis_dir, n_snapshots, cluster_masses):
    """
    Analyses the power law slope of the burst clusters (few clusters).
    We get the masses for all clusters in their last detected snapshot and bin using the freedman-diaconis rule. 
    Then we perform a linear regression on the histogram data in log-log space to find the slope and intercept.
    """

    all_snapshot_masses = cluster_masses
    nonempty_clusters = [c for c in all_snapshot_masses if any(len(m) > 0 for m in all_snapshot_masses[c].values())]
    individual_plots_dir = f"{power_law_analysis_dir}individual_plots/"
    os.makedirs(individual_plots_dir, exist_ok=True)

    #---Use this for analysis of all snapshots (few total clusters)---
    peak_cluster_masses = []
    for cluster_idx in nonempty_clusters:
        # Get all mass measurements for this one cluster across all snapshots it exists in
        masses_over_time = all_snapshot_masses[cluster_idx].values()
        
        # Calculate the total cluster mass at each of those snapshots
        total_masses_per_snap = [np.sum(m) for m in masses_over_time if m.size > 0]
        
        # If the cluster had any mass at any point, find its maximum mass
        if total_masses_per_snap:
            peak_mass = np.max(total_masses_per_snap)
            median_mass = np.median(total_masses_per_snap)
            peak_cluster_masses.append(median_mass)

    # This is now our primary dataset for the CMF analysis
    cluster_masses = np.array(peak_cluster_masses)
    print(f"Found a total of {len(cluster_masses)} unique clusters across all snapshots.")

    if len(cluster_masses) < 10:
            #print(f"Not enough clusters found for snapshot {final_snapshot_idx}. Skipping power law analysis.")
            print("Not enough clusters found for power law analysis. Skipping.")
    else:   
        log_masses = np.log10(cluster_masses)  # Convert to log scale for power law fitting
        q75, q25 = np.percentile(log_masses, [75, 25])
        iqr = q75 - q25
        # Use IQR to determine bin_width
        if iqr > 0:
            bin_width = 2*iqr*(len(log_masses)**(-1/3))  # Freedman-Diaconis rule
            num_bins = int(np.ceil((log_masses.max() - log_masses.min()) / bin_width))
        else:
            num_bins = 10  # Fallback if IQR is zero
        num_bins = max(5, min(num_bins, 100))  # Ensure num_bins is between 5 and 100
        #num_bins = 10  # Ensure num_bins is between 5 and 50
        min_logM = np.log10(cluster_masses.min())
        max_logM = np.log10(cluster_masses.max())
        log_bins = np.linspace(min_logM, max_logM, num_bins + 1)  # Create bins in log scale
        linear_bins = 10**log_bins  # Convert back to linear scale for histogramming
        original_dN, _ = np.histogram(cluster_masses, bins=linear_bins)

        #----
        log_bin_centers = 0.5 * (log_bins[:-1] + log_bins[1:])  # Centers of the log bins
        dM = linear_bins[1:] - linear_bins[:-1]  # Width of the bins in linear scale
        dM[dM==0] = 1e-9
        dN_dM = original_dN / dM  # Number of clusters per bin width
        mask = (10**log_bin_centers >= 1e4) & (original_dN > 0)  # Filter for Mcl ≥ 1e4 and dN > 0
        x_fit_data = log_bin_centers[mask]
        y_fit_data = np.log10(dN_dM[mask])
        if x_fit_data.size > 1:
            # Perform linear regression to find the slope and intercept
            res = stats.linregress(x_fit_data, y_fit_data)
            slope, y_intercept, slope_error, y_intercept_error, r_sqr = res.slope, res.intercept, res.stderr, res.intercept_stderr, res.rvalue**2

    
            # Plotting the cluster mass function
            x_fit_line = np.linspace(min_logM, max_logM, 100)  # Fit line for plotting
            y_fit_line = slope * x_fit_line + y_intercept  # Linear fit line in log-log space
            y_err_propagated = np.sqrt((x_fit_line * slope_error)**2 + y_intercept_error**2)
            y_upper = y_fit_line + y_err_propagated
            y_lower = y_fit_line - y_err_propagated
            plt.figure(figsize=(10, 7))
            plt.fill_between(10**x_fit_line, 10**y_lower, 10**y_upper, color='red', alpha=0.2, label='1-$\sigma$ Uncertainty')
            plt.scatter(10**log_bin_centers[mask], dN_dM[mask], color='blue', label='Simulation Data', s=10)
            plt.plot(10**x_fit_line, 10**y_fit_line, 'r--', color='red', label=f"Fit: Slope = {slope:.2f} ± {slope_error:.2f}, R² = {r_sqr:.2f}")
            plt.grid(True, linestyle='--', alpha=0.7, axis='both')
            plt.legend()    
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Cluster Mass (M$_\odot$)", fontsize=18)
            plt.ylabel("Number of Clusters (N) per delta log(M)", fontsize=18)
            #plt.title("Cluster Mass Function")
            plt.tight_layout()
            #plt.savefig(f"{individual_plots_dir}cluster_mass_function_snapshot{final_snapshot_idx}.png", dpi=200)
            plt.savefig(f"{individual_plots_dir}cluster_mass_function_overall.png", dpi=200)
            plt.close()

        
def power_law_analysis_many_clusters(power_law_analysis_dir, snapshots, cluster_masses):
    
    """
    Analyses the power law slope of all fuzzy clusters.
    We initially bin the cluster masses for every snapshot using the freedman-diaconis rule. 
    Then we perform a linear regression on the histogram data in log-log space to find the slope and intercept.
    The analysis is done for each snapshot separately, and we store the results in lists.
    We then plot the resulting slope and error for each snapshot.
    """

    all_snapshot_masses = cluster_masses
    nonempty_clusters = [c for c in all_snapshot_masses if any(len(m) > 0 for m in all_snapshot_masses[c].values())]
    results_snapshots = []
    results_mean_slopes = []
    results_slope_errors = []
    make_individual_plots = True  # Set to False to skip individual plots and only save overview over slopes
    individual_plots_dir = f"{power_law_analysis_dir}individual_plots/"
    os.makedirs(individual_plots_dir, exist_ok=True)


    #---Use this for analysis of the final snapshot only (many total clusters)---

    for snapshot_idx in snapshots:
        cluster_masses_at_final_snap = []
        #---- Mass power law analysis ---
        for cluster_idx in nonempty_clusters:
            snap_masses = all_snapshot_masses[cluster_idx].get(snapshot_idx, np.array([]))
            if snap_masses.size > 0:
                Mcl = np.sum(snap_masses)
                cluster_masses_at_final_snap.append(Mcl)
        cluster_masses = np.array(cluster_masses_at_final_snap)
        print(f"Found a total of {len(cluster_masses)} clusters for initial mass function at snapshot {snapshot_idx}.")
        if len(cluster_masses) < 10:
                #print(f"Not enough clusters found for snapshot {final_snapshot_idx}. Skipping power law analysis.")
                print("Not enough clusters found for power law analysis. Skipping.")
        else:   
            log_masses = np.log10(cluster_masses)  # Convert to log scale for power law fitting
            q75, q25 = np.percentile(log_masses, [75, 25])
            iqr = q75 - q25
            # Use IQR to determine bin_width
            if iqr > 0:
                bin_width = 2*iqr*(len(log_masses)**(-1/3))  # Freedman-Diaconis rule
                num_bins = int(np.ceil((log_masses.max() - log_masses.min()) / bin_width))
            else:
                num_bins = 10  # Fallback if IQR is zero
            num_bins = max(2, min(num_bins, 100))  # Ensure num_bins is between 2 and 100


            min_logM = np.log10(cluster_masses.min())
            max_logM = np.log10(cluster_masses.max())
            log_bins = np.linspace(min_logM, max_logM, num_bins + 1)  # Create bins in log scale

            linear_bins = 10**log_bins  # Convert back to linear scale for histogramming
            original_dN, _ = np.histogram(cluster_masses, bins=linear_bins)

            #----
            log_bin_centers = 0.5 * (log_bins[:-1] + log_bins[1:])  # Centers of the log bins
            dM = linear_bins[1:] - linear_bins[:-1]  # Width of the bins in linear scale
            dM[dM==0] = 1e-9
            dN_dM = original_dN / dM  # Number of clusters per bin width
            mask = (10**log_bin_centers >= 1e4) & (original_dN > 0)  # Filter for Mcl ≥ 1e4 and dN > 0
            x_fit_data = log_bin_centers[mask]
            y_fit_data = np.log10(dN_dM[mask])
            if x_fit_data.size > 1:
                # Perform linear regression to find the slope and intercept
                res = stats.linregress(x_fit_data, y_fit_data)
                slope, y_intercept, slope_error, y_intercept_error, r_sqr = res.slope, res.intercept, res.stderr, res.intercept_stderr, res.rvalue**2

            
            results_snapshots.append(snapshot_idx)
            results_mean_slopes.append(slope)
            results_slope_errors.append(slope_error)
            if not make_individual_plots:
                continue
            # Plotting the cluster mass function for every snapshot
            x_fit_line = np.linspace(min_logM, max_logM, 100)  # Fit line for plotting
            y_fit_line = slope * x_fit_line + y_intercept  # Linear fit line in log-log space
            y_err_propagated = np.sqrt((x_fit_line * slope_error)**2 + y_intercept_error**2)
            y_upper = y_fit_line + y_err_propagated
            y_lower = y_fit_line - y_err_propagated
            plt.figure(figsize=(10, 7))
            plt.fill_between(10**x_fit_line, 10**y_lower, 10**y_upper, color='red', alpha=0.2, label='Uncertainty')
            plt.scatter(10**log_bin_centers[mask], dN_dM[mask], color='blue', label='Simulation Data', s=10)
            plt.plot(10**x_fit_line, 10**y_fit_line, 'r--', color='red', label=f"Fit: slope = {slope:.2f} ± {slope_error:.2f}, R² = {r_sqr:.2f}")
            plt.grid(True, linestyle='--', alpha=0.5, axis='both')
            plt.legend()    
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Cluster Mass (M$_\odot$)", fontsize=18)
            plt.ylabel("Number of Clusters per delta log(M)", fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            #plt.title("Cluster Mass Function")
            plt.savefig(f"{individual_plots_dir}cluster_mass_function_snapshot{snapshot_idx}.png", dpi=200)
            plt.close()
            print(f"Saved cluster mass function plot for snapshot {snapshot_idx} to {individual_plots_dir}cluster_mass_function_snapshot{snapshot_idx}.png")

        # Combine the lists into a single (N, 3) array
    output_data = np.vstack((results_snapshots, results_mean_slopes, results_slope_errors)).T
    # Define a header for the text file
    header = "Snapshot_Index  Mean_Slope_alpha  StdDev_Error"
    # Save using np.savetxt
    output_filename = f"{power_law_analysis_dir}slope_evolution.txt"
    np.savetxt(output_filename, output_data, header=header, fmt="%-15d %-18.4f %-18.4f")
    print(f"\nSaved slope evolution data to {output_filename}")

    # --- 1. Load the Data ---
    data_path = f"{power_law_analysis_dir}slope_evolution.txt"
    data = np.loadtxt(data_path)

    # Unpack the columns
    snapshots = data[:, 0]
    slopes = data[:, 1]
    errors = data[:, 2]

    fig, ax = plt.subplots(figsize=(12, 8))

    # --- Plot the main trend line ---
    # Use a solid line with small, subtle markers.
    ax.plot(snapshots, slopes, 
            marker='o',          # Small circle markers at each data point
            markersize=5,        # Control the size of the markers
            linestyle='-',       # A solid line connecting the points
            color='C0',          # Use the default blue color
            label='Slope (β)')

    # --- Plot the shaded error region ---
    # This is the key function: plt.fill_between
    ax.fill_between(snapshots,           # X-values
                    slopes - errors,     # The lower boundary of the shaded region
                    slopes + errors,     # The upper boundary of the shaded region
                    color='C0',          # Match the line color
                    alpha=0.2,           # Use transparency to make it subtle
                    label='Binning Uncertainty')

    # --- Add a reference line for the theoretical initial slope ---
    ax.axhline(-1.0, 
            color='orange', 
            linestyle='--', 
            linewidth=2,
            label='Theoretical Slope (β = -1.0)')

    ax.set_xlabel("Snapshot Index", fontsize=18)
    ax.set_ylabel("Mass Function Slope (β)", fontsize=18)
    #ax.set_title("Evolution of the Cluster Mass Function Slope", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(snapshots.min() - 1, snapshots.max() + 1)
    ax.set_ylim(-1.8, -0.9) # Adjust this based on your data's range
    ax.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{power_law_analysis_dir}slope_evolution_plot.png", dpi=300)


def star_forming_vs_non_star_forming_age_distribution(analysis_dir, cluster_data):

    """
    Analyze the median age distribution of star forming clusters during their burst snapshot.
    Compare that to the median age distribution of non-star forming clusters during their detection snapshot.
    """
    
    # Initialize a dictionary to store the median ages of star forming clusters
    median_ages_star_forming = []
    median_ages_non_star_forming = []
    radii_star_forming = []
    radii_non_star_forming = []
    # Loop through each cluster in the cluster data
    for cluster_idx, metrics in cluster_data.items():
        # Check if the cluster is a burst cluster
        if metrics['burst_cluster']:
            # If it's a burst cluster, use the burst snapshot for mean age analysis
            median_age = metrics['median_age']
            median_ages_star_forming.append(median_age)
            radii_star_forming.append(metrics['median_radial_distance'])
        else:
            # If it's not a burst cluster, use the detection snapshot for mean age analysis
            median_age = metrics['median_age']
            median_ages_non_star_forming.append(median_age)
            radii_non_star_forming.append(metrics['median_radial_distance'])
            
    # Convert the median ages to numpy arrays for easier manipulation
    median_ages_star_forming = np.array(median_ages_star_forming)
    median_ages_non_star_forming = np.array(median_ages_non_star_forming)
    radii_star_forming = np.array(radii_star_forming)
    radii_non_star_forming = np.array(radii_non_star_forming)


    # Define the overall age and radius range for consistent coloring
    all_ages = np.concatenate([median_ages_star_forming, median_ages_non_star_forming])
    all_radii = np.concatenate([radii_star_forming, radii_non_star_forming])
    age_min, age_max = all_ages.min(), all_ages.max()
    radius_min, radius_max = all_radii.min(), all_radii.max()

    # Plot radial distance vs median age for star forming clusters, markers based on star-forming and non-star-forming
    plt.figure(figsize=(12, 8))
    plt.scatter(median_ages_star_forming, radii_star_forming,
                c='blue', alpha=0.9, label='Star Forming Clusters', s=40, marker='x')
    plt.scatter(median_ages_non_star_forming, radii_non_star_forming,
                c='orange', alpha=0.9, label='Non-Star Forming Clusters', s=30, marker='o')
    plt.title("Radial Distance vs Median Age of Clusters")
    plt.xlabel("Median Age (Gyr)")
    plt.ylabel("Median Radial Distance (kpc)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}radial_distance_vs_median_age_star_vs_non_star_forming.png", dpi=300)
    plt.close()


