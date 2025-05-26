def analyze_mixed_clusters(snapshot_file_paths, working_directory_path):
    """Analyzes mixed gas-star clusters to track star formation within clusters.

    Given a list of snapshot file paths and a working directory path, this function gets all astrolink clusters in a fuzzycat cluster, and from that
    all the star particles in the cluster. It then calculates the gas mass, star mass, and gas fraction for each cluster in each snapshot.
    It also calculates the star formation rate and gas depletion rate for each cluster.
    The function saves the results in a specified directory.
    
    Provides improved analysis with:
    - Correct membership handling
    - Star formation rate calculation
    - Star formation efficiency tracking
    - More flexible star formation detection
    """
    print("Starting improved analysis of mixed gas-star clusters...")
    
    # Create output directories
    analysis_dir = f"{working_directory_path}Mixed_cluster_analysis/"
    plots_dir = f"{analysis_dir}Plots/"
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load fuzzy cluster data
    print(f"Loading fuzzy cluster data from {working_directory_path}...")

    clustered_ids = np.load(f"{working_directory_path}clusteredIDs.npy")
    print(f"clustered_ids shape: {clustered_ids.shape}")
    ordering = np.load(f"{working_directory_path}ordering.npy")
    print(f"ordering shape: {ordering.shape}")
    fuzzy_clusters = np.load(f"{working_directory_path}fuzzyClusters.npy")
    print(f"fuzzy_clusters shape: {fuzzy_clusters.shape}")
    memberships = np.load(f"{working_directory_path}memberships.npy")
    print(f"memberships shape: {memberships.shape}")

    # Verify data consistency
    n_snapshots = len(snapshot_file_paths)
    n_clusters = len(fuzzy_clusters)
    print(f"Processing {n_snapshots} snapshots for {n_clusters} fuzzy clusters")
    
    # Verify that membership data matches cluster sizes
    # for i, cluster in enumerate(fuzzy_clusters[:5]):  # Check first 5 clusters
    #     cluster_size = cluster[1] - cluster[0]
    #     membership_size = len(memberships[i])
    #     print(f"Cluster {i}: {cluster_size} members, {membership_size} membership values")
    #     if cluster_size != membership_size:
    #         print("WARNING: Membership array size doesn't match cluster size!")
    
    

    # Initialize data structures
    gas_masses = np.zeros((n_clusters, n_snapshots))
    star_masses = np.zeros((n_clusters, n_snapshots))
    gas_fractions = np.zeros((n_clusters, n_snapshots))
    star_formation_rates = np.zeros((n_clusters, n_snapshots-1))
    gas_depletion_rates = np.zeros((n_clusters, n_snapshots-1))
    star_formation_efficiencies = np.zeros((n_clusters, n_snapshots-1))
    snapshot_times = np.zeros(n_snapshots)


    # Create optimized particle tracking structures
    print("Setting up efficient particle tracking...")
    clustered_particle_ids = np.unique(clustered_ids)
    n_clustered_particles = len(clustered_particle_ids)
    print(f"Tracking {n_clustered_particles} clustered particles")

    # Create a mapping from particle ID to index
    particle_id_to_idx = {pid: idx for idx, pid in enumerate(clustered_particle_ids)}

    # Initialize particle type history array (0=none, 1=gas, 2=star)
    # This replaces the dictionary approach with a more efficient NumPy array
    particle_type_history = np.zeros((n_clustered_particles, n_snapshots), dtype=np.int8)
    
    star_clusters = np.zeros((n_snapshots, n_clusters), dtype=int)
    # Process each snapshot
    for snap_idx, snapshot_path in enumerate(snapshot_file_paths):
        snapshot_name = os.path.basename(snapshot_path)
        print(f"Processing snapshot {snap_idx+1}/{n_snapshots}: {snapshot_name}")
        
        # Load the simulation data
        simulation = pb.load(snapshot_path)
        snapshot_times[snap_idx] = simulation.properties['time'].in_units('Gyr')
        
        # Load the main halo and make it face-on
        main_halo = simulation.halos()[0]
        pb.analysis.angmom.faceon(main_halo)
        main_halo.physical_units()

        
        # Get particle data
        star_particles = main_halo.stars
        gas_particles = main_halo.gas
        
        # Extract IDs
        star_ids = np.array(star_particles['iord'])
        print(max(star_ids))
        gas_ids = np.array(gas_particles['iord'])
        print(max(gas_ids))

        print(np.sum(np.isin(gas_ids, star_ids)))

        star_masses_snap = np.array(star_particles['mass'].in_units('Msol'))
        gas_masses_snap = np.array(gas_particles['mass'].in_units('Msol'))

        # Create mass lookup dictionaries
        star_id_to_mass = dict(zip(star_ids, star_masses_snap))
        gas_id_to_mass = dict(zip(gas_ids, gas_masses_snap))

        # Create masks for finding clustered particles in gas and stars
        # This is much faster than dictionary lookups for each particle
        clustered_in_stars = np.isin(clustered_particle_ids, star_ids)
        clustered_in_gas = np.isin(clustered_particle_ids, gas_ids)
        
        # Vectorized update of particle types - updates all particles at once!
        particle_type_history[clustered_in_gas, snap_idx] = 1  # gas particles
        particle_type_history[clustered_in_stars, snap_idx] = 2  # star particles

        
        # Process each fuzzy cluster
        for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):
            
            

            astrolink_cluster_ids_in_fuzzycat_cluster = ordering[start_idx:end_idx]

            clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")

            index = f"{snap_idx:03d}"
            

            # Get AstroLink clusters for this snapshot that belong to the current FuzzyCat cluster
            cluster_filenames = [clusterFileNames[idx] for idx in astrolink_cluster_ids_in_fuzzycat_cluster if clusterFileNames[idx].startswith(index)]

            print(f"Found {len(cluster_filenames)} AstroLink clusters in FuzzyCat cluster {cluster_idx}")

            # Load the raw cluster data
            all_clusters = []
            for cluster_filename in cluster_filenames:
                cluster = np.load(f"{workingDirectoryPath}Clusters_iord/{cluster_filename}")
                all_clusters.append(cluster)

            # If we found any clusters, combine their particle data
            if all_clusters:
                member_ids = np.concatenate(all_clusters)
                member_memberships = np.ones(len(member_ids))
            else:
                member_ids = np.array([], dtype=int)

            #print(len(member_ids))
            mask = np.isin(member_ids, main_halo['iord'])
            no = np.sum(mask)



            # Calculate weighted gas and star masses
            gas_mass = 0.0
            star_mass = 0.0

            for i, particle_id in enumerate(member_ids):
                membership_weight = member_memberships[i]

                # Check if it's a gas particle
                if particle_id in gas_id_to_mass:
                    gas_mass += membership_weight * gas_id_to_mass[particle_id]
                # Check if it's a star particle
                elif particle_id in star_id_to_mass:
                    star_mass += membership_weight * star_id_to_mass[particle_id]

            # Store masses
            gas_masses[cluster_idx, snap_idx] = gas_mass
            star_masses[cluster_idx, snap_idx] = star_mass
            
            # Calculate gas fraction
            total_mass = gas_mass + star_mass
            if total_mass > 0:
                gas_fractions[cluster_idx, snap_idx] = gas_mass / total_mass
            else:
                gas_fractions[cluster_idx, snap_idx] = 0

            gasParticles = np.isin(member_ids, gas_ids)
            starParticles = np.isin(member_ids, star_ids)
            
            numGasParticles = np.sum(gasParticles)
            numStarParticles = np.sum(starParticles)

            #print("Not gas" + np.sum(~gasParticles))
            #print("Not Stars" + np.sum(~starParticles))

            #print(len(member_ids))
            #print(len(member_ids) - len(set(member_ids)))

            #print(member_ids[~gasParticles])
            #not_gas_ids = member_ids[~gasParticles]
            #get the particle objects from their ids
            # Get particles by their IDs from the main halo
            #not_gas_ids_set = set(not_gas_ids)

            # Create a boolean mask for all particles in the halo where ID is in not_gas_ids
            #mask = np.array([pid in not_gas_ids_set for pid in main_halo['iord']])
            #mask_no_values = np.sum(mask)
            # Use the mask to get the particles
            #not_gas_particles = main_halo[mask]

            #print(f"Found {len(not_gas_particles)} particles")
            #print(f"Families present: {not_gas_particles.families()}")
            #pb.plot.sph.image(not_gas['pos'], width=2*axis_limits, cmap='hot', log=True, dpi=500)
            

            
            

            print(f"Gas particles: {numGasParticles}, Star particles: {numStarParticles}")

            if(numStarParticles > 0):
                print(f"Star particles: {numStarParticles}")
                star_clusters[snap_idx][cluster_idx] = numStarParticles
            
        
    
    
    
    # Save raw data
    np.save(f"{analysis_dir}gas_masses.npy", gas_masses)
    np.save(f"{analysis_dir}star_masses.npy", star_masses)
    np.save(f"{analysis_dir}gas_fractions.npy", gas_fractions)
    np.save(f"{analysis_dir}star_clusters.npy", star_clusters)



def analysisresult(snapshotFilePaths, workingDirectoryPath, axisLimits, frameRate):
    # Identify significant clusters using more sophisticated criteria
    # For example, clusters with a gas fraction drop of more than 30% after a stable period
    # this would be a good indicator of star formation activity



    #load data
    analysis_dir = f"{workingDirectoryPath}Mixed_cluster_analysis/"
    gas_masses = np.load(f"{analysis_dir}gas_masses.npy")
    star_masses = np.load(f"{analysis_dir}star_masses.npy")
    gas_fractions = np.load(f"{analysis_dir}gas_fractions.npy")
    star_formation_rates = np.load(f"{analysis_dir}star_formation_rates.npy")
    star_formation_efficiencies = np.load(f"{analysis_dir}star_formation_efficiencies.npy")
    snapshot_times = np.load(f"{analysis_dir}snapshot_times.npy")

    star_clusters = np.load(f"{analysis_dir}star_clusters.npy")
    star_clusters = star_clusters.T
    # Count how many entries in star_clusters are non-zero
    non_zero_count = np.count_nonzero(star_clusters)
    print(f"Number of non-zero entries in star_clusters: {non_zero_count}")

    # Calculate percentage of entries that are non-zero
    total_entries = star_clusters.size
    non_zero_percentage = (non_zero_count / total_entries) * 100
    print(f"Percentage of non-zero entries: {non_zero_percentage:.2f}%")


    #get gas fractions where star_clusters is non-zero

    non_zero_indices = np.nonzero(star_clusters)
    non_zero_gas_fractions = gas_fractions[non_zero_indices]


    # Get the unique cluster indices that have stars
    unique_clusters = np.unique(non_zero_indices[0])
    print(f"Found {len(unique_clusters)} unique clusters containing stars")

    max_gas_fraction_changes = []
    cluster_gas_histories = []

        # For each cluster with stars, analyze its gas fraction history
        # for cluster_idx in unique_clusters:
        #     # Get all snapshots where cluster has stars
        #     snapshots_with_stars = non_zero_indices[1][non_zero_indices[0] == cluster_idx]
            
        #     # Get gas fraction history for this cluster
        #     cluster_gas_history = gas_fractions[cluster_idx]
            
        #     # Calculate maximum change in gas fraction
            
        #     max_change = np.max(cluster_gas_history) - np.min(cluster_gas_history)
        #     max_gas_fraction_changes.append(max_change)
            
        #     # Store the full history
        #     cluster_gas_histories.append(cluster_gas_history)
            
        #     print(f"Cluster {cluster_idx}: Max gas fraction change: {max_change:.2f}")
        
        #     if max_change > 0.3:  # 30% threshold
        #         print(f"  Starting gas fraction: {cluster_gas_history[0]:.2f}")
        #         print(f"  Ending gas fraction: {cluster_gas_history[-1]:.2f}")
        #         print(f"  Snapshots with stars: {snapshots_with_stars}")
        # Identify clusters with the most gas fraction changes
        # top_changing_clusters = np.argsort(max_gas_fraction_changes)  # Top 5 clusters
        # for i, idx in enumerate(top_changing_clusters):
        #     cluster_idx = unique_clusters[idx]
        #     print(f"{i+1}. Cluster {cluster_idx}: {max_gas_fraction_changes[idx]:.2f} change")


        # plt.figure(figsize=(12, 8))
        # snapshot_range = np.arange(len(snapshot_times)) if 'snapshot_times' in locals() else np.arange(gas_fractions.shape[1])

        # for i, idx in enumerate(top_changing_clusters):  
        #     cluster_idx = unique_clusters[idx]
        #     if(max_gas_fraction_changes[idx] > 0.3):
        #         plt.plot(snapshot_range, cluster_gas_histories[idx], 
        #                 marker='o', label=f"Cluster {cluster_idx} (Δ={max_gas_fraction_changes[idx]:.2f})")

        # plt.xlabel('Snapshot Index')
        # plt.ylabel('Gas Fraction')
        # plt.title('Gas Fraction Evolution in Clusters with Significant Changes')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.savefig(f"{analysis_dir}top_changing_clusters_gas_fraction.png", dpi=300)
        # plt.close()
    for cluster_idx in unique_clusters:
        # Get all snapshots where cluster has stars
        snapshots_with_stars = non_zero_indices[1][non_zero_indices[0] == cluster_idx]
        
        # Get gas fraction history for this cluster
        cluster_gas_history = gas_fractions[cluster_idx]
        
        # Parameters for detection
        stable_period_size = 3  # Number of snapshots to consider for a "stable" period
        drop_period_size = 3  # Number of snapshots to consider for the drop
        stable_threshold = 0.05  # Maximum variation allowed in the stable period
        
        # Find the best "stable then drop" pattern
        max_drop = 0
        best_stable_start = -1
        
        for i in range(len(cluster_gas_history) - stable_period_size - drop_period_size + 1):
            stable_window = cluster_gas_history[i:i+stable_period_size]
            drop_window = cluster_gas_history[i+stable_period_size:i+stable_period_size+drop_period_size]
            
            # Check if the first window is stable
            stability = np.max(stable_window) - np.min(stable_window)
            
            if stability <= stable_threshold:
                # Measure the drop from the end of the stable window to the minimum of the drop window
                drop = stable_window[-1] - np.min(drop_window)
                
                if drop > max_drop:
                    max_drop = drop
                    best_stable_start = i
        
        max_gas_fraction_changes.append(max_drop)
        cluster_gas_histories.append(cluster_gas_history)
        
        if best_stable_start != -1:
            stable_start = best_stable_start
            stable_end = best_stable_start + stable_period_size - 1
            drop_start = stable_end + 1
            drop_end = drop_start + drop_period_size - 1
            
            print(f"Cluster {cluster_idx}: Gas fraction is stable from snapshots {stable_start} to {stable_end}, then drops by {max_drop:.2f} between snapshots {drop_start} and {drop_end}")
        else:
            print(f"Cluster {cluster_idx}: No pattern of stable gas fraction followed by a drop detected")

    # Identify clusters with the most significant gas fraction drops after stable periods
    top_changing_indices = np.argsort(max_gas_fraction_changes)[::-1]  # Sort in descending order
    top_changing_clusters = [unique_clusters[idx] for idx in top_changing_indices]

    # Print the clusters with the most significant drops
    for i, cluster_idx in enumerate(top_changing_clusters[:5]):  # Show top 5
        idx = np.where(unique_clusters == cluster_idx)[0][0]
        print(f"{i+1}. Cluster {cluster_idx}: {max_gas_fraction_changes[idx]:.2f} drop after stable period")

    plt.figure(figsize=(12, 8))
    snapshot_range = np.arange(len(snapshot_times)) if 'snapshot_times' in locals() else np.arange(gas_fractions.shape[1])

    # Track the identified stable and drop regions
    stable_regions = {}
    drop_regions = {}

    # Plot clusters with significant drops (> 0.3)
    for idx in top_changing_indices:
        cluster_idx = unique_clusters[idx]
        drop_value = max_gas_fraction_changes[idx]
        
        if drop_value > 0.03:  # Only show clusters with significant drops
            plt.plot(snapshot_range, cluster_gas_histories[idx], 
                    marker='o', label=f"Cluster {cluster_idx} (Drop={drop_value:.2f})")
            
            # If we stored the best period locations in the analysis loop, highlight them
            if 'best_stable_start' in locals() and best_stable_start != -1:
                stable_start = best_stable_start
                stable_end = best_stable_start + stable_period_size - 1
                drop_start = stable_end + 1
                drop_end = drop_start + drop_period_size - 1
                
                # Store the regions for this cluster
                stable_regions[cluster_idx] = (stable_start, stable_end)
                drop_regions[cluster_idx] = (drop_start, drop_end)
                
                # Highlight the stable period with a green background
                plt.axvspan(stable_start, stable_end, color='green', alpha=0.2)
                # Highlight the drop period with a red background
                plt.axvspan(drop_start, drop_end, color='red', alpha=0.2)

    plt.xlabel('Snapshot Index')
    plt.ylabel('Gas Fraction')
    plt.title('Gas Fraction Evolution in Clusters with Significant Drops After Stable Periods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{analysis_dir}top_changing_clusters_gas_fraction.png", dpi=300)
    plt.close()



def analyze_all_burst_clusters(snapshot_file_paths, working_directory_path):
    """
    Main function to identify burst clusters and track their age spikes
    """
    # First identify all burst clusters
    burst_clusters = identify_burst_clusters(snapshot_file_paths, working_directory_path)
    
    # Create summary dataframe
    summary_data = []
    
    # Track age spike for each burst cluster
    for cluster_idx, metrics in burst_clusters.items():
        print(f"\nTracking age spike for burst cluster {cluster_idx}")
        spike_positions, age_distributions = track_age_spike(cluster_idx, snapshot_file_paths, working_directory_path)
        
        # Analyze spike positions to find birth snapshot
        valid_positions = [(snap, age) for snap, age in spike_positions if age is not None]
        if len(valid_positions) > 2:
            snaps, ages = zip(*valid_positions)
            slope, intercept, r_value, p_value, std_err = stats.linregress(snaps, ages)
            birth_snapshot = -intercept / slope if slope != 0 else None
            
            summary_data.append({
                'cluster_idx': cluster_idx,
                'star_count': metrics['star_count'],
                'burst_fraction': metrics['max_burst_fraction'],
                'peak_age_latest': valid_positions[-1][1],
                'estimated_birth_snapshot': birth_snapshot,
                'regression_r2': r_value**2,
                'data_points': len(valid_positions)
            })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{working_directory_path}Cluster_origin_analysis/burst_clusters_summary.csv", index=False)
    
    # Create visualizations for the most promising clusters
    if not summary_df.empty:
        # Sort by highest r-squared and closest estimated birth snapshot
        promising_clusters = summary_df.sort_values(by=['regression_r2', 'estimated_birth_snapshot'], 
                                                   ascending=[False, True]).head(10)
        
        print("\nMost promising clusters for birth tracking:")
        print(promising_clusters[['cluster_idx', 'estimated_birth_snapshot', 'regression_r2']])
    
    return burst_clusters



def track_age_spike(cluster_idx, snapshot_file_paths, working_directory_path):
    """
    Track the position of the age spike across snapshots for a burst cluster
    
    Parameters:
    -----------
    cluster_idx: int, the index of the burst cluster to track
    
    Returns:
    --------
    spike_positions: list of (snapshot_idx, peak_age) tuples
    """
    # Load fuzzy cluster data
    clustered_ids = np.load(f"{working_directory_path}clusteredIDs.npy")
    ordering = np.load(f"{working_directory_path}ordering.npy")
    fuzzy_clusters = np.load(f"{working_directory_path}fuzzyClusters.npy")
    clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")
    
    start_idx, end_idx = fuzzy_clusters[cluster_idx]
    astrolink_cluster_ids = ordering[start_idx:end_idx]
    
    spike_positions = []
    age_distributions = []
    
    for snap_idx, snapshot_path in enumerate(snapshot_file_paths):
        print(f"Processing snapshot {snap_idx+1}/{len(snapshot_file_paths)}")
        
        # Load the simulation data
        simulation = pb.load(snapshot_path)
        main_halo = simulation.halos()[0]
        pb.analysis.angmom.faceon(main_halo)
        main_halo.physical_units()
        
        star_particles = main_halo.stars
        star_ids = star_particles['iord']
        
        # Get AstroLink clusters for this snapshot that belong to the current FuzzyCat cluster
        index = f"{snap_idx:03d}"
        cluster_filenames = [clusterFileNames[idx] for idx in astrolink_cluster_ids 
                            if clusterFileNames[idx].startswith(index)]
        
        if not cluster_filenames:
            spike_positions.append((snap_idx, None))
            continue
            
        # Load clusters and combine member particles
        all_clusters = []
        for cluster_filename in cluster_filenames:
            cluster = np.load(f"{working_directory_path}Clusters_iord/{cluster_filename}")
            all_clusters.append(cluster)
            
        if all_clusters:
            member_ids = np.concatenate(all_clusters)
        else:
            spike_positions.append((snap_idx, None))
            continue
            
        star_ids_mask = np.isin(member_ids, star_ids)
        if not np.any(star_ids_mask):
            spike_positions.append((snap_idx, None))
            continue
            
        member_ids = member_ids[star_ids_mask]
        star_ages = star_particles['age'][np.isin(star_ids, member_ids)]
        
        if len(star_ages) < 10:
            spike_positions.append((snap_idx, None))
            continue
            
        # Identify the age spike using kernel density estimation
        kde = stats.gaussian_kde(star_ages)
        x_grid = np.linspace(0, max(star_ages), 1000)
        kde_vals = kde(x_grid)
        peak_idx = np.argmax(kde_vals)
        peak_age = x_grid[peak_idx]
        
        spike_positions.append((snap_idx, peak_age))
        age_distributions.append((snap_idx, star_ages))
        
        # Plot the age distribution with the identified peak
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot KDE
        ax.plot(x_grid, kde_vals, 'r-', lw=2)
        
        # Plot histogram
        ax.hist(star_ages, bins=30, density=True, alpha=0.6)
        
        # Mark the peak
        ax.axvline(x=peak_age, color='k', linestyle='--', label=f'Peak Age: {peak_age:.3f} Gyr')
        
        ax.set_xlabel('Age (Gyr)')
        ax.set_ylabel('Density')
        ax.set_title(f'Cluster {cluster_idx} - Snapshot {snap_idx}')
        ax.legend()
        
        # Save the plot
        os.makedirs(f"{working_directory_path}Cluster_origin_analysis/age_peaks/cluster_{cluster_idx}/", exist_ok=True)
        plt.savefig(f"{working_directory_path}Cluster_origin_analysis/age_peaks/cluster_{cluster_idx}/snapshot_{snap_idx}.png")
        plt.close()
        if(snap_idx == 15):
            break
    
    # Plot the evolution of the age spike
    valid_positions = [(snap, age) for snap, age in spike_positions if age is not None]
    if valid_positions:
        snaps, ages = zip(*valid_positions)
        
        plt.figure(figsize=(10, 6))
        plt.plot(snaps, ages, 'o-', markersize=8)
        plt.xlabel('Snapshot Index')
        plt.ylabel('Peak Age (Gyr)')
        plt.title(f'Evolution of Age Peak for Cluster {cluster_idx}')
        plt.grid(True, alpha=0.3)
        
        # Add linear regression to predict birth snapshot
        if len(snaps) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(snaps, ages)
            birth_snapshot = -intercept / slope if slope != 0 else None
            
            # Plot regression line
            x_reg = np.array([min(snaps), max(snaps)])
            y_reg = slope * x_reg + intercept
            plt.plot(x_reg, y_reg, 'r--', label=f'Regression: Birth at snapshot ~{birth_snapshot:.1f}')
            
            # Extend line to x-axis to visualize birth snapshot
            if birth_snapshot > 0:
                plt.plot([birth_snapshot, birth_snapshot], [0, 0], 'ro', markersize=10)
                plt.axvline(x=birth_snapshot, color='r', linestyle=':')
            
            plt.legend()
        
        os.makedirs(f"{working_directory_path}Cluster_origin_analysis/spike_tracking/", exist_ok=True)
        plt.savefig(f"{working_directory_path}Cluster_origin_analysis/spike_tracking/cluster_{cluster_idx}_peak_evolution.png")
        plt.close()
    
    return spike_positions, age_distributions


def star_cluster_analysis_2(snapshot_file_paths, working_directory_path, axisLimits, frameRate):
    """
    Same as star_cluster_analysis, but with different reference frame for time: 
    The age distribution is plotted once for each cluster -> no movie, but a plot for each cluster
    Depict changes over time by using colors
    """
    # Create output directories
    analysis_dir = f"{working_directory_path}further_Cluster_origin_analysis_2/"
    plots_dir = f"{analysis_dir}plots/"
    star_ages_dir = f"{analysis_dir}star_ages/"
    os.makedirs(star_ages_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load fuzzy cluster data
    print(f"Loading fuzzy cluster data from {working_directory_path}...")

    clustered_ids = np.load(f"{working_directory_path}clusteredIDs.npy")
    print(f"clustered_ids shape: {clustered_ids.shape}")
    ordering = np.load(f"{working_directory_path}ordering.npy")
    print(f"ordering shape: {ordering.shape}")
    fuzzy_clusters = np.load(f"{working_directory_path}fuzzyClusters.npy")
    print(f"fuzzy_clusters shape: {fuzzy_clusters.shape}")

    n_snapshots = len(snapshot_file_paths)
    n_clusters = len(fuzzy_clusters)
    print(f"Processing {n_snapshots} snapshots for {n_clusters} fuzzy clusters")


    #get the snapshot ranges for each fuzzycat cluster
    fuzzy_cluster_snapshot_ranges = np.zeros((n_clusters, 2), dtype=int)
    for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):
        astrolink_cluster_ids_in_fuzzycat_cluster = ordering[start_idx:end_idx]
        clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")
        cluster_filenames = [clusterFileNames[idx] for idx in astrolink_cluster_ids_in_fuzzycat_cluster]
        # Get the snapshot indices for this cluster
        snapshot_indices = set([int(filename.split('_')[0]) for filename in cluster_filenames])
        fuzzy_cluster_snapshot_ranges[cluster_idx] = [min(snapshot_indices), max(snapshot_indices)]

        
    burst_clusters = identify_burst_clusters(snapshot_file_paths, working_directory_path, star_ages_dir, threshold_fraction=0.8, time_window=0.2)

    for cluster_idx in burst_clusters:
        cluster_dir = f"{plots_dir}cluster_{cluster_idx}/"
        os.makedirs(cluster_dir, exist_ok=True)

        all_member_ids_until_now = np.array([], dtype=int)


        for snap_idx, snapshot_path in enumerate(snapshot_file_paths):
            snapshot_name = os.path.basename(snapshot_path)
            print(f"Processing snapshot {snap_idx+1}/{n_snapshots}: {snapshot_name}")
            
            # # Load the simulation data
            # simulation = pb.load(snapshot_path)
            
            # # Load the main halo and make it face-on
            # main_halo = simulation.halos()[0]
            # pb.analysis.angmom.faceon(main_halo)
            # main_halo.physical_units()

            
            # # Get particle data
            # star_particles = main_halo.stars
            # star_ids = star_particles['iord']


            star_data = np.load(f"{star_ages_dir}star_ages_{snap_idx:03d}.npy")
            star_ids = star_data[0]
            star_ages = star_data[1]


            start_idx, end_idx = fuzzy_clusters[cluster_idx]

            astrolink_cluster_ids_in_fuzzycat_cluster = ordering[start_idx:end_idx]

            clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")

            index = f"{snap_idx:03d}"
            

            # Get AstroLink clusters for this snapshot that belong to the current FuzzyCat cluster
            cluster_filenames = [clusterFileNames[idx] for idx in astrolink_cluster_ids_in_fuzzycat_cluster if clusterFileNames[idx].startswith(index)]

            print(f"Found {len(cluster_filenames)} AstroLink clusters in FuzzyCat cluster {cluster_idx}")

            # Load the raw cluster data
            all_clusters = []
            for cluster_filename in cluster_filenames:
                cluster = np.load(f"{workingDirectoryPath}Clusters_iord/{cluster_filename}")
                all_clusters.append(cluster)

            # If we found any clusters, combine their particle data
            if all_clusters:
                member_ids = np.concatenate(all_clusters)
            else:
                member_ids = np.array([], dtype=int)

            all_member_ids_until_now = np.unique(np.concatenate((all_member_ids_until_now, member_ids)))
            #get birth dates of star particles
            this_snapshot_star_ages = star_ages[np.isin(star_ids, member_ids)]

            star_ages_for_cluster_until_now = star_ages[np.isin(star_ids, all_member_ids_until_now)]  


def identify_burst_clusters(snapshot_file_paths, working_directory_path, analysis_path, star_ages_dir, threshold_fraction=0.8, time_window=0.2):
    """
    Identify clusters that formed in bursts
    
    Parameters:
    -----------
    threshold_fraction: float, fraction of stars that need to form within time_window to qualify as a burst cluster
    time_window: float, time window in Gyr to define a burst
    
    Returns:
    --------
    burst_clusters: dict, keys are cluster indices, values are dictionaries with metrics
    """


    # Load fuzzy cluster data
    clustered_ids = np.load(f"{working_directory_path}clusteredIDs.npy")
    ordering = np.load(f"{working_directory_path}ordering.npy")
    fuzzy_clusters = np.load(f"{working_directory_path}fuzzyClusters.npy")
    clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")

    print("Starting analysis of burst clusters...")

    burst_clusters = {}
    metrics_data = []
    for snap_idx, snapshot_path in enumerate(snapshot_file_paths):
        print(f"Processing snapshot {snap_idx}/{len(snapshot_file_paths)}")
        
        star_data = np.load(f"{star_ages_dir}star_ages_{snap_idx:03d}.npy")
        star_ids = star_data[:,0]
        star_ages = star_data[:,1]
        
        # Analyze each fuzzy cluster
        for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):

            if(burst_clusters.get(cluster_idx) is not None):
                continue


            print(f"Analyzing cluster {cluster_idx}/{len(fuzzy_clusters)}")

            
            astrolink_cluster_ids_in_fuzzycat_cluster = ordering[start_idx:end_idx]

            clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")

            index = f"{snap_idx:03d}"
            

            # Get AstroLink clusters for this snapshot that belong to the current FuzzyCat cluster
            cluster_filenames = [clusterFileNames[idx] for idx in astrolink_cluster_ids_in_fuzzycat_cluster if clusterFileNames[idx].startswith(index)]

            #print(f"Found {len(cluster_filenames)} AstroLink clusters in FuzzyCat cluster {cluster_idx}")

            # Load the raw cluster data
            all_clusters = []
            for cluster_filename in cluster_filenames:
                cluster = np.load(f"{workingDirectoryPath}Clusters_iord/{cluster_filename}")
                all_clusters.append(cluster)

            # If we found any clusters, combine their particle data
            if all_clusters:
                member_ids = np.concatenate(all_clusters)
            else:
                member_ids = np.array([], dtype=int)

            
            star_ages_in_cluster = star_ages[np.isin(star_ids, member_ids)]
            
            if len(star_ages_in_cluster) < 50:  # Skip clusters with too few stars
                continue
                
            # Calculate metrics
            sorted_ages = np.sort(star_ages_in_cluster)
            age_range = sorted_ages.max() - sorted_ages.min()
            
            # Calculate max fraction of stars formed within any time_window
            max_fraction = 0
            peak_age = None

            
            
            for i in range(len(sorted_ages)):
                window_end = sorted_ages[i] + time_window
                stars_in_window = np.sum((sorted_ages >= sorted_ages[i]) & (sorted_ages <= window_end))
                fraction = stars_in_window / len(sorted_ages)
                
                if fraction > max_fraction:
                    max_fraction = fraction
                    peak_age = sorted_ages[i]
            
            # Calculate t25-t75 (time to form middle 50% of stars)
            t25 = np.percentile(sorted_ages, 25)
            t75 = np.percentile(sorted_ages, 75)
            t25_t75 = t75 - t25
            
            # Calculate formation steepness (max rate of CDF change)
            hist, bin_edges = np.histogram(star_ages_in_cluster, bins=1000)
            cdf = np.cumsum(hist) * (bin_edges[1] - bin_edges[0])
            max_steepness = np.max(np.diff(cdf) / np.diff(bin_edges[1:]))
            
            metrics = {
                'cluster_idx': cluster_idx,
                'star_count': len(star_ages_in_cluster),
                'max_burst_fraction': max_fraction,
                'peak_age': peak_age,
                'age_range': age_range,
                't25_t75': t25_t75,
                'max_steepness': max_steepness
            }
        
            
            # Store information for burst clusters
            if max_fraction >= threshold_fraction:
                burst_clusters[cluster_idx] = metrics
                print(f"Cluster {cluster_idx} is a burst cluster with max fraction {max_fraction:.2f} and peak age {peak_age:.2f} Gyr")
