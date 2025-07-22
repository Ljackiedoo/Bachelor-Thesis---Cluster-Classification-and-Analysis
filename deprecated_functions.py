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


def plotTemperatureCuts(snapshotFilePaths, workingDirectoryPath, particleName, axisLimits, frameRate):
    """Plots the temperature cuts for the gas particles in the simulation.
    """
    # Prepare output path for this particle type
    saveFileNameStem = f"{workingDirectoryPath}Halo_plots/temperature_cuts_"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(saveFileNameStem), exist_ok=True)

    for index, snapshotFilePath in enumerate(snapshotFilePaths):
        snapshotFileName = snapshotFilePath.split('/')[-1]
        
        print(f"Processing temperature cuts for snapshot {snapshotFileName}...")
        
        # Load the simulation snapshot directly
        simulation = pb.load(snapshotFilePath)
        mainHalo = simulation.halos()[0]
        pb.analysis.angmom.faceon(mainHalo)
        mainHalo.physical_units()
        
        # Center the halo
        center = np.median(mainHalo.dm['pos'], axis=0)
        mainHalo['pos'] -= center
        
        # Get gas particles and filter by temperature
        all_gas = mainHalo.gas
        hot_gas = all_gas[all_gas['temp'] > 1e5]
        cold_gas = all_gas[all_gas['temp'] < 1e5]
        
        # Make plot of temperature cuts
        print(f"Plotting temperature cuts for {snapshotFileName}")
        # Create a figure with three subplots
        # Create separate directories for each gas type
        os.makedirs(f"{workingDirectoryPath}Halo_plots/all_gas/", exist_ok=True)
        os.makedirs(f"{workingDirectoryPath}Halo_plots/hot_gas/", exist_ok=True)
        os.makedirs(f"{workingDirectoryPath}Halo_plots/cold_gas/", exist_ok=True)
        
        # Create separate plots for each gas type
        # All gas
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        pb.plot.sph.image(all_gas, qty='rho', width=2*axisLimits, cmap='hot', log=True, ax=ax1, dpi=500)
        ax1.set_title('All Gas Particles')
        plt.savefig(f"{workingDirectoryPath}Halo_plots/all_gas/frame_{index:04d}_{snapshotFileName}.png", dpi=500, bbox_inches='tight')
        plt.close(fig1)
        
        # Hot gas
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        pb.plot.sph.image(hot_gas, qty='rho', width=2*axisLimits, cmap='hot', log=True, ax=ax2, dpi = 500)
        ax2.set_title('Hot Gas Particles (T > 10^5 K)')
        plt.savefig(f"{workingDirectoryPath}Halo_plots/hot_gas/frame_{index:04d}_{snapshotFileName}.png", dpi=500, bbox_inches='tight')
        plt.close(fig2)
        
        # Cold gas
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        pb.plot.sph.image(cold_gas, qty='rho', width=2*axisLimits, cmap='hot', log=True, ax=ax3, dpi=500)
        ax3.set_title('Cold Gas Particles (T < 10^5 K)')
        plt.savefig(f"{workingDirectoryPath}Halo_plots/cold_gas/frame_{index:04d}_{snapshotFileName}.png", dpi=500, bbox_inches='tight')
        plt.close(fig3)
        
        gc.collect()


    (
        ffmpeg
        .input(f"{workingDirectoryPath}Halo_plots/all_gas/*.png", pattern_type='glob', framerate=frameRate)
        .output(f"{workingDirectoryPath}{workingDirectoryPath.split('/')[-2]}_all_gas_movie.mp4")
        .run()
    )

    (
        ffmpeg
        .input(f"{workingDirectoryPath}Halo_plots/hot_gas/*.png", pattern_type='glob', framerate=frameRate)
        .output(f"{workingDirectoryPath}{workingDirectoryPath.split('/')[-2]}_hot_gas_movie.mp4")
        .run()
    )

    (
        ffmpeg
        .input(f"{workingDirectoryPath}Halo_plots/cold_gas/*.png", pattern_type='glob', framerate=frameRate)
        .output(f"{workingDirectoryPath}{workingDirectoryPath.split('/')[-2]}_cold_gas_movie.mp4")
        .run()
    )


def mass_distribution_analysis(burst_clusters, working_directory_path, star_data_dir, mass_distributions_dir, n_snapshots, ordering, fuzzy_clusters):
    """
    Analyzes the mass distribution of stars in burst clusters over time and creates plots for each cluster.
    First plot is a 2D heatmap of mass over time, showing how the density of star masses in the cluster changes over time.
    Second plot is how the mean and median mass the cluster changes over time for each cluster.
    """
    burst_clusters_masses_all_snapshots = { c: {} for c in burst_clusters }

    #only use a few random burst clusters for now for faster analysis
    # num_clusters_to_select = 5
    # all_cluster_ids = list(burst_clusters.keys())
    # selected_cluster_ids = random.sample(all_cluster_ids, num_clusters_to_select)
    # burst_clusters = {cluster_id: burst_clusters[cluster_id] for cluster_id in selected_cluster_ids}


    clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy") 

    for cluster_idx in burst_clusters:
        cluster_dir = f"{mass_distributions_dir}cluster_{cluster_idx}/"
        os.makedirs(cluster_dir, exist_ok=True)

        fuzzy_start_idx, fuzzy_end_idx = fuzzy_clusters[cluster_idx]
        astrolink_cluster_ids_in_fuzzycat_cluster = ordering[fuzzy_start_idx:fuzzy_end_idx]

        burst_snapshot = burst_clusters[cluster_idx]['burst_snapshot']
        cluster_detected_snapshot = burst_clusters[cluster_idx]['cluster_start_snapshot']
        cluster_lost_snapshot = burst_clusters[cluster_idx]['cluster_end_snapshot']
        mass_list = []
        snap_indices = []
        # Determine the range of snapshots to iterate over
        loop_start_snap_idx = max(0, burst_snapshot - 5) 
        
        print(f"Processing Cluster {cluster_idx}, Snapshots from {loop_start_snap_idx} up to {n_snapshots-1}")

        for snap_idx in range(loop_start_snap_idx, n_snapshots):
                
            star_data = np.load(f"{star_data_dir}star_data_{snap_idx:03d}.npy")
            star_ids = star_data[:,0]
            star_masses = star_data[:,2]

            # Use a descriptive name for the snapshot index string
            snapshot_index_str = f"{snap_idx:03d}"
            
            # Get AstroLink cluster filenames for this snapshot that belong to the current FuzzyCat cluster
            # This assumes astrolink_cluster_ids_in_fuzzycat_cluster are indices for clusterFileNames array
            candidate_filenames = clusterFileNames[astrolink_cluster_ids_in_fuzzycat_cluster]
            relevant_cluster_filenames = [cfn for cfn in candidate_filenames if cfn.startswith(snapshot_index_str)]
            
            all_clusters_star_ids = []
            for cluster_filename in relevant_cluster_filenames:
                # Ensure workingDirectoryPath is correctly cased for Clusters_iord
                cluster_data_path = os.path.join(working_directory_path, "Clusters_iord", cluster_filename)
                try:
                    cluster_star_ids = np.load(cluster_data_path)
                    all_clusters_star_ids.append(cluster_star_ids)
                except FileNotFoundError:
                    print(f"Warning: Cluster file {cluster_data_path} not found for snap {snap_idx}. Skipping this file.")
                    continue

            if all_clusters_star_ids:
                member_ids = np.concatenate(all_clusters_star_ids)
                member_ids = np.unique(member_ids) # Important if stars can be in multiple sub-clusters
            else:
                member_ids = np.array([], dtype=int)
            
            star_masses_in_cluster = star_masses[np.isin(star_ids, member_ids)]
            mass_list.append(star_masses_in_cluster)
            snap_indices.append(snap_idx)

            burst_clusters_masses_all_snapshots[cluster_idx][snap_idx] = star_masses_in_cluster

        # --- Plotting Section ---
        
        # Filter out snapshots where no stars were found in the cluster for this specific mass_list
        mass_list_with_data = []
        snap_indices_with_data = []
        for m, s_idx in zip(mass_list, snap_indices):
            if len(m) > 0:
                mass_list_with_data.append(m)
                snap_indices_with_data.append(s_idx)

        if not mass_list_with_data:
            print(f"  → No stars with mass data found for Cluster {cluster_idx} across any processed snapshot. Skipping plots.")
            plt.close('all')
            continue

        # Determine common bins for histograms based on all mass data for this cluster
        all_masses_for_bins = np.concatenate(mass_list_with_data)

        min_m_overall, max_m_overall = np.min(all_masses_for_bins), np.max(all_masses_for_bins)
        
        # Define plot range for mass axis; handle cases with single mass value or very narrow range
        if np.isclose(min_m_overall, max_m_overall):
            offset = 0.5 if np.isclose(min_m_overall, 0.0) else 0.1 * abs(min_m_overall)
            if np.isclose(offset, 0.0): offset = 0.5 # Ensure offset is non-zero (e.g. if min_m_overall is extremely small)
            min_m_plot = min_m_overall - offset
            max_m_plot = max_m_overall + offset
        else:
            min_m_plot = min_m_overall
            max_m_plot = max_m_overall

        # Global bins for all plots for this cluster (30 bins, 31 edges)
        bins = np.linspace(min_m_plot, max_m_plot, 31) 
        
        # Determine snapshot range for titles from actual data plotted
        title_snap_min = snap_indices_with_data[0]
        title_snap_max = snap_indices_with_data[-1]

        # # --- Plot 1: 2D Heatmap of Mass Distribution over Time ---
    
        # Calculate histogram data for each snapshot
        # hist_data_list will store 1D arrays of histogram counts (densities)
        hist_data_list = []
        for current_masses in mass_list_with_data: # Already filtered for non-empty lists
            counts, _ = np.histogram(current_masses, bins=bins)
            hist_data_list.append(counts)
        
        # Convert list of 1D arrays to a 2D numpy array.
        # Each row corresponds to a snapshot, each column to a mass bin.
        # So, hist_data_2d has shape (number_of_snapshots, number_of_mass_bins)
        hist_data_2d_original_orientation = np.array(hist_data_list)
        
        plt.figure(figsize=(10, 7))
        
        # Define mesh edges for pcolormesh.
        # X-axis will be Snapshot Index
        # Y-axis will be Stellar Mass
        x_mesh_edges = np.arange(len(snap_indices_with_data) + 1) 
        y_mesh_edges = bins 
        
        # Prepare data for pcolormesh:
        # pcolormesh(X_edges, Y_edges, C_values) expects C_values[i,j] to map to Y_edges[i] and X_edges[j].
        # If Y is mass (rows) and X is snapshots (columns), C needs shape (num_mass_bins, num_snapshots).
        # So, we transpose hist_data_2d_original_orientation.
        data_for_pcolormesh = hist_data_2d_original_orientation.T

        plt.pcolormesh(
            x_mesh_edges,        # Edges for Snapshot Index axis
            y_mesh_edges,        # Edges for Stellar Mass axis
            data_for_pcolormesh, # Transposed data, shape (num_mass_bins, num_snapshots)
            cmap='viridis', 
            shading='flat', 
            vmin=0  # Density is non-negative
        )
        plt.colorbar(label="Absolute Count")
        plt.xlabel("Snapshot Index")
        plt.ylabel("Stellar Population Mass [Msol]")
        
        # Set x-axis ticks to actual snapshot numbers.
        # tick_positions_x are centers of the pcolormesh cells along the x-axis (snapshots).
        tick_positions_x = x_mesh_edges[:-1] + 0.5 
        tick_labels_x = [str(s) for s in snap_indices_with_data]
        
        # Reduce number of x-ticks if too many snapshots to display clearly
        if len(snap_indices_with_data) > 20: # Adjust this threshold as needed
            num_ticks_display = 10 
            step = max(1, len(snap_indices_with_data) // num_ticks_display)
            plt.xticks(tick_positions_x[::step], tick_labels_x[::step])
        else:
            plt.xticks(tick_positions_x, tick_labels_x)

        plt.title(f"Cluster {cluster_idx}: Mass Distribution Evolution, Snapshots {title_snap_min}→{title_snap_max}")
        plt.tight_layout()
    
        outfn_heatmap = os.path.join(cluster_dir, f"mass_dist_heatmap_absolute_counts.png")
        plt.savefig(outfn_heatmap, dpi=200)
        plt.close()
        print(f"  → Saved heatmap for Cluster {cluster_idx} at {outfn_heatmap}")

        
        # # --- Plot 2: Summary Statistics (Mean, Median, IQR) over Time ---
        
        mean_masses = [np.mean(m) for m in mass_list_with_data]
        median_masses = [np.median(m) for m in mass_list_with_data]
        q25_masses = [np.percentile(m, 25) for m in mass_list_with_data]
        q75_masses = [np.percentile(m, 75) for m in mass_list_with_data]

        plt.figure(figsize=(10, 6))
        plt.plot(snap_indices_with_data, mean_masses, label='Mean Mass', marker='o', linestyle='-')
        plt.plot(snap_indices_with_data, median_masses, label='Median Mass', marker='s', linestyle='--')
        # Shaded region for IQR
        plt.fill_between(snap_indices_with_data, q25_masses, q75_masses, alpha=0.2, label='IQR (25th-75th percentile)', color='gray')
        plt.axvline(x=cluster_detected_snapshot, color='purple', linestyle='--', label='Cluster detected')
        plt.axvline(x=cluster_lost_snapshot, color='orange', linestyle='--', label='Cluster lost')

        plt.xlabel("Snapshot Index")
        plt.ylabel("Stellar Mass [Msol]")
        plt.title(f"Cluster {cluster_idx}: Mass Statistics Evolution, Snapshots {title_snap_min}→{title_snap_max}")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        
        outfn_stats = os.path.join(cluster_dir, f"mass_stats_evolution.png")
        plt.savefig(outfn_stats, dpi=200)
        plt.close() # Close the statistics figure
        print(f"  → Saved mass statistics plot for Cluster {cluster_idx} at {outfn_stats}")

        #---- Plot 3: All Masses in Cluster Over Time ----
        # Shows how total mass in cluster changes over time
        total_masses = [np.sum(m) for m in mass_list_with_data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(snap_indices_with_data, total_masses, label='Total Mass', marker='o', linestyle='-')
        plt.axvline(x=cluster_detected_snapshot, color='purple', linestyle='--', label='Cluster detected')
        plt.axvline(x=cluster_lost_snapshot, color='orange', linestyle='--', label='Cluster lost')
        
        plt.xlabel("Snapshot Index")
        plt.ylabel("Total Stellar Mass [Msol]")
        plt.title(f"Cluster {cluster_idx}: Total Mass Evolution, Snapshots {title_snap_min}→{title_snap_max}")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        outfn_total_mass = os.path.join(cluster_dir, f"total_mass_evolution.png")
        plt.savefig(outfn_total_mass, dpi=200)
        plt.close()

        #---- Plot 4: No. of stars in Cluster Over Time ----
        # Shows how number of stars in cluster changes over time
        num_stars_in_cluster = [len(m) for m in mass_list_with_data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(snap_indices_with_data, num_stars_in_cluster, label='Number of Stars', marker='o', linestyle='-')
        plt.axvline(x=cluster_detected_snapshot, color='purple', linestyle='--', label='Cluster detected')
        plt.axvline(x=cluster_lost_snapshot, color='orange', linestyle='--', label='Cluster lost')
        
        plt.xlabel("Snapshot Index")
        plt.ylabel("Number of Stars")
        plt.title(f"Cluster {cluster_idx}: Number of Stars Evolution, Snapshots {title_snap_min}→{title_snap_max}")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        outfn_num_stars = os.path.join(cluster_dir, f"num_stars_evolution.png")
        plt.savefig(outfn_num_stars, dpi=200)
        plt.close()
        print(f"  → Saved number of stars plot for Cluster {cluster_idx} at {outfn_num_stars}")

    #save all snapshots' masses for later use
    np.save(f"{mass_distributions_dir}burst_clusters_masses_all_snapshots.npy", burst_clusters_masses_all_snapshots, allow_pickle=True)

    # Calculate the exponential decline 

def contamination_analysis(burst_clusters, cluster_masses, contamination_analysis_dir, workingDirectoryPath, ordering, fuzzy_clusters):
    """
    Analyze the contamination of burst clusters by other particles.
    This function calculates the fraction of particles in burst clusters over time that are not part of the initial cluster.
    It then plots the contamination fraction over time for each burst cluster.
    Contamination: Fraction of members at time T that were NOT present at the start.
    Loss: Fraction of original members at the start that are NO LONGER present at time T.
    """

    star_formation_timespan = 0.025 # in Gyr, adjust as needed
    star_formation_timespan_in_snapshots = star_formation_timespan * 2000 / 13.8  # Convert Gyr to snapshots (2000 snapshots for 13.8 Gyr simulation)

    earliest_start_snapshot = min(metrics['cluster_start_snapshot'] for metrics in burst_clusters.values())
    latest_end_snapshot = max(metrics['cluster_end_snapshot'] for metrics in burst_clusters.values())
    # Initialize a dictionary to store contamination data for each cluster
    if not os.path.exists(f"{contamination_analysis_dir}burst_clusters_contamination_data.npy"):
        contamination_data = {}
        for cluster_idx, metrics in burst_clusters.items():
            # Extract the start and end snapshots for the cluster
            print(f"Analyzing contamination for cluster {cluster_idx}...")
            start_snapshot = metrics['cluster_start_snapshot']
            end_snapshot = metrics['cluster_end_snapshot']
            # Initialize a list to store contamination fractions for each snapshot
            contamination_fractions = []
            loss_fractions = []

            star_formation_snapshots = list(range(int(np.floor(start_snapshot - star_formation_timespan_in_snapshots/2)), int(np.ceil(start_snapshot + star_formation_timespan_in_snapshots/2 + 1))))

            initial_member_ids = set()
            for snap in star_formation_snapshots:
                if snap > end_snapshot:
                    print(f"  → Skipping snapshot {snap} for cluster {cluster_idx} as it exceeds the end snapshot {end_snapshot}.")
                    continue

                new_ids = set(get_member_ids_for_fuzzycat_cluster(cluster_idx, ordering, fuzzy_clusters, workingDirectoryPath, snap))
                initial_member_ids.update(new_ids)


                snapshot_member_ids = set(get_member_ids_for_fuzzycat_cluster(cluster_idx, ordering, fuzzy_clusters, workingDirectoryPath, snap))

                if(len(snapshot_member_ids) == 0):
                    contamination_fractions.append(0.0)
                    loss_fractions.append(1.0)
                    continue

                # Calculate the contamination fraction
                differing_ids = snapshot_member_ids.difference(initial_member_ids)
                contamination_fraction = len(differing_ids) / len(snapshot_member_ids)
                # Store the contamination fraction for this snapshot
                contamination_fractions.append(contamination_fraction)

                lost_ids = initial_member_ids.difference(snapshot_member_ids)
                loss_fraction = len(lost_ids) / len(initial_member_ids)
                loss_fractions.append(loss_fraction)
                
                print(f"  → Snapshot {snap}: Contamination fraction = {contamination_fraction:.4f}, Loss fraction = {loss_fraction:.4f}")    
        
            
            
            snapshots = list(range(star_formation_snapshots[-1], end_snapshot + 1))
            # Loop through each snapshot in the cluster's lifetime
            for snap_idx in snapshots:
                
                # Get the member IDs of the cluster at this snapshot
                snapshot_member_ids = set(get_member_ids_for_fuzzycat_cluster(cluster_idx, ordering, fuzzy_clusters, workingDirectoryPath, snap_idx))

                if(len(snapshot_member_ids) == 0):
                    contamination_fractions.append(0.0)
                    loss_fractions.append(1.0)
                    continue
                
                # Calculate the contamination fraction
                differing_ids = snapshot_member_ids.difference(initial_member_ids)
                contamination_fraction = len(differing_ids) / len(snapshot_member_ids)
                # Store the contamination fraction for this snapshot
                contamination_fractions.append(contamination_fraction)

                lost_ids = initial_member_ids.difference(snapshot_member_ids)
                loss_fraction = len(lost_ids) / len(initial_member_ids)
                loss_fractions.append(loss_fraction)
                
                print(f"  → Snapshot {snap}: Contamination fraction = {contamination_fraction:.4f}, Loss fraction = {loss_fraction:.4f}")

            all_snapshots = star_formation_snapshots + snapshots
            # Store the contamination data for this cluster
            contamination_data[cluster_idx] = {
                'snapshots': all_snapshots,
                'contamination_fractions': contamination_fractions,
                'loss_fractions': loss_fractions,
            }
        contamination_data = {}
        for cluster_idx, metrics in burst_clusters.items():
            start_snapshot = metrics['cluster_start_snapshot']
            end_snapshot = metrics['cluster_end_snapshot']

            contamination_fractions = []
            loss_fractions = []

            #for given cluster in given snapshot, consider all stars born within 5 snapshots of their first appearance in that cluster as 'original members'
            for snap in range(start_snapshot, end_snapshot + 1):
                snapshot_member_ids = set(get_member_ids_for_fuzzycat_cluster(cluster_idx, ordering, fuzzy_clusters, workingDirectoryPath, snap))

                if len(snapshot_member_ids) == 0:
                    contamination_fractions.append(0.0)
                    loss_fractions.append(1.0)
                    continue
                # Get the member IDs of the cluster at this snapshot


                
        # # Save the contamination data to a file
        np.save(f"{contamination_analysis_dir}burst_clusters_contamination_data.npy", contamination_data, allow_pickle=True)

    contamination_data = np.load(f"{contamination_analysis_dir}burst_clusters_contamination_data.npy", allow_pickle=True).item()
    dir = f"{contamination_analysis_dir}contamination/"
    os.makedirs(dir, exist_ok=True)
    # Plot the contamination fraction over time for each burst cluster
    plt.figure(figsize=(12, 8))
    
    for cluster_idx, data in contamination_data.items():
        snapshots = np.array(data['snapshots'])
        contamination_fractions = np.array(data['contamination_fractions'])
        loss_fractions = np.array(data['loss_fractions'])
        no_info_mask = (loss_fractions == 1) & (contamination_fractions == 0)
        plot_cont = contamination_fractions.copy().astype(float)
        plot_loss = loss_fractions.copy().astype(float)
        plot_cont[no_info_mask] = np.nan  # Set no info points to NaN for better plotting
        plot_loss[no_info_mask] = np.nan

        fig, ax = plt.subplots()

        # --- 4. Draw the Translucent Gray Boxes ---
        # Get the indices of the snapshots with no info
        no_info_indices = np.where(no_info_mask)[0]

        if len(snapshots) > 1:
            # Calculate the typical distance between snapshots to set the box width
            snapshot_spacing = np.median(np.diff(snapshots))
        else:
            snapshot_spacing = 1.0 # A sensible default if there's only one point

        for i in no_info_indices:
            # Center the box on the snapshot
            left_edge = snapshots[i] - snapshot_spacing / 2
            right_edge = snapshots[i] + snapshot_spacing / 2
            ax.axvspan(left_edge, right_edge, color='gray', alpha=0.3, zorder=0, linewidth=0, edgecolor='none')

        # --- 5. Plot the Data ---
        ax.plot(snapshots, plot_cont, label=f'Contamination {cluster_idx}')
        ax.plot(snapshots, plot_loss, linestyle='--', label=f'Loss {cluster_idx}')

        # --- 6. Finalize the Plot ---
        ax.set_title("Contamination/Loss Fraction of Burst Clusters Over Time")
        ax.set_xlabel("Snapshot Index")
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.9)

        # Create a custom legend handle for the gray box
        legend_elements = [
            *ax.get_legend_handles_labels()[0], # Get existing lines
            Patch(facecolor='gray', alpha=0.3, label='No Information')
        ]
        ax.legend(handles=legend_elements)

        fig.tight_layout()
        
        # --- 7. Save and Close ---
        output_path = f"{dir}cluster_{cluster_idx}_contamination.png"
        plt.savefig(output_path, dpi=300)
        plt.close(fig) # Close the figure to free up memory
        print(f"Saved contamination analysis plot to {output_path}")

    # plot loss/contamination vs mass in last snapshot of cluster
    dir = f"{contamination_analysis_dir}contamination_vs_mass/"
    os.makedirs(dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster_idx, data in contamination_data.items():
        snapshots = np.array(data['snapshots'])
        contamination_fractions = np.array(data['contamination_fractions'])
        loss_fractions = np.array(data['loss_fractions'])
        
        # Get the last snapshot for this cluster
        last_snapshot = snapshots[-1]
        
        # Get the mass of the cluster at the last snapshot
        if cluster_idx in cluster_masses and last_snapshot in cluster_masses[cluster_idx]:
            mass = np.sum(cluster_masses[cluster_idx][last_snapshot])
            ax.scatter(mass, contamination_fractions[-1], label=f'Contamination {cluster_idx}', alpha=0.5)
            ax.scatter(mass, loss_fractions[-1], label=f'Loss {cluster_idx}', alpha=0.5, marker='x')
        
    ax.set_title(f"Contamination/Loss Fraction vs Cluster Mass at Last Snapshot of Cluster {cluster_idx}")
    ax.set_xlabel("Cluster Mass (Msol)")
    ax.set_ylabel("Fraction")
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.9)
    ax.legend()
    fig.tight_layout()
    
    output_path = f"{dir}contamination_loss_vs_mass_last_snapshot.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

                
    





    # #plot median contamination and loss fractions for all snapshots

    
    # fig, ax = plt.subplots()
    # snapshots = np.arange(earliest_start_snapshot, latest_end_snapshot + 1)
    # median_contamination = []
    # median_loss = []
    # for snap in snapshots:
    #     contamination_values = []
    #     loss_values = []
    #     for cluster_idx, data in contamination_data.items():
    #         if snap in data['snapshots']:
    #             idx = data['snapshots'].index(snap)
    #             contamination_values.append(data['contamination_fractions'][idx])
    #             loss_values.append(data['loss_fractions'][idx])
    #     if contamination_values:
    #         median_contamination.append(np.median(contamination_values))
    #     else:
    #         median_contamination.append(np.nan)
    #     if loss_values:
    #         median_loss.append(np.median(loss_values))
    #     else:
    #         median_loss.append(np.nan)

    # # Plot the median contamination and loss fractions
    # ax.plot(snapshots, median_contamination, label='Median Contamination', color='blue')
    # ax.plot(snapshots, median_loss, label='Median Loss', linestyle='--', color='orange')
    # ax.set_title("Median Contamination and Loss Fractions Over Time")
    # ax.set_xlabel("Snapshot Index")
    # ax.set_ylabel("Median Fraction")
    # ax.set_ylim(0, 1)
    # ax.grid(True, linestyle='--', alpha=0.9)
    # ax.legend()
    # fig.tight_layout()
    # output_path = f"{contamination_analysis_dir}median_contamination_loss_fractions.png"
    # plt.savefig(output_path, dpi=300)
    # plt.close(fig) # Close the figure to free up memory


def fast_identify_burst_clusters(snapshot_file_paths, working_directory_path, analysis_path, star_data_dir, n_snapshots, n_clusters, ordering, fuzzy_clusters, clustered_ids, threshold_fraction = 0.2):

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

    print("Finding potential star forming clusters...")
    clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")
    #get the snapshot ranges for each fuzzycat cluster
    fuzzy_cluster_snapshot_ranges = np.zeros((n_clusters, 2), dtype=int)
    for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):
        astrolink_cluster_ids_in_fuzzycat_cluster = ordering[start_idx:end_idx]
        cluster_filenames = [clusterFileNames[idx] for idx in astrolink_cluster_ids_in_fuzzycat_cluster]
        # Get the snapshot indices for this cluster
        snapshot_indices = set([int(filename.split('_')[0]) for filename in cluster_filenames])
        fuzzy_cluster_snapshot_ranges[cluster_idx] = [min(snapshot_indices), max(snapshot_indices)]

    csv_filepath = os.path.join(analysis_path, "cluster_formation_metrics_3.csv")
    header_columns = ['cluster_idx', 'cluster_start_snapshot', 'cluster_end_snapshot', 'star_count', 't25_t75', 'burst_snapshot', 'median_age', 'cluster_structure_age']
    processed_cluster_ids = set()
    file_exists = os.path.exists(csv_filepath)

    if file_exists:
        try:
            df_existing = pd.read_csv(csv_filepath)
            if not df_existing.empty and 'cluster_idx' in df_existing.columns:
                processed_cluster_ids = set(df_existing['cluster_idx'].astype(int).unique())
                print(f"Resuming. Found {len(processed_cluster_ids)} already processed cluster(s) in {csv_filepath}")
            elif df_existing.empty:
                print(f"File {csv_filepath} exists but is empty.")
            else: # Not empty, but missing 'cluster_idx'
                print(f"Warning: File {csv_filepath} exists but is missing 'cluster_idx'. Accurate resume not guaranteed.")
        except pd.errors.EmptyDataError:
            # This is expected if the file was created with a header but no data yet, or if it's just an empty file.
            print(f"File {csv_filepath} exists but is empty (or unreadable as CSV).")
        except Exception as e:
            # For other errors (e.g., malformed CSV), we might not be able to get processed_cluster_ids.
            # The script will then re-process items, potentially leading to duplicates if the file is not empty.
            print(f"Warning: Could not reliably read {csv_filepath} to check for processed clusters: {e}. May re-process items.")
    needs_header = not file_exists or (file_exists and os.path.getsize(csv_filepath) == 0)

    if needs_header:
        pd.DataFrame(columns=header_columns).to_csv(csv_filepath, index=False)
        print(f"Header written to {csv_filepath}")
    

    for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):
        
        print(f"Analyzing cluster {cluster_idx}/{len(fuzzy_clusters)}")
        if cluster_idx in processed_cluster_ids:
            print(f"Cluster {cluster_idx} already processed and in CSV. Skipping.")
            continue
        cluster_start_snapshot = fuzzy_cluster_snapshot_ranges[cluster_idx][0]
        cluster_end_snapshot = fuzzy_cluster_snapshot_ranges[cluster_idx][1]

        burst_recorded_for_this_cluster_idx_this_run = False

        for snap_idx in range(cluster_start_snapshot - 4, cluster_start_snapshot + 5):
            if snap_idx < 0 or snap_idx >= len(snapshot_file_paths):
                continue
            if burst_recorded_for_this_cluster_idx_this_run:
            # If we found and recorded a burst for this cluster_idx in the current run, we stop checking further snapshots for this cluster_idx.
                break
            snapshot_path = snapshot_file_paths[snap_idx]
            print(f"Processing snapshot {snap_idx}/{len(snapshot_file_paths)}")
            star_data = np.load(f"{star_data_dir}star_data_{snap_idx:03d}.npy")
            star_ids = star_data[:,0]
            star_ages = star_data[:,1]

            member_ids = get_member_ids_for_fuzzycat_cluster(cluster_idx, ordering, fuzzy_clusters, working_directory_path, snap_idx)

            
            star_ages_in_cluster = star_ages[np.isin(star_ids, member_ids)]
            
            if len(star_ages_in_cluster) == 0:
                continue  # No stars in this cluster at this snapshot
                
            # Calculate metrics
            sorted_ages = np.sort(star_ages_in_cluster)

            # Calculate t25-t75 (time to form middle 50% of stars)
            t25 = np.percentile(sorted_ages, 25)
            t75 = np.percentile(sorted_ages, 75)
            t25_t75 = t75 - t25

            median_age = np.median(star_ages_in_cluster)

            cluster_structure_age = (snap_idx - cluster_start_snapshot)*13.8/2000

            #Are we in snapshot range of cluster formation (ca. 1 Gyr after cluster start)?

            if t25_t75 < threshold_fraction and len(star_ages_in_cluster) > 50 and median_age < 1.5:
                #and np.abs(cluster_detected_time - snap_idx*13.8/2000) < 1.0:
            
              # Only consider clusters with median age < 1 Gyr
                metrics = {
                'cluster_idx': cluster_idx,
                'cluster_start_snapshot': cluster_start_snapshot,
                'cluster_end_snapshot': cluster_end_snapshot,
                'star_count': len(star_ages_in_cluster),
                't25_t75': t25_t75,
                'burst_snapshot': snap_idx,
                'median_age': median_age,
                'cluster_structure_age': cluster_structure_age
                }
                df_row = pd.DataFrame([metrics])
                df_row.to_csv(csv_filepath, mode='a', header=False, index=False)
                burst_recorded_for_this_cluster_idx_this_run = True


def new_identify_burst_clusters(snapshot_file_paths, working_directory_path, analysis_path, star_data_dir, n_snapshots, n_clusters, ordering, fuzzy_clusters, clustered_ids, threshold_fraction = 0.2):

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

    print("Finding potential star forming clusters...")
    clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")
    #get the snapshot ranges for each fuzzycat cluster
    fuzzy_cluster_snapshot_ranges = np.zeros((n_clusters, 2), dtype=int)
    for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):
        astrolink_cluster_ids_in_fuzzycat_cluster = ordering[start_idx:end_idx]
        cluster_filenames = [clusterFileNames[idx] for idx in astrolink_cluster_ids_in_fuzzycat_cluster]
        # Get the snapshot indices for this cluster
        snapshot_indices = set([int(filename.split('_')[0]) for filename in cluster_filenames])
        fuzzy_cluster_snapshot_ranges[cluster_idx] = [min(snapshot_indices), max(snapshot_indices)]

    csv_filepath = os.path.join(analysis_path, "cluster_formation_metrics_2.csv")
    header_columns = ['cluster_idx', 'cluster_start_snapshot', 'cluster_end_snapshot', 'star_count', 't25_t75', 'median_age', 'burst_snapshot']
    processed_cluster_ids = set()
    file_exists = os.path.exists(csv_filepath)

    if file_exists:
        try:
            df_existing = pd.read_csv(csv_filepath)
            if not df_existing.empty and 'cluster_idx' in df_existing.columns:
                processed_cluster_ids = set(df_existing['cluster_idx'].astype(int).unique())
                print(f"Resuming. Found {len(processed_cluster_ids)} already processed cluster(s) in {csv_filepath}")
            elif df_existing.empty:
                print(f"File {csv_filepath} exists but is empty.")
            else: # Not empty, but missing 'cluster_idx'
                print(f"Warning: File {csv_filepath} exists but is missing 'cluster_idx'. Accurate resume not guaranteed.")
        except pd.errors.EmptyDataError:
            # This is expected if the file was created with a header but no data yet, or if it's just an empty file.
            print(f"File {csv_filepath} exists but is empty (or unreadable as CSV).")
        except Exception as e:
            # For other errors (e.g., malformed CSV), we might not be able to get processed_cluster_ids.
            # The script will then re-process items, potentially leading to duplicates if the file is not empty.
            print(f"Warning: Could not reliably read {csv_filepath} to check for processed clusters: {e}. May re-process items.")
    needs_header = not file_exists or (file_exists and os.path.getsize(csv_filepath) == 0)

    if needs_header:
        pd.DataFrame(columns=header_columns).to_csv(csv_filepath, index=False)
        print(f"Header written to {csv_filepath}")
    

    for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):
        
        print(f"Analyzing cluster {cluster_idx}/{len(fuzzy_clusters)}")
        if cluster_idx in processed_cluster_ids:
            print(f"Cluster {cluster_idx} already processed and in CSV. Skipping.")
            continue
        cluster_start_snapshot = fuzzy_cluster_snapshot_ranges[cluster_idx][0]
        cluster_end_snapshot = fuzzy_cluster_snapshot_ranges[cluster_idx][1]

        snap_idx = cluster_start_snapshot

        snap_age = (snap_idx * 13.8 / 2000)  # Convert snapshot index to age in Gyr

        snapshot_path = snapshot_file_paths[snap_idx]
        print(f"Processing snapshot {snap_idx}/{len(snapshot_file_paths)}")
        star_data = np.load(f"{star_data_dir}star_data_{snap_idx:03d}.npy")
        star_ids = star_data[:,0]
        star_ages = star_data[:,1]

        member_ids = get_member_ids_for_fuzzycat_cluster(cluster_idx, ordering, fuzzy_clusters, working_directory_path, snap_idx)
        
        star_ages_in_cluster = star_ages[np.isin(star_ids, member_ids)]
        
        if len(star_ages_in_cluster) == 0:
            continue  # No stars in this cluster at this snapshot
            
        # Calculate metrics
        sorted_ages = np.sort(star_ages_in_cluster)

        median_age = np.median(star_ages_in_cluster)

        # Calculate t25-t75 (time to form middle 50% of stars)
        t25 = np.percentile(sorted_ages, 25)
        t75 = np.percentile(sorted_ages, 75)
        t25_t75 = t75 - t25
        

        #Are we in snapshot range of cluster formation (ca. 1 Gyr after cluster start)?

        if t25_t75 < threshold_fraction and len(star_ages_in_cluster) > 50 and median_age < 1.0:
        #and np.abs(cluster_detected_time - snap_idx*13.8/2000) < 1.0 
        
            # Only consider clusters with median age < 1 Gyr
            metrics = {
            'cluster_idx': cluster_idx,
            'cluster_start_snapshot': cluster_start_snapshot,
            'cluster_end_snapshot': cluster_end_snapshot,
            'star_count': len(star_ages_in_cluster),
            't25_t75': t25_t75,
            'median_age': median_age,
            'burst_snapshot': snap_idx
            }
            df_row = pd.DataFrame([metrics])
            df_row.to_csv(csv_filepath, mode='a', header=False, index=False)

def power_law_analysis(power_law_analysis_dir, n_snapshots, cluster_masses):
    """
    Analyses the power law slope of the burst clusters.
    We initially bin the cluster masses using the freedman-diaconis rule. 
    Then we assume a beta distribution for the number of clusters in each bin and perform a bootstrap analysis to estimate the slope and its error.
    We then plot the resulting slope and error for each snapshot.
    """

    all_snapshot_masses = cluster_masses
    nonempty_clusters = [c for c in all_snapshot_masses if any(len(m) > 0 for m in all_snapshot_masses[c].values())]
    results_snapshots = []
    results_mean_slopes = []
    results_slope_errors = []
    num_bootstraps = 1000
    make_individual_plots = True  # Set to False to skip individual plots and only save overview over slopes
    individual_plots_dir = f"{power_law_analysis_dir}individual_plots/"
    os.makedirs(individual_plots_dir, exist_ok=True)

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

    for final_snapshot_idx in range(n_snapshots):
        # cluster_masses_at_final_snap = []
        # #---- Mass power law analysis ---
        # for cluster_idx in nonempty_clusters:
        #     snap_masses = all_snapshot_masses[cluster_idx].get(final_snapshot_idx, np.array([]))
        #     if snap_masses.size > 0:
        #         Mcl = np.sum(snap_masses)
        #         cluster_masses_at_final_snap.append(Mcl)
        # cluster_masses = np.array(cluster_masses_at_final_snap)
        # print(f"Found a total of {len(cluster_masses)} clusters for initial mass function at snapshot {final_snapshot_idx}.")
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
            #num_bins = 10


            min_logM = np.log10(cluster_masses.min())
            max_logM = np.log10(cluster_masses.max())

            log_bins = np.linspace(min_logM, max_logM, num_bins + 1)  # Create bins in log scale

            linear_bins = 10**log_bins  # Convert back to linear scale for histogramming

            original_dN, _ = np.histogram(cluster_masses, bins=linear_bins)
            
            #bootstrap to estimate the error of the power law fit
            bootstrap_slopes = []
            bootstrap_y_intercepts = []
            dN_dM_array = []
            for i in range(num_bootstraps):
                resampled_dN = np.random.poisson(original_dN)

                log_bin_centers = 0.5 * (log_bins[:-1] + log_bins[1:])  # Centers of the log bins

                dM = linear_bins[1:] - linear_bins[:-1]  # Width of the bins in linear scale

                dM[dM==0] = 1e-9  # Avoid division by zero

                resampled_dN_dM = resampled_dN / dM

                mask = (10**log_bin_centers >= 1e4) & (resampled_dN > 0)  # Filter for Mcl ≥ 1e4 and dN > 0

                x_fit_data = log_bin_centers[mask]

                y_fit_data = np.log10(resampled_dN_dM[mask])
                
                if x_fit_data.size > 1:
                    slope, y_intercept = np.polyfit(x_fit_data, y_fit_data, 1)
                    bootstrap_slopes.append(slope)
                    bootstrap_y_intercepts.append(y_intercept)
                    dN_dM_array.append(resampled_dN_dM)
            
            if(len(bootstrap_slopes) > 15):
                slope = np.mean(bootstrap_slopes)
                slope_error = np.std(bootstrap_slopes)
                y_intercept = np.mean(bootstrap_y_intercepts)
                y_intercept_error = np.std(bootstrap_y_intercepts)
                dN_dM = np.mean(dN_dM_array, axis=0)  # Average over bootstrap samples
                results_snapshots.append(final_snapshot_idx)
                results_mean_slopes.append(slope)
                results_slope_errors.append(slope_error)
                #print(f"Estimated power law slope for cluster masses at snapshot {final_snapshot_idx}: {slope:.2f} ± {slope_error:.2f}")
                if not make_individual_plots:
                    continue
                # Plotting the cluster mass function for every snapshot
                x_fit_line = np.linspace(min_logM, max_logM, 100)  # Fit line for plotting
                y_fit_line = slope * x_fit_line + y_intercept  # Linear fit line in log-log space
                y_err_propagated = np.sqrt((x_fit_line * slope_error)**2 + y_intercept_error**2)
                y_upper = y_fit_line + y_err_propagated
                y_lower = y_fit_line - y_err_propagated
                plt.figure(figsize=(10, 6))
                plt.fill_between(10**x_fit_line, 10**y_lower, 10**y_upper, color='red', alpha=0.2, label='1-$\sigma$ Uncertainty')
                plt.scatter(10**log_bin_centers[mask], dN_dM[mask], color='blue', label='Simulation Data', s=10)
                plt.plot(10**x_fit_line, 10**y_fit_line, 'r--', color='red', label=f"Fit: slope = {slope:.2f}")
                plt.grid(True, linestyle='--', alpha=0.5, axis='both')
                plt.legend()    
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel("Cluster Mass")
                plt.ylabel("Number of Clusters per delta log(M)")
                plt.title("Cluster Mass Function")
                plt.savefig(f"{individual_plots_dir}cluster_mass_function_snapshot{final_snapshot_idx}.png", dpi=200)
                #plt.savefig(f"{individual_plots_dir}cluster_mass_function_overall.png", dpi=200)
                plt.close()
            #print(f"Saved cluster mass function plot for snapshot {final_snapshot_idx} to {individual_plots_dir}cluster_mass_function_snapshot{final_snapshot_idx}.png")
            else:
                print("Not enough data points to fit a power law for cluster masses at the final snapshot.")

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

    # --- Create combined Plot ---
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Plot the main trend line ---
    # Use a solid line with small, subtle markers.
    ax.plot(snapshots, slopes, 
            marker='o',          # Small circle markers at each data point
            markersize=5,        # Control the size of the markers
            linestyle='-',       # A solid line connecting the points
            color='C0',          # Use the default blue color
            label='Slope (α)')

    # --- Plot the shaded error region ---
    # This is the key function: plt.fill_between
    ax.fill_between(snapshots,           # X-values
                    slopes - errors,     # The lower boundary of the shaded region
                    slopes + errors,     # The upper boundary of the shaded region
                    color='C0',          # Match the line color
                    alpha=0.2,           # Use transparency to make it subtle
                    label='Binning Uncertainty')

    # --- Add a reference line for the theoretical initial slope ---
    ax.axhline(-2.0, 
            color='red', 
            linestyle='--', 
            linewidth=2,
            label='Theoretical Slope (α = -2.0)')

    ax.set_xlabel("Snapshot Index", fontsize=14)
    ax.set_ylabel("Mass Function Slope (α)", fontsize=14)
    ax.set_title("Evolution of the Cluster Mass Function Slope", fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(snapshots.min() - 1, snapshots.max() + 1)
    ax.set_ylim(-2.1, -0.5) # Adjust this based on your data's range
    ax.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{power_law_analysis_dir}slope_evolution_plot.png", dpi=300)


def power_law_analysis_2(working_directory_path, star_data_dir, power_law_dir, n_snapshots, ordering, fuzzy_clusters):
    n_clusters = len(fuzzy_clusters)
    print(f"Found {n_clusters} fuzzycat clusters in the data.")
    cluster_masses_all_snapshots = { c: {} for c in range(n_clusters)}
    clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")
    #get the snapshot ranges for each fuzzycat cluster
    fuzzy_cluster_snapshot_ranges = np.zeros((n_clusters, 2), dtype=int)
    # for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):
    #     print(f"Processing FuzzyCat cluster {cluster_idx}")
    #     astrolink_cluster_ids_in_fuzzycat_cluster = ordering[start_idx:end_idx]
    #     cluster_filenames = [clusterFileNames[idx] for idx in astrolink_cluster_ids_in_fuzzycat_cluster]
    #     # Get the snapshot indices for this cluster
    #     snapshot_indices = set([int(filename.split('_')[0]) for filename in cluster_filenames])
    #     fuzzy_cluster_snapshot_ranges[cluster_idx] = [min(snapshot_indices), max(snapshot_indices)]

    
    #     for snap_idx in range(fuzzy_cluster_snapshot_ranges[cluster_idx][0], fuzzy_cluster_snapshot_ranges[cluster_idx][1] + 1):
    #         star_data = np.load(f"{star_data_dir}star_data_{snap_idx:03d}.npy")
    #         star_ids = star_data[:,0]
    #         star_masses = star_data[:,2]

    #         member_ids = get_member_ids_for_fuzzycat_cluster(
    #             cluster_idx, ordering, fuzzy_clusters, workingDirectoryPath, snap_idx
    #         )
            
    #         star_masses_in_cluster = star_masses[np.isin(star_ids, member_ids)]
    #         cluster_masses_all_snapshots[cluster_idx][snap_idx] = star_masses_in_cluster

    # np.save(f"{power_law_dir}cluster_masses_all_snapshots.npy", cluster_masses_all_snapshots, allow_pickle=True)
    print(f"Found {len(clusterFileNames)} AstroLink clusters")
    #get all masses for every astrolink cluster
    all_cluster_masses = {}
    for clusterfilename in clusterFileNames:

        # Get the snapshot index from the filename
        snap_idx = int(clusterfilename.split('_')[0])
        # Get the cluster index from the filename
        cluster_idx = clusterfilename.split('_')[1].split('.')[0]
        print(f"Processing AstroLink cluster {cluster_idx} for snapshot {snap_idx}...")
        
        # Load the cluster data
        ids_in_cluster = np.load(f"{working_directory_path}Clusters_iord/{clusterfilename}")
        
        # Load the star data for this snapshot
        star_data = np.load(f"{star_data_dir}star_data_{snap_idx:03d}.npy")
        star_ids = star_data[:,0]
        star_masses = star_data[:,2]
        
        # Get the masses of the stars in this cluster
        member_masses = star_masses[np.isin(star_ids, ids_in_cluster)]
        
        if len(member_masses) > 0:
            all_cluster_masses.setdefault(cluster_idx, {})[snap_idx] = member_masses

    # Save
    np.save(f"{power_law_dir}cluster_masses_all_snapshots.npy", all_cluster_masses, allow_pickle=True)


    all_snapshot_masses = np.load(f"{power_law_dir}cluster_masses_all_snapshots.npy", allow_pickle=True).item()

    nonempty_clusters = [c for c in all_snapshot_masses if any(len(m) > 0 for m in all_snapshot_masses[c].values())]
    results_snapshots = []
    results_mean_slopes = []
    results_slope_errors = []
    num_bootstraps = 1000
    make_individual_plots = True  # Set to False to skip individual plots and only save overview over slopes
    individual_plots_dir = f"{power_law_dir}individual_plots/"
    os.makedirs(individual_plots_dir, exist_ok=True)

    # peak_cluster_masses = []
    # for cluster_idx in nonempty_clusters:
    #     # Get all mass measurements for this one cluster across all snapshots it exists in
    #     masses_over_time = all_snapshot_masses[cluster_idx].values()
        
    #     # Calculate the total cluster mass at each of those snapshots
    #     total_masses_per_snap = [np.sum(m) for m in masses_over_time if m.size > 0]
        
    #     # If the cluster had any mass at any point, find its maximum mass
    #     if total_masses_per_snap:
    #         peak_mass = np.max(total_masses_per_snap)
    #         median_mass = np.median(total_masses_per_snap)
    #         peak_cluster_masses.append(median_mass)

    # This is now our primary dataset for the CMF analysis
    # cluster_masses = np.array(peak_cluster_masses)
    # print(f"Found a total of {len(cluster_masses)} unique clusters across all snapshots.")

    for final_snapshot_idx in range(n_snapshots):
        cluster_masses_at_final_snap = []
        #---- Mass power law analysis ---
        for cluster_idx in nonempty_clusters:
            snap_masses = all_snapshot_masses[cluster_idx].get(final_snapshot_idx, np.array([]))
            if snap_masses.size > 0:
                Mcl = np.sum(snap_masses)
                cluster_masses_at_final_snap.append(Mcl)
        cluster_masses = np.array(cluster_masses_at_final_snap)
        print(f"Found a total of {len(cluster_masses)} clusters for initial mass function at snapshot {final_snapshot_idx}.")
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
            num_bins = max(5, min(num_bins, 100))  # Ensure num_bins is between 5 and 50
            #num_bins = 10  # Ensure num_bins is between 5 and 50
            min_logM = np.log10(cluster_masses.min())
            max_logM = np.log10(cluster_masses.max())
            log_bins = np.linspace(min_logM, max_logM, num_bins + 1)  # Create bins in log scale
            linear_bins = 10**log_bins  # Convert back to linear scale for histogramming
            original_dN, _ = np.histogram(cluster_masses, bins=linear_bins)
            

            #bootstrap to estimate the error of the power law fit
            bootstrap_slopes = []
            bootstrap_y_intercepts = []
            dN_dM_array = []
            
            for i in range(num_bootstraps):
                resampled_dN = np.random.poisson(original_dN)
                log_bin_centers = 0.5 * (log_bins[:-1] + log_bins[1:])  # Centers of the log bins
                dM = linear_bins[1:] - linear_bins[:-1]  # Width of the bins in linear scale
                dM[dM==0] = 1e-9  # Avoid division by zero
                resampled_dN_dM = resampled_dN / dM
                mask = (resampled_dN > 0)  # Filter for Mcl ≥ 1e6 and dN > 0
                x_fit_data = log_bin_centers[mask]
                y_fit_data = np.log10(resampled_dN_dM[mask])
                if x_fit_data.size > 1:
                    slope, y_intercept = np.polyfit(x_fit_data, y_fit_data, 1)
                    bootstrap_slopes.append(slope)
                    bootstrap_y_intercepts.append(y_intercept)
                    dN_dM_array.append(resampled_dN_dM)
            
            if(len(bootstrap_slopes) > 15):
                slope = np.mean(bootstrap_slopes)
                slope_error = np.std(bootstrap_slopes)
                y_intercept = np.mean(bootstrap_y_intercepts)
                y_intercept_error = np.std(bootstrap_y_intercepts)
                dN_dM = np.mean(dN_dM_array, axis=0)  # Average over bootstrap samples
                results_snapshots.append(final_snapshot_idx)
                results_mean_slopes.append(slope)
                results_slope_errors.append(slope_error)
                print(f"Estimated power law slope for cluster masses at snapshot {final_snapshot_idx}: {slope:.2f} ± {slope_error:.2f}")
                if not make_individual_plots:
                    continue
                # Plotting the cluster mass function for every snapshot
                x_fit_line = np.linspace(min_logM, max_logM, 100)  # Fit line for plotting
                y_fit_line = slope * x_fit_line + y_intercept  # Linear fit line in log-log space
                y_err_propagated = np.sqrt((x_fit_line * slope_error)**2 + y_intercept_error**2)
                y_upper = y_fit_line + y_err_propagated
                y_lower = y_fit_line - y_err_propagated

                plt.figure(figsize=(10, 6))
                plt.fill_between(10**x_fit_line, 10**y_lower, 10**y_upper, color='red', alpha=0.2, label='1-$\sigma$ Uncertainty')
                plt.scatter(10**log_bin_centers[mask], dN_dM[mask], color='blue', label='Simulation Data', s=10)
                plt.plot(10**x_fit_line, 10**y_fit_line, 'r--', color='red', label=f"Fit: slope = {slope:.2f} ± {slope_error:.2f}")
                plt.grid(True, linestyle='--', alpha=0.5, axis='both')
                plt.legend()    
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel("Cluster Mass")
                plt.ylabel("Number of Clusters per delta log(M)")
                plt.title("Cluster Mass Function")
                plt.savefig(f"{individual_plots_dir}cluster_mass_function_snapshot{final_snapshot_idx}.png", dpi=200)
                #plt.savefig(f"{individual_plots_dir}cluster_mass_function_overall.png", dpi=200)
                plt.close()
                #print(f"Saved cluster mass function plot for snapshot {final_snapshot_idx} to {individual_plots_dir}cluster_mass_function_snapshot{final_snapshot_idx}.png")
            else:
                print("Not enough data points to fit a power law for cluster masses at the final snapshot.")


def std_deviation_age_analysis(cluster_metrics, cluster_masses, std_deviation_analysis_dir, working_directory_path, star_data_dir, ordering, fuzzy_clusters):
    """Take std(age) as metric for spread of ages in cluster, plot it vs M, R, Z, Contamination, Loss"""
    
    contamination_data = np.load(f"{working_directory_path}star_cluster_analysis_3/burst_cluster_analysis/contamination_analysis_5/burst_clusters_contamination_data.npy", allow_pickle=True).item()

    if not os.path.exists(f"{std_deviation_analysis_dir}std_deviations.npy"):
        all_std_deviations = []
        all_masses = []
        all_radial_distances = []
        all_irons = []
        all_z_distances = []
        all_contamination_fractions = []
        all_loss_fractions = []

        for cluster_idx, metrics in cluster_metrics.items():

            print(f"Processing cluster {cluster_idx} for standard deviation age analysis")
            snap_idx = int(metrics['cluster_start_snapshot'])

            star_data = np.load(f"{star_data_dir}star_data_{snap_idx:03d}.npy")
            star_ids = star_data[:,0]
            star_ages = star_data[:,1]

            member_ids = get_member_ids_for_fuzzycat_cluster(cluster_idx, ordering, fuzzy_clusters, working_directory_path, snap_idx)

            star_ages_in_cluster = star_ages[np.isin(star_ids, member_ids)]

            standard_deviation_age = np.std(star_ages_in_cluster)

            all_std_deviations.append(standard_deviation_age)

            all_masses.append(np.sum(cluster_masses[cluster_idx][snap_idx]))
            all_radial_distances.append(metrics['median_radial_distance'])
            all_irons.append(metrics['median_iron'])
            all_z_distances.append(metrics['median_z_distance'])

            all_loss_fractions.append(contamination_data[cluster_idx]['loss_fractions'][-1])
            all_contamination_fractions.append(contamination_data[cluster_idx]['contamination_fractions'][-1])

        # Convert lists to numpy arrays for easier manipulation

        all_std_deviations = np.array(all_std_deviations)
        all_masses = np.array(all_masses)
        all_radial_distances = np.array(all_radial_distances)
        all_irons = np.array(all_irons)
        all_z_distances = np.array(all_z_distances)
        all_contamination_fractions = np.array(all_contamination_fractions)
        all_loss_fractions = np.array(all_loss_fractions)

        # Save the data for further analysis if needed
        np.save(f"{std_deviation_analysis_dir}std_deviations.npy", all_std_deviations)
        np.save(f"{std_deviation_analysis_dir}masses.npy", all_masses)
        np.save(f"{std_deviation_analysis_dir}radial_distances.npy", all_radial_distances)
        np.save(f"{std_deviation_analysis_dir}irons.npy", all_irons)
        np.save(f"{std_deviation_analysis_dir}z_distances.npy", all_z_distances)
        np.save(f"{std_deviation_analysis_dir}contamination_fractions.npy", all_contamination_fractions)
        np.save(f"{std_deviation_analysis_dir}loss_fractions.npy", all_loss_fractions)
    
    all_loss_fractions = []
    all_contamination_fractions = []
    for cluster_idx, metrics in cluster_metrics.items():
        if cluster_idx not in contamination_data:
            print(f"Cluster {cluster_idx} not found in contamination data. Skipping.")
            continue
        all_loss_fractions.append(contamination_data[cluster_idx]['loss_fractions'][-1])
        all_contamination_fractions.append(contamination_data[cluster_idx]['contamination_fractions'][-1])

        

    all_std_deviations = np.load(f"{std_deviation_analysis_dir}std_deviations.npy")
    all_masses = np.load(f"{std_deviation_analysis_dir}masses.npy")
    all_radial_distances = np.load(f"{std_deviation_analysis_dir}radial_distances.npy")
    all_irons = np.load(f"{std_deviation_analysis_dir}irons.npy")
    all_z_distances = np.load(f"{std_deviation_analysis_dir}z_distances.npy")
    all_contamination_fractions = np.load(f"{std_deviation_analysis_dir}contamination_fractions.npy")
    all_loss_fractions = np.load(f"{std_deviation_analysis_dir}loss_fractions.npy")

    contamination_mask = ~((all_loss_fractions == 1.0) | (all_contamination_fractions == 0.0))
    all_std_deviations_masked = all_std_deviations[contamination_mask]
    all_contamination_fractions = all_contamination_fractions[contamination_mask]
    all_loss_fractions = all_loss_fractions[contamination_mask]
    
    # Create scatter plots for std deviation vs mass, radial distance, iron metallicity, contamination and loss
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    


    # coeffs = np.polyfit(all_std_deviations, np.log10(all_masses), 1)
    # poly_fit_func = np.poly1d(coeffs)
    # x_trend = np.linspace(min(all_std_deviations), max(all_std_deviations), 100)
    # y_trend = 10**poly_fit_func(x_trend)
    
    linregress_result = stats.linregress(all_std_deviations, np.log10(all_masses))
    slope = linregress_result.slope
    intercept = linregress_result.intercept
    r_value = linregress_result.rvalue

    x = np.linspace(0, np.max(all_std_deviations) * 1.2, 100)
    y = 10**(slope * x + intercept)


    ax[0].scatter(all_std_deviations, all_masses)
    ax[0].set_title("Standard Deviation of Ages vs Cluster Mass")
    ax[0].set_xlabel("Standard Deviation of Ages (Gyr)")
    ax[0].set_ylabel("Cluster Mass (Msol)")
    ax[0].set_yscale('log')
    ax[0].grid(True, linestyle='--', alpha=0.9)
    ax[0].plot(x, y, "r--", linewidth=2, label="Linear Fit: $y = {:.2f}x + {:.2f}$\n$R^2 = {:.2f}$".format(slope, intercept, r_value**2))
    ax[0].legend()

    # ax[0, 1].scatter(all_std_deviations, all_radial_distances )
    # ax[0, 1].set_title("Standard Deviation of Ages vs Median Radial Distance in birth snapshot")
    # ax[0, 1].set_xlabel("Standard Deviation of Ages (Gyr)")
    # ax[0, 1].set_ylabel("Median Radial Distance (kpc)")
    # ax[0, 1].grid(True, linestyle='--', alpha=0.9)

    # ax[1, 0].scatter( all_std_deviations,all_irons)
    # ax[1, 0].set_title("Standard Deviation of Ages vs Median Iron Metallicity in birth snapshot")
    # ax[1, 0].set_xlabel("Standard Deviation of Ages (Gyr)")
    # ax[1, 0].set_ylabel("Median Iron Metallicity [Fe/H]")
    # ax[1, 0].grid(True, linestyle='--', alpha=0.9)
    
    ax[1].scatter(all_std_deviations_masked, all_contamination_fractions, alpha=0.7, label='Contamination Fraction')
    ax[1].scatter(all_std_deviations_masked, all_loss_fractions, alpha=0.7, label='Loss Fraction', color='orange')
    ax[1].set_title("Standard Deviation of Ages vs Contamination and Loss Fractions")
    ax[1].set_xlabel("Standard Deviation of Ages (Gyr)")
    ax[1].set_ylabel("Fraction")
    ax[1].set_xlim(0, np.max(all_std_deviations) * 1.2)
    ax[1].grid(True, linestyle='--', alpha=0.9)
    ax[1].legend()

    # ax[1, 0].scatter(all_std_deviations, all_z_distances)
    # ax[1, 0].set_title("Standard Deviation of Ages vs Median Z Distance in birth snapshot")
    # ax[1, 0].set_xlabel("Standard Deviation of Ages (Gyr)")
    # ax[1, 0].set_ylabel("Median Z Distance (kpc)")
    # ax[1, 0].grid(True, linestyle='--', alpha=0.9)


    fig.suptitle("Standard Deviation of Ages in Star Forming Clusters", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{std_deviation_analysis_dir}std_deviation_age_analysis.png", dpi=300)
    plt.close(fig)

def mean_age_distribution_analysis(burst_clusters, snapshot_file_paths, working_directory_path, analysis_dir, star_data_dir, n_snapshots, ordering, fuzzy_clusters):


    """
    Analyze the mean age distribution of star forming clusters during their burst snapshot.
    Compare that to the mean age distribution of non-star forming clusters during their detection snapshot.
    """

    age_distribution_analysis_dir = f"{analysis_dir}age_distribution_analysis/"
    os.makedirs(age_distribution_analysis_dir, exist_ok=True)

    # Initialize a dictionary to store the mean ages of star forming clusters
    median_ages_star_forming = {}
    median_ages_non_star_forming = {}

    clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")

    # Loop through each burst cluster
#     for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):

#         if cluster_idx not in burst_clusters:
#             astrolink_cluster_ids_in_fuzzycat_cluster = ordering[start_idx:end_idx]
#             cluster_filenames = [clusterFileNames[idx] for idx in astrolink_cluster_ids_in_fuzzycat_cluster]
#             # Get the snapshot indices for this cluster
#             snapshot_indices = set([int(filename.split('_')[0]) for filename in cluster_filenames])
#             detection_snapshot = min(snapshot_indices)
#             print(f"Cluster {cluster_idx} is not a burst cluster, using detection snapshot {detection_snapshot} for mean age analysis.")
#             # Load the star data for the detection snapshot
#             star_data_detection = np.load(f"{star_data_dir}star_data_{detection_snapshot:03d}.npy")
#             star_ids_detection = star_data_detection[:, 0]
#             star_ages_detection = star_data_detection[:, 1]  # Assuming ages are in column 1
#             # Get member IDs for the current cluster at detection snapshot
#             member_ids_detection = get_member_ids_for_fuzzycat_cluster(cluster_idx, ordering, fuzzy_clusters, working_directory_path, detection_snapshot)
#             # Calculate mean age for non-star forming clusters at detection snapshot
#             if len(member_ids_detection) > 0:
#                 ages_in_detection_cluster = star_ages_detection[np.isin(star_ids_detection, member_ids_detection)]
#                 if len(ages_in_detection_cluster) > 0:
#                     median_age_non_star_forming = np.median(ages_in_detection_cluster)
#                     sorted_ages = np.sort(ages_in_detection_cluster)
#                     t25_75 = np.percentile(sorted_ages, 75) - np.percentile(sorted_ages, 25)
#                     #mean_ages_non_star_forming[cluster_idx] = [mean_age_non_star_forming, len(ages_in_detection_cluster), t25_75]
#                     median_ages_non_star_forming[cluster_idx] = median_age_non_star_forming
                    
#                 else:
#                     print(f"No stars found in non-star forming cluster {cluster_idx} at snapshot {detection_snapshot}")
#                     median_ages_non_star_forming[cluster_idx] = np.nan

#         else:    
#             # Get the burst snapshot and the detection snapshot
#             burst_snapshot = burst_clusters[cluster_idx]['burst_snapshot']

#             print(f"Analyzing burst cluster {cluster_idx} at snapshot {burst_snapshot}...")
#             # Load the star data for the burst snapshot
#             star_data_burst = np.load(f"{star_data_dir}star_data_{burst_snapshot:03d}.npy")
#             star_ids_burst = star_data_burst[:, 0]
#             star_ages_burst = star_data_burst[:, 1]  # Assuming ages are in column 1

#             # Get member IDs for the current cluster at both snapshots
#             member_ids_burst = get_member_ids_for_fuzzycat_cluster(cluster_idx, ordering, fuzzy_clusters, working_directory_path, burst_snapshot)

#             # Calculate mean age for star forming clusters at burst snapshot
#             if len(member_ids_burst) > 0:
#                 ages_in_burst_cluster = star_ages_burst[np.isin(star_ids_burst, member_ids_burst)]
#                 if len(ages_in_burst_cluster) > 0:
#                     median_age_star_forming = np.median(ages_in_burst_cluster)
#                     median_ages_star_forming[cluster_idx] = median_age_star_forming
#                 else:
#                     print(f"No stars found in burst cluster {cluster_idx} at snapshot {burst_snapshot}")
#                     median_ages_star_forming[cluster_idx] = np.nan
#             else:
#                 print(f"No member IDs found for burst cluster {cluster_idx} at snapshot {burst_snapshot}")
#                 median_ages_star_forming[cluster_idx] = np.nan

#     potentially_star_forming_clusters = {
#     cluster_idx: stats
#     for cluster_idx, stats in mean_ages_non_star_forming.items()
#     if not np.isnan(stats[0]) and stats[0] < 1.0
# }

#     for cluster_idx in potentially_star_forming_clusters:
#         print(f"Cluster {cluster_idx} is potentially star forming ")

        
#     np.save(f"{age_distribution_analysis_dir}mean_ages_star_forming.npy", potentially_star_forming_clusters, allow_pickle=True)
#     # # Convert the mean ages to numpy arrays for easier manipulation
#     median_ages_star_forming = np.array(list(median_ages_star_forming.values()))
#     median_ages_non_star_forming = np.array(list(median_ages_non_star_forming.values()))



    np.save(f"{age_distribution_analysis_dir}median_ages_star_forming_2.npy", median_ages_star_forming)
    np.save(f"{age_distribution_analysis_dir}median_ages_non_star_forming_2.npy", median_ages_non_star_forming)



    # Load the mean ages from the saved files
    median_ages_star_forming = np.load(f"{age_distribution_analysis_dir}median_ages_star_forming_2.npy", allow_pickle=True)
    median_ages_non_star_forming = np.load(f"{age_distribution_analysis_dir}median_ages_non_star_forming_2.npy", allow_pickle=True)

    # potentially_star_forming_clusters = [age for age in mean_ages_non_star_forming if not np.isnan(age) and age < 1.0]
    # print(f"Found {len(potentially_star_forming_clusters)} potentially star forming clusters with mean age < 1.0 Gyr in non-star forming clusters.")

    #plot the mean age distributions
    plt.figure(figsize=(12, 6))
    sns.histplot(median_ages_star_forming, bins=10, color='blue', label='Star Forming Clusters', alpha=0.6)
    sns.histplot(median_ages_non_star_forming, bins=90, color='orange', label='Non-Star Forming Clusters', alpha=0.4)
    plt.title("Median Age Distribution of Star Forming vs Non-Star Forming Clusters")
    plt.xlabel("Median Age (Gyr)")
    plt.ylabel("Cluster Count")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{age_distribution_analysis_dir}new_median_age_distribution_star_vs_non_star_forming_absolute_count.png", dpi=300)
    plt.close()
