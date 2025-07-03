import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy import stats

def plot_cluster_mass_distribution(data_path):

    data_path_dyn = f"{data_path}/dynamical_tagging/slope_evolution.txt"
    data_path_chemdyn = f"{data_path}/chemodynamical_tagging/slope_evolution.txt"
    data_dyn = np.loadtxt(data_path_dyn)
    data_chemdyn = np.loadtxt(data_path_chemdyn)

    # Unpack the columns
    snapshots_dyn = data_dyn[:, 0]
    slopes_dyn = data_dyn[:, 1]
    errors_dyn = data_dyn[:, 2]

    mean_slopes_dyn = np.mean(slopes_dyn)
    mean_errors_dyn = np.mean(errors_dyn)

    snapshots_chemdyn = data_chemdyn[:, 0]
    slopes_chemdyn = data_chemdyn[:, 1]
    errors_chemdyn = data_chemdyn[:, 2]
    mean_slopes_chemdyn = np.mean(slopes_chemdyn)
    mean_errors_chemdyn = np.mean(errors_chemdyn)

    # --- Create combined Plot ---
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Plot the main trend line ---
    # Use a solid line with small, subtle markers.
    ax.plot(snapshots_dyn, slopes_dyn, 
            marker='o',          # Small circle markers at each data point
            markersize=5,        # Control the size of the markers
            linestyle='-',       # A solid line connecting the points
            color='C0',          # Use the default blue color
            label=f'Slope dynamical (α): {mean_slopes_dyn:.2f} ± {mean_errors_dyn:.2f}')  # Add mean slope and error to the label

    # --- Plot the shaded error region ---
    # This is the key function: plt.fill_between
    ax.fill_between(snapshots_dyn,           # X-values
                    slopes_dyn - errors_dyn,     # The lower boundary of the shaded region
                    slopes_dyn + errors_dyn,     # The upper boundary of the shaded region
                    color='C0',          # Match the line color
                    alpha=0.2)           # Use transparency to make it subtle


    ax.plot(snapshots_chemdyn, slopes_chemdyn, 
            marker='o',          # Small circle markers at each data point
            markersize=5,        # Control the size of the markers
            linestyle='-',       # A solid line connecting the points
            color='C2',          # Use the default blue color
            label=f'Slope chemodynamical (α): {mean_slopes_chemdyn:.2f} ± {mean_errors_chemdyn:.2f}')  # Add mean slope and error to the label

    # --- Plot the shaded error region ---
    # This is the key function: plt.fill_between
    ax.fill_between(snapshots_chemdyn,           # X-values
                    slopes_chemdyn - errors_chemdyn,     # The lower boundary of the shaded region
                    slopes_chemdyn + errors_chemdyn,     # The upper boundary of the shaded region
                    color='C2',          # Match the line color
                    alpha=0.2)           # Use transparency to make it subtle

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
    ax.set_xlim(snapshots_dyn.min() - 1, snapshots_dyn.max() + 1)
    ax.set_ylim(-2.1, -1.0) # Adjust this based on your data's range
    ax.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{data_path}/IMF_evolution_comparison.png", dpi=300)

def spacial_distribution_analysis(cluster_data, dir):

    """
    Analyzes the spatial distribution of burst clusters in the simulation.
    At what radius from the center of the disk do the clusters form?
    How far out of the disk vertically are they?
    """

    # Plot the spatial distribution of clusters
    radial_distances = [data['median_radial_distance'] for data in cluster_data.values()]
    vertical_distances = [data['median_z_distance'] for data in cluster_data.values()]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

    # Create a figure with two subplots, side by side
    # figsize controls the overall size of the figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    ticks_linear = list(range(0,4))  # in linear region (you can adjust)
    ticks_log = [10,20,30]  # for the log region
    all_ticks = ticks_linear + ticks_log

    # --- Plot 1: Radial Distribution ---
    ax1 = axes[0]
    sns.histplot(data=radial_distances, ax=ax1, color='royalblue')

    ax1.set_title("Radial Distribution of Clusters")
    ax1.set_xlabel("Median Radial Distance (kpc)")
    ax1.set_ylabel("Cluster Count")

    # Add a vertical line for the median of the distribution
    median_r = np.median(radial_distances)
    ax1.axvline(median_r, color='red', linestyle='--', linewidth=2, label=f'Median: {median_r:.2f} kpc')
    ax1.legend()


    # --- Plot 2: Vertical Distribution ---
    ax2 = axes[1]
    sns.histplot(data=vertical_distances, ax=ax2, color='seagreen')
    ax2.set_xscale('symlog', linthresh=3)
    ax2.set_xticks(all_ticks, [str(tick) for tick in ticks_linear] + [str(tick) for tick in ticks_log])
    ax2.set_title("Vertical Distribution of Clusters")
    ax2.set_xlabel("Median Vertical Distance |z| (kpc)")
    ax2.set_ylabel("Cluster Count")
    ax2.margins(0.01)

    # Add a vertical line for the median of the distribution
    median_z = np.median(vertical_distances)
    ax2.axvline(median_z, color='red', linestyle='--', linewidth=2, label=f'Median: {median_z:.2f} kpc')
    ax2.legend()

    # Add a main title for the entire figure
    fig.suptitle("Spatial Distribution of Star Clusters in the Galactic Disk", fontsize=20, y=1.02)

    # Adjust layout to prevent titles/labels from overlapping
    plt.tight_layout()
    plt.savefig(f"{dir}cluster_spatial_distribution_count_2.png", dpi=300, bbox_inches='tight')

def power_law_analysis_few_clusters(power_law_analysis_dir, n_snapshots, cluster_masses):
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

    #---Use this for analysis of the final snapshot only (many total clusters)---

    # for snapshot_idx in range(n_snapshots):
    #     cluster_masses_at_final_snap = []
    #     #---- Mass power law analysis ---
    #     for cluster_idx in nonempty_clusters:
    #         snap_masses = all_snapshot_masses[cluster_idx].get(snapshot_idx, np.array([]))
    #         if snap_masses.size > 0:
    #             Mcl = np.sum(snap_masses)
    #             cluster_masses_at_final_snap.append(Mcl)
    #     cluster_masses = np.array(cluster_masses_at_final_snap)
    #     print(f"Found a total of {len(cluster_masses)} clusters for initial mass function at snapshot {snapshot_idx}.")
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
            #results_snapshots.append(snapshot_idx)
            results_mean_slopes.append(slope)
            results_slope_errors.append(slope_error)
            #print(f"Estimated power law slope for cluster masses at snapshot {final_snapshot_idx}: {slope:.2f} ± {slope_error:.2f}")
            # if not make_individual_plots:
            #     continue
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
            #plt.savefig(f"{individual_plots_dir}cluster_mass_function_snapshot{final_snapshot_idx}.png", dpi=200)
            plt.savefig(f"{individual_plots_dir}cluster_mass_function_overall.png", dpi=200)
            plt.close()
        #print(f"Saved cluster mass function plot for snapshot {final_snapshot_idx} to {individual_plots_dir}cluster_mass_function_snapshot{final_snapshot_idx}.png")
        else:
            print("Not enough data points to fit a power law for cluster masses at the final snapshot.")

def power_law_analysis_many_clusters(power_law_analysis_dir, n_snapshots, cluster_masses):
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

    #---Use this for analysis of all snapshots at once (few total clusters)---
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

    # # This is now our primary dataset for the CMF analysis
    # cluster_masses = np.array(peak_cluster_masses)
    # print(f"Found a total of {len(cluster_masses)} unique clusters across all snapshots.")

    #---Use this for analysis of the final snapshot only (many total clusters)---

    for snapshot_idx in range(n_snapshots):
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
            num_bins = max(2, min(num_bins, 100))  # Ensure num_bins is between 5 and 100
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
                results_snapshots.append(snapshot_idx)
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
                plt.savefig(f"{individual_plots_dir}cluster_mass_function_snapshot{snapshot_idx}.png", dpi=200)
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


def star_forming_vs_non_star_forming_age_distribution(analysis_dir, cluster_data):


    """
    Analyze the median age distribution of star forming clusters during their burst snapshot.
    Compare that to the median age distribution of non-star forming clusters during their detection snapshot.
    """

    age_distribution_analysis_dir = f"{analysis_dir}age_distribution_analysis/"
    os.makedirs(age_distribution_analysis_dir, exist_ok=True)
    
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

    # Setup the colormap and normalization
    # 'viridis', 'plasma', 'cividis' are good choices
    cmap = plt.get_cmap('viridis') 
    norm = colors.Normalize(vmin=radius_min, vmax=radius_max)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot both histograms on the same axes
    plot_colored_histogram(ax, median_ages_star_forming, radii_star_forming,
                        n_bins=10, colormap=cmap, norm=norm, 
                        label='Star Forming Clusters', alpha=0.5)

    plot_colored_histogram(ax, median_ages_non_star_forming, radii_non_star_forming,
                        n_bins=90, colormap=cmap, norm=norm, 
                        label='Non-Star Forming Clusters', alpha=0.9)

    # Add the colorbar as a legend for the radius
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # You need this to make the colorbar work
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Median Radial Distance (kpc)', rotation=270, labelpad=20)


    # Final plot formatting
    ax.set_title("Median Age Distribution Colored by Radial Distance")
    ax.set_xlabel("Median Age (Gyr)")
    ax.set_ylabel("Cluster Count")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{age_distribution_analysis_dir}colored_median_age_distribution_colored_by_radius.png", dpi=300)
    plt.show()
    plt.close()
    

    #plot the mean age distributions
    plt.figure(figsize=(12, 8))
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

def plot_colored_histogram(ax, ages, radii, n_bins, colormap, norm, label, alpha=1.0):
    """
    Plots a histogram where each bar is colored by the median radius of the data in that bin.
    """
    # 1. Calculate the histogram counts and bin edges without plotting
    counts, bin_edges = np.histogram(ages, bins=n_bins)
    
    # 2. Calculate the median radius for each bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    median_radii_per_bin = []
    for i in range(len(bin_edges) - 1):
        # Find which data points fall into the current bin
        in_bin_mask = (ages >= bin_edges[i]) & (ages < bin_edges[i+1])
        
        # Get the radii for those points
        radii_in_bin = radii[in_bin_mask]
        
        # Calculate the median radius, handle empty bins
        if len(radii_in_bin) > 0:
            median_radii_per_bin.append(np.median(radii_in_bin))
        else:
            median_radii_per_bin.append(np.nan) # Use NaN for empty bins

    # 3. Plot the bars one by one with the correct color
    for i in range(len(counts)):
        if not np.isnan(median_radii_per_bin[i]):
            # Use the colormap and normalization to get the color
            bar_color = colormap(norm(median_radii_per_bin[i]))
            # The first bar gets the label for the legend
            bar_label = label if i == 0 else None
            ax.bar(bin_centers[i], counts[i], 
                   width=(bin_edges[1]-bin_edges[0]), 
                   color=bar_color, 
                   alpha=alpha,
                   label=bar_label)

def loss_contamination_analysis(cluster_metrics, cluster_masses, contamination_analysis_dir, dir):
    contamination_data = np.load(f"{contamination_analysis_dir}burst_clusters_contamination_data.npy", allow_pickle=True).item()
    dir = f"{dir}contamination/"
    os.makedirs(dir, exist_ok=True)

    #structure of contamination_data: cluster_idx: {snapshot_idx: {'loss': loss_value, 'contamination': contamination_value}}

    # Plot Loss/Contamination vs N/M as scatter plot in last snapshot of cluster
    losses = []
    contaminations = []
    masses = []

    for cluster_idx, data in contamination_data.items():
        # Get the last snapshot where the cluster was detected
        last_snapshot_idx = max(data['snapshots'])-min(data['snapshots']) + 1  # Assuming snapshots are 0-indexed and continuous

        # Get the loss and contamination values for that snapshot
        loss = data['loss_fractions'][last_snapshot_idx]
        contamination = data['contamination_fractions'][last_snapshot_idx]

        # Get the mass of the cluster at that snapshot
        mass = np.sum(cluster_masses[cluster_idx][max(data['snapshots'])])  # Assuming cluster_masses is a dict with cluster_idx as keys and mass arrays as values

        losses.append(loss)
        contaminations.append(contamination)
        masses.append(mass)
        
    # Convert to numpy arrays for easier manipulation
    loss = np.array(losses)
    contamination = np.array(contaminations)
    mass = np.array(masses)

    mask = ~((loss == 1) & (contamination == 0))
    loss = loss[mask]
    contamination = contamination[mask]
    mass = mass[mask]

    log_mass = np.log10(mass)

    # --- Fit 1: Loss vs. Log of Mass ---
    # linregress returns: slope, intercept, r_value, p_value, stderr
    slope_loss, intercept_loss, r_value_loss, _, _ = stats.linregress(log_mass, loss)

    # --- Fit 2: Contamination vs. Log of Mass ---
    slope_cont, intercept_cont, r_value_cont, _, _ = stats.linregress(log_mass, contamination)


    # --- Create the plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    fig.suptitle('Loss and Contamination Trends for Star Clusters', fontsize=16)


    # --- Plot 1: Loss vs. Mass ---
    ax1.scatter(mass, loss, color='blue', label='Loss Data', alpha=0.5)

    # Create the points for the trendline
    # We calculate the y-values using the fit results on the log_mass
    trend_y_loss = slope_loss * log_mass + intercept_loss
    # Plot the trendline. We can sort the values to ensure the line is drawn correctly.
    sorted_indices = np.argsort(mass)
    ax1.plot(mass[sorted_indices], trend_y_loss[sorted_indices], color='darkblue', linestyle='--', 
            label=f'Fit (R²={r_value_loss**2:.2f})') # Add R-squared to the label

    ax1.set_title('Loss vs. Cluster Mass')
    ax1.set_xlabel('Cluster Mass (M☉)')
    ax1.set_ylabel('Fraction')
    ax1.legend()


    # --- Plot 2: Contamination vs. Mass ---
    ax2.scatter(mass, contamination, color='red', label='Contamination Data', alpha=0.5)

    # Create the points for the trendline
    trend_y_cont = slope_cont * log_mass + intercept_cont
    # Plot the trendline
    ax2.plot(mass[sorted_indices], trend_y_cont[sorted_indices], color='darkred', linestyle='--', 
            label=f'Fit (R²={r_value_cont**2:.2f})')

    ax2.set_title('Contamination vs. Cluster Mass')
    ax2.set_xlabel('Cluster Mass (M☉)')
    ax2.legend()


    # Apply common settings to both axes
    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.05)  # Set y-limits to [0, 1] for both plots

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{dir}loss_contamination_with_trendlines.png", dpi=300)
    plt.close()

    print("Plot saved successfully!")


    # Plot the loss and contamination values against the mass
    plt.figure(figsize=(8, 6))
    plt.scatter(mass, loss, color='blue', label='Loss', alpha=0.6)
    plt.scatter(mass, contamination, color='red', label='Contamination', alpha=0.6)
    plt.xscale('log')
    plt.xlabel('Cluster Mass (M)')
    plt.ylabel('Loss/Contamination')
    plt.title(f'Loss and Contamination for star clusters at last snapshots')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dir}loss_contamination_scatter.png", dpi=300)
    plt.close()





# Load CSV files
cluster_metrics_chemdyn = np.load('/home/samuel_data/nihao_uhd_2.79e12_zoom_6_rerun_stars_all_snapshots_chemodynamical_tagging_S=auto/star_cluster_analysis_3/all_cluster_metrics.npy', allow_pickle=True).item()
cluster_masses_chemdyn = np.load('/home/samuel_data/nihao_uhd_2.79e12_zoom_6_rerun_stars_all_snapshots_chemodynamical_tagging_S=auto/star_cluster_analysis_3/cluster_masses.npy', allow_pickle=True).item()
cluster_metrics_dyn = np.load('/home/samuel_data/nihao_uhd_2.79e12_zoom_6_rerun_stars_all_snapshots_dynamical_tagging_S=auto/star_cluster_analysis_3/all_cluster_metrics.npy', allow_pickle=True).item()
cluster_masses_dyn = np.load('/home/samuel_data/nihao_uhd_2.79e12_zoom_6_rerun_stars_all_snapshots_dynamical_tagging_S=auto/star_cluster_analysis_3/cluster_masses.npy', allow_pickle=True).item()

star_forming_clusters_chemdyn = {k: v for k, v in cluster_metrics_chemdyn.items() if v['burst_cluster']}
star_forming_clusters_dyn = {k: v for k, v in cluster_metrics_dyn.items() if v['burst_cluster']}

star_forming_masses_chemdyn = {k: v for k, v in cluster_masses_chemdyn.items() if k in star_forming_clusters_chemdyn}
star_forming_masses_dyn = {k: v for k, v in cluster_masses_dyn.items() if k in star_forming_clusters_dyn}

#dir = "/home/samuel_data/nihao_uhd_2.79e12_zoom_6_rerun_stars_all_snapshots_chemodynamical_tagging_S=auto/star_cluster_analysis_3/burst_cluster_analysis/spacial_distribution_analysis/"

#spacial_distribution_analysis(star_forming_clusters_chemdyn, dir)

#dir = "/home/samuel_data/nihao_uhd_2.79e12_zoom_6_rerun_stars_all_snapshots_dynamical_tagging_S=auto/star_cluster_analysis_3/burst_cluster_analysis/spacial_distribution_analysis/"
#dir = "/home/samuel_data/nihao_uhd_2.79e12_zoom_6_rerun_stars_all_snapshots_dynamical_tagging_S=auto/star_cluster_analysis_3/burst_cluster_analysis/spacial_distribution_analysis/"
#spacial_distribution_analysis(star_forming_clusters_dyn, dir)

dir = "/home/samuel_data/further_analysis/"
os.makedirs(dir, exist_ok=True)

chemdyn_dir = f"{dir}chemodynamical_tagging/"
os.makedirs(chemdyn_dir, exist_ok=True)

dyn_dir = f"{dir}dynamical_tagging/"
os.makedirs(dyn_dir, exist_ok=True)

# # # Perform power law analysis for chemodynamical clusters
# power_law_analysis_dir_chemdyn = f"{dir}chemodynamical_tagging/"
# os.makedirs(power_law_analysis_dir_chemdyn, exist_ok=True)
# n_snapshots = 201
# power_law_analysis_many_clusters(power_law_analysis_dir_chemdyn, n_snapshots, cluster_masses_chemdyn)
# power_law_analysis_few_clusters(power_law_analysis_dir_chemdyn, n_snapshots, star_forming_masses_chemdyn)

# # Perform power law analysis for dynamical clusters
# power_law_analysis_dir_dyn = f"{dir}dynamical_tagging/"
# os.makedirs(power_law_analysis_dir_dyn, exist_ok=True)
# n_snapshots = 201
# #power_law_analysis_many_clusters(power_law_analysis_dir_dyn, n_snapshots, cluster_masses_dyn)
# power_law_analysis_few_clusters(power_law_analysis_dir_dyn, n_snapshots, star_forming_masses_dyn)


#plot_cluster_mass_distribution(dir)

star_forming_vs_non_star_forming_age_distribution(chemdyn_dir, cluster_metrics_chemdyn)
star_forming_vs_non_star_forming_age_distribution(dyn_dir, cluster_metrics_dyn)
contamination_data_dir_chemdyn = "/home/samuel_data/nihao_uhd_2.79e12_zoom_6_rerun_stars_all_snapshots_chemodynamical_tagging_S=auto/star_cluster_analysis_3/burst_cluster_analysis/contamination_analysis/"
contamination_data_dir_dyn = "/home/samuel_data/nihao_uhd_2.79e12_zoom_6_rerun_stars_all_snapshots_dynamical_tagging_S=auto/star_cluster_analysis_3/burst_cluster_analysis/contamination_analysis/"

#loss_contamination_analysis(star_forming_clusters_chemdyn, star_forming_masses_chemdyn, contamination_data_dir_chemdyn, chemdyn_dir)
#loss_contamination_analysis(star_forming_clusters_dyn, star_forming_masses_dyn, contamination_data_dir_dyn, dyn_dir)








