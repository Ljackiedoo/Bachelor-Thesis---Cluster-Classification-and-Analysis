import os
import gc
import glob

import numpy as np
import pynbody as pb
import matplotlib.pyplot as plt
import matplotlib.colors as col
import ffmpeg

from astrolink import AstroLink
from fuzzycat import FuzzyCat, FuzzyPlots

from astrolink.io import loadAstroLinkObject
from astrolink.io import saveAstroLinkObject

import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d

def loadGalaxyAsArrays(snapshotFilePath, particleName, featureSpaceNames = ['pos', 'vel', 'mass']):
    """Returns the main halo data, from the simulation file `snapshotFilePath`,
    for particle `particleName`, in the feature spaces specified by
    `featureSpaceNames`.
    """

    # Load the simulation snapshot
    simulation = pb.load(snapshotFilePath)

    # Take only the largest halo and make it face-on (stellar disk is in the x-y plane)
    mainHalo = simulation.halos()[0]
    pb.analysis.angmom.faceon(mainHalo)
    mainHalo.physical_units()

    # Centre data on the median of the dark matter halo
    darkMatter = np.column_stack([mainHalo.dm[feature] for feature in featureSpaceNames])
    centre = np.median(darkMatter, axis = 0)

    # Get particle data and IDs
    if particleName == 'dm':
        darkMatter -= centre
        darkMatterIDs = mainHalo.dm['iord']
        darkMatterWeights = np.array(mainHalo.dm['mass'])
        return darkMatter, darkMatterIDs, darkMatterWeights, simulation
    if particleName == 'stars':
        stars = np.column_stack([mainHalo.stars[feature] for feature in featureSpaceNames])
        stars -= centre
        starsIDs = mainHalo.stars['iord']
        starMasses = np.array(mainHalo.stars['mass'])
        return stars, starsIDs, starMasses, simulation
    if particleName == 'gas':
        gas = np.column_stack([mainHalo.gas[feature] for feature in featureSpaceNames])
        gas -= centre
        gasIDs = mainHalo.gas['iord']
        gasMasses = np.array(mainHalo.gas['mass'])
        gasTemperatures = np.array(mainHalo.gas['temp'])
        return gas, gasIDs, gasMasses, gasTemperatures, simulation
    if particleName == 'stars_gas':
        stars = np.column_stack([mainHalo.stars[feature] for feature in featureSpaceNames])
        stars -= centre
        starsIDs = mainHalo.stars['iord']
        starMasses = np.array(mainHalo.stars['mass'])
        gas = np.column_stack([mainHalo.gas[feature] for feature in featureSpaceNames])
        gas -= centre
        gasIDs = mainHalo.gas['iord']
        gasMasses = np.array(mainHalo.gas['mass'])
        particles = np.vstack([stars, gas])
        particleIDs = np.hstack([starsIDs, gasIDs])
        allMasses = np.hstack([starMasses, gasMasses])

        return particles, particleIDs, allMasses, simulation
    
    if particleName == 'stars_gas_dm':
        stars = np.column_stack([mainHalo.stars[feature] for feature in featureSpaceNames])
        stars -= centre
        starsIDs = mainHalo.stars['iord']
        starMasses = np.array(mainHalo.stars['mass'])
        gas = np.column_stack([mainHalo.gas[feature] for feature in featureSpaceNames])
        gas -= centre
        gasIDs = mainHalo.gas['iord']
        gasMasses = np.array(mainHalo.gas['mass'])
        darkMatter = np.column_stack([mainHalo.dm[feature] for feature in featureSpaceNames])
        darkMatter -= centre
        darkMatterIDs = mainHalo.dm['iord']
        darkMatterWeights = np.array(mainHalo.dm['mass'])
        particles = np.vstack([stars, gas, darkMatter])
        particleIDs = np.hstack([starsIDs, gasIDs, darkMatterIDs])
        allMasses = np.hstack([starMasses, gasMasses, darkMatterWeights])

        return particles, particleIDs, allMasses, simulation
        

def findAndSaveClustersFromSnapshots(snapshotFilePaths, workingDirectoryPath, particleName, nSamples, significance, rerun, dir_with_astrolink):
    """Uses AstroLink to find the clusters within each main halo specified by
    `snapshotFilePaths` and `particleName`. Then saves them in different formats
    into the directory specified by `workingDirectoryPath`. `nSamples` is used
    to format the cluster file names.
    """

    # The number of leading digits in the saved cluster file names
    sampleNumberFormat = np.log10(nSamples).astype(int) + 1

    # For tracking which star particles have been clustered over all snapshots (for FuzzyCat memory efficiency)
    veryLargeN = 10**8 # Must be larger than the maximum iord value
    particleIDsBool = np.zeros(veryLargeN, dtype = np.bool_)

    # Track cluster file names
    clusterFileNames = []

    # Cycle through each snapshot, run AstroLink, and save the clusters
    for index, snapshotFilePath in enumerate(snapshotFilePaths):
        print(f"Loading {snapshotFilePath.split('/')[-1]}                                                         \t\t", end = '\r')
        # Load the galaxy
        if(particleName == 'gas'):
            particleArr, particleIDs, weights, temperatures, _ = loadGalaxyAsArrays(snapshotFilePath, particleName)
        else:
            particleArr, particleIDs, weights,  _ = loadGalaxyAsArrays(snapshotFilePath, particleName)


        print(f"Running AstroLink on the {particleName} particles of snapshot {snapshotFilePath.split('/')[-1]}   \t\t", end = '\r')
        # Run AstroLink for the first time
        if rerun == False:
            c = AstroLink(particleArr, workers=16, weights=weights, S = significance)          
            c.run()
            del c.P
            saveAstroLinkObject(c, f"{dir_with_astrolink}{index:0{sampleNumberFormat}}")
        else:
            c = loadAstroLinkObject(f"{dir_with_astrolink}{index:0{sampleNumberFormat}}")
            c.S = significance
            c.extract_clusters()
        
        for clst, clst_id in zip(c.clusters[1:], c.ids[1:]):
            # Cluster file name
            clusterFileName = f"{index:0{sampleNumberFormat}}_{clst_id}.npy"
            clusterFileNames.append(clusterFileName)

            # Save the cluster with respect to the order of the data in the snapshot file
            cluster_raw = c.ordering[clst[0]:clst[1]]
            np.save(f"{workingDirectoryPath}Clusters_raw/{clusterFileName}", cluster_raw)

            # Save the cluster with respect to the particle IDs in the snapshot file
            cluster_iord = particleIDs[cluster_raw]
            np.save(f"{workingDirectoryPath}Clusters_iord/{clusterFileName}", cluster_iord)

            # Mark the particles that have been clustered
            particleIDsBool[cluster_iord] = 1

    # Save the IDs of the star particles that have been clustered
    clusteredIDs = np.where(particleIDsBool)[0]
    np.save(f"{workingDirectoryPath}clusteredIDs.npy", clusteredIDs)

    # Translate the clusters (with respect to the particle IDs) into reduced arrays (with respect to the order of the IDs of clustered particles) for improved memory efficiency with FuzzyCat
    for clusterFileName in clusterFileNames:
        cluster_iord = np.load(workingDirectoryPath + 'Clusters_iord/' + clusterFileName)
        cluster_reduced = np.where(np.isin(clusteredIDs, cluster_iord, assume_unique = True))[0].astype(cluster_iord.dtype)
        np.save(workingDirectoryPath + 'Clusters/' + clusterFileName, cluster_reduced)


def runFuzzyCatOnClustersFromSnapshots(workingDirectoryPath, nSamples, minStability, fuzzycat_window):
    """Runs FuzzyCat on the clusters contained in `workingDirectoryPath` with
    parameters `nSamples` and `minStability`. The `nPoints` parameter is
    determined automatically from a file containing the IDs of clustered
    particles.
    """

    # Number of points clustered
    clusteredIDs = np.load(f"{workingDirectoryPath}clusteredIDs.npy")
    nPoints = clusteredIDs.size
    del clusteredIDs

    # Run FuzzyCat
    fc = FuzzyCat(nSamples, nPoints, workingDirectoryPath, minStability = minStability, checkpoint = True, verbose = 2, windowSize = fuzzycat_window)
    fc.run()

    # Plot the basic results
    FuzzyPlots.plotOrderedJaccardIndex(fc)
    FuzzyPlots.plotStabilities(fc)
    FuzzyPlots.plotMemberships(fc)

    # Save outputs
    np.save(f"{workingDirectoryPath}jaccardIndices.npy", fc.jaccardIndices)
    np.save(f"{workingDirectoryPath}ordering.npy", fc.ordering)
    np.save(f"{workingDirectoryPath}fuzzyClusters.npy", fc.fuzzyClusters)
    np.save(f"{workingDirectoryPath}stabilities.npy", fc.stabilities)
    np.save(f"{workingDirectoryPath}memberships.npy", fc.memberships)
    np.save(f"{workingDirectoryPath}memberships_flat.npy", fc.memberships_flat)
    np.save(f"{workingDirectoryPath}fuzzyHierarchy.npy", fc.fuzzyHierarchy)
    np.save(f"{workingDirectoryPath}groups.npy", fc.groups)
    np.save(f"{workingDirectoryPath}intraJaccardIndicesGroups.npy", fc.intraJaccardIndicesGroups)
    np.save(f"{workingDirectoryPath}interJaccardIndicesGroups.npy", fc.interJaccardIndicesGroups)
    np.save(f"{workingDirectoryPath}stabilitiesGroups.npy", fc.stabilitiesGroups)


def paintLabelsOntoSnapshot(particleArr, clusters_raw, labels, saveFileNameStem, snapshotFileName, axisLimits, plot_labels, withDiskZoomIn = True, color_mapping = None):
    """Creates a two-panel plot of the clusters within a snapshot. The left
    panel is a 3D scatter plot and the right panel is a top-down view of the
    region around the disk of the galaxy.
    """

    # Colour the data according to the cluster
    colourList = [f"C{i}" for i in range(10) if i != 7]
    colours = np.zeros((particleArr.shape[0], 4))
    sizes = np.zeros(particleArr.shape[0])

    original_to_filtered = {}
    
    for cluster_idx, (cluster_raw, label) in enumerate(zip(clusters_raw, labels)):
        if color_mapping is not None and label in color_mapping:
            # Use the consistent color mapping if provided
            color = col.to_rgba(color_mapping[label], alpha=1)
        else:
            # Fall back to the default behavior
            color = col.to_rgba(colourList[label % len(colourList)], alpha=1)
        
        colours[cluster_raw] = color
        sizes[cluster_raw] = 0.5
        
        # Store the mapping from original cluster to its particles
        original_to_filtered[cluster_idx] = cluster_raw

    # Filter out entries with zero size
    non_zero_mask = sizes > 0

    original_indices = np.arange(particleArr.shape[0])
    filtered_indices = original_indices[non_zero_mask]
    index_mapping = {orig: i for i, orig in enumerate(filtered_indices)}

    # Update the cluster indices to point to the filtered array
    filtered_clusters = {}
    for cluster_idx, orig_indices in original_to_filtered.items():
        # Keep only indices that remain after filtering
        valid_indices = [index_mapping[i] for i in orig_indices if i in index_mapping]
        if valid_indices:  # If any particles remain
            filtered_clusters[cluster_idx] = np.array(valid_indices)


    particleArr = particleArr[non_zero_mask]
    colours = colours[non_zero_mask]
    sizes = sizes[non_zero_mask]

    sample_rate = 2

    sampled_clusters = {}
    
    if sample_rate > 1:
        # Create new mapping for sampled indices
        particle_indices = np.arange(0, len(particleArr), sample_rate)
        sample_mask = np.zeros(len(particleArr), dtype=bool)
        sample_mask[particle_indices] = True
        
        # Update the cluster indices to point to the sampled array
        for cluster_idx, filtered_indices in filtered_clusters.items():
            # Find which particles in this cluster are kept after sampling
            cluster_mask = np.isin(np.arange(len(particleArr)), filtered_indices)
            keep_mask = sample_mask & cluster_mask
            
            if np.any(keep_mask):
                # Map to new indices in the sampled array
                new_indices = np.cumsum(sample_mask) - 1
                sampled_clusters[cluster_idx] = new_indices[keep_mask]
        
        # Apply sampling
        particleArr = particleArr[particle_indices]
        colours = colours[particle_indices]
        sizes = sizes[particle_indices]
    else:
        sampled_clusters = filtered_clusters
    
    
    # Create figure
    width = 16 if withDiskZoomIn else 8
    height = 8
    figAspectRatio = height/width
    fig = plt.figure(figsize = (width, height))
    fig.patch.set_facecolor('k')

    # Plot the 3D data
    ax = fig.add_axes((0, 0, figAspectRatio, 1), projection = '3d')
    
    ax.scatter(*particleArr[:, :3].T, facecolors = colours, edgecolors = 'w', s = sizes, lw = 0.05)
    
    current_zlim = ax.get_zlim()
    current_view = ax.get_proj()
    
    # Adjust data limits
    ax.set_xlim(-axisLimits, axisLimits)
    ax.set_ylim(-axisLimits, axisLimits)
    ax.set_zlim(-axisLimits, axisLimits)
    # Remove axes
    ax.axis('off')
    ax.patch.set_facecolor('k')
    # Add cartesian coordinate axes of length 100 kpc for reference
    ax.quiver([0]*6, [0]*6, [0]*6, [1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1],
              color = 'w', alpha = 1, length = 100, arrow_length_ratio = 0.1)
    
    if(plot_labels):
        for cluster_idx, sampled_indices in sampled_clusters.items():
            # Get the original label for this cluster
            label = labels[cluster_idx]
            
            # Calculate centroid of this cluster's particles
            centroid = np.mean(particleArr[sampled_indices, :3], axis=0)
            
            # Add the label text
            text = ax.text(centroid[0], centroid[1], centroid[2], f"{label}", 
                color='white', fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
            
            text.set_zorder(1000)


    ax.text(100, 0, 0, 'X', color = 'w')
    ax.text(0, 100, 0, 'Y', color = 'w')
    ax.text(0, 0, 100, 'Z', color = 'w')
    if withDiskZoomIn:
        # Add zoom-in box around disk
        prismColour, prismAlpha = col.to_rgba('w', alpha = 0.2), 0.05
        xyRange, zRange, onesArray = np.array([-25, 25]), np.array([-5, 5]), np.ones(4).reshape(2, 2)
        for i in range(2):
            # z-direction faces
            xx, yy = np.meshgrid(xyRange, xyRange)
            ax.plot_wireframe(xx, yy, zRange[i]*onesArray, color = prismColour)
            ax.plot_surface(xx, yy, zRange[i]*onesArray, color = prismColour, alpha = prismAlpha)
            # x-direction faces
            xy, zz = np.meshgrid(xyRange, zRange)
            ax.plot_wireframe(xyRange[i]*onesArray, xy, zz, color = prismColour)
            ax.plot_surface(xyRange[i]*onesArray, xy, zz, color = prismColour, alpha = prismAlpha)
            # y-direction faces
            ax.plot_wireframe(xy, xyRange[i]*onesArray, zz, color = prismColour)
            ax.plot_surface(xy, xyRange[i]*onesArray, zz, color = prismColour, alpha = prismAlpha)

        # Plot the 2D disk data
        axisCentre, axisHalfWidth = 0.5 + figAspectRatio/2, 0.9*(1 - figAspectRatio)
        axDisk = fig.add_axes((axisCentre - axisHalfWidth/2,
                            0.5*(1 - axisHalfWidth/figAspectRatio),
                            axisHalfWidth,
                            axisHalfWidth/figAspectRatio))
        inBoxBool = (particleArr[:, 0] > xyRange[0])*(particleArr[:, 0] < xyRange[1]) # particles in x limits
        inBoxBool *= (particleArr[:, 1] > xyRange[0])*(particleArr[:, 1] < xyRange[1]) # particles in y limits
        inBoxBool *= (particleArr[:, 2] > zRange[0])*(particleArr[:, 2] < zRange[1]) # particles in z limits
        axDisk.scatter(*particleArr[inBoxBool, :2].T, facecolors = colours[inBoxBool], edgecolors = 'w', s = 2*sizes[inBoxBool], lw = 0.05)
        
        if(plot_labels):
            # Add cluster labels to 2D plot
            for cluster_idx, sampled_indices in sampled_clusters.items():
                # Create a mask for particles in this cluster
                cluster_mask = np.zeros(len(particleArr), dtype=bool)
                cluster_mask[sampled_indices] = True
                
                # Find particles from this cluster that are also in the box
                cluster_in_box_mask = cluster_mask & inBoxBool
                
                # Only add label if there are particles from this cluster in the box
                if np.any(cluster_in_box_mask):
                    # Get the original label for this cluster
                    label = labels[cluster_idx]
                    
                    # Calculate centroid of this cluster's particles within the box (only X and Y)
                    centroid_2d = np.mean(particleArr[cluster_in_box_mask, :2], axis=0)
                    
                    # Add the label text to 2D plot
                    text_2d = axDisk.text(centroid_2d[0], centroid_2d[1], f"{label}", 
                        color='white', fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
                    
                    text_2d.set_zorder(1000)

        # Adjust data limits
        axDisk.set_xlim(xyRange[0], xyRange[1])
        axDisk.set_ylim(xyRange[0], xyRange[1])
        # Remove axes
        axDisk.patch.set_facecolor('k')
        for side in ['top', 'left', 'bottom', 'right']:
            axDisk.spines[side].set_color('w')

    # Adjust figure margins
    top, bottom, left, right = 1, 0, 0, 1
    fig.subplots_adjust(top = top, bottom = bottom, left = left, right = right)
    # Add snapshot number
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = 'none', top = False, bottom = False, left = False, right = False)
    plt.grid(False)
    plt.text(0, 1, snapshotFileName, ha = 'left', va = 'top', fontsize = 10, color = 'w', transform = plt.gca().transAxes)
    # Save figure
    plt.savefig(f"{saveFileNameStem}{snapshotFileName}.png", dpi = 200, bbox_inches = 'tight')
    fig.clf()
    plt.close()
    gc.collect()


def makeMovieOfFuzzyClustersOverTimeByParticleType(snapshotFilePaths, workingDirectoryPath, particleType, axisLimits, frameRate, color_mapping = None):
    """Makes a movie of the fuzzy clusters for a specific particle type (stars or gas) found by AstroLink and FuzzyCat
    as they evolve over time. Optimized for performance.
    """
    # Prepare output path for this particle type
    saveFileNameStem = f"{workingDirectoryPath}Cluster_plots_2/plotted_clusters_{particleType}_"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(saveFileNameStem), exist_ok=True)

    print(f"Loading cluster metadata...")
    # Load cluster data
    clusterFileNames = np.load(workingDirectoryPath + 'clusterFileNames.npy')
    ordering = np.load(workingDirectoryPath + 'ordering.npy')
    fuzzyClusters = np.load(workingDirectoryPath + 'fuzzyClusters.npy')
    
    # Create mapping for fuzzy clusters
    whichCluster = -np.ones(clusterFileNames.size, dtype=np.int32)
    for i, clst in enumerate(fuzzyClusters):
        whichCluster[ordering[clst[0]:clst[1]]] = i


    for index, snapshotFilePath in enumerate(snapshotFilePaths):
        snapshotFileName = snapshotFilePath.split('/')[-1]
        print(f"Loading {snapshotFileName}                                                         \t\t", end = '\r')
        # Load the galaxy
        
        particleArr = loadGalaxyAsArrays(snapshotFilePath, particleName)[0]
            

        print(f"Loading clusters of {particleName} particles from snapshot {snapshotFileName}   \t\t", end = '\r')
        # Load AstroLink clusters (found in this snapshot) that belong to the fuzzy clusters from FuzzyCat
        clusters_raw, fuzzyLabels = [], []
        for clusterFileName, whichFuzzyClst in zip(clusterFileNames, whichCluster):
            clstSnapshot = int(clusterFileName.split('_')[0])
            if whichFuzzyClst != -1 and clstSnapshot == index:
                cluster_raw = np.load(workingDirectoryPath + 'Clusters_raw/' + clusterFileName)
                clusters_raw.append(cluster_raw)
                fuzzyLabels.append(whichFuzzyClst)


        # Make plot of clusters if there are any
        if clusters_raw:
            print(f"Plotting {len(clusters_raw)} astrolink {particleType} clusters for {snapshotFileName}")
            paintLabelsOntoSnapshot(
                particleArr, 
                clusters_raw, 
                fuzzyLabels, 
                saveFileNameStem, 
                snapshotFileName, 
                axisLimits,
                color_mapping = color_mapping
            )

    # Make movie
    print(f"Creating movie for {particleType}...")
    (
        ffmpeg
        .input(f"{saveFileNameStem}*.png", pattern_type='glob', framerate=frameRate)
        .output(f"{workingDirectoryPath}{workingDirectoryPath.split('/')[-2]}_{particleType}_movie.mp4")
        .run()
    )


def makeMovieOfFuzzyClustersOverTime(snapshotFilePaths, workingDirectoryPath, particleName, axisLimits, frameRate):
    """Makes movies of the fuzzy clusters found by AstroLink and FuzzyCat as
    they evolve over time. For combined data, creates separate movies for each particle type.
    """
     # If we're working with combined stars and gas data, make separate movies for each
    print("Loading cluster metadata for color consistency...")
    # Load cluster data once to identify all fuzzy cluster IDs
    clusterFileNames = np.load(workingDirectoryPath + 'clusterFileNames.npy')
    ordering = np.load(workingDirectoryPath + 'ordering.npy')
    fuzzyClusters = np.load(workingDirectoryPath + 'fuzzyClusters.npy')
    
    # Create mapping for fuzzy clusters
    whichCluster = -np.ones(clusterFileNames.size, dtype=np.int32)
    for i, clst in enumerate(fuzzyClusters):
        whichCluster[ordering[clst[0]:clst[1]]] = i
        
    # Find all unique fuzzy cluster IDs
    unique_fuzzy_ids = np.unique(whichCluster[whichCluster != -1])
    
    # Create a fixed color mapping for all fuzzy cluster IDs
    # This ensures the same cluster gets the same color in both stars and gas visualizations
    colourList = [f"C{i}" for i in range(10) if i != 7]  # Exclude C7 (gray)
    color_mapping = {cluster_id: colourList[i % len(colourList)] for i, cluster_id in enumerate(unique_fuzzy_ids)}

    for particleType in particleName.split('_'):

        print(f"Creating movie for {particleType}...")
        makeMovieOfFuzzyClustersOverTimeByParticleType(
            snapshotFilePaths, workingDirectoryPath, particleType, axisLimits, frameRate, color_mapping
        )     


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
  
    

    
def star_cluster_analysis(snapshot_file_paths, working_directory_path, axisLimits, frameRate):
    # Create output directories
    analysis_dir = f"{working_directory_path}further_Cluster_origin_analysis_2/"
    plots_dir = f"{analysis_dir}plots_3/"
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

        
    burst_clusters = fast_identify_burst_clusters(snapshot_file_paths, working_directory_path, analysis_dir, star_ages_dir, fuzzy_cluster_snapshot_ranges)

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
            star_ids = star_data[:,0]
            star_ages = star_data[:,1]


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


            #convert birth and death to relative years

            cluster_detected_time = (snap_idx - fuzzy_cluster_snapshot_ranges[cluster_idx][0])*13.8/2000
            cluster_lost_time = (snap_idx - fuzzy_cluster_snapshot_ranges[cluster_idx][1] )*13.8/2000

            ticks_linear = [0.1, 0.5, 1.0]  # in linear region (you can adjust)
            ticks_log = list(range(1, 14))  # for the log region
            all_ticks = ticks_linear + ticks_log

            #plot pdf of birth dates
            plt.figure(figsize=(8, 6))
            plt.hist(star_ages_for_cluster_until_now, bins=1000, histtype='stepfilled', color='b', linewidth=2, label='All Snapshots until now', )
            plt.hist(this_snapshot_star_ages, bins=1000, histtype='step', color='r', linewidth=2, label='Current Snapshot')
            plt.axvline(x=cluster_detected_time, color='purple', linestyle='--', label='Cluster detected')
            plt.axvline(x=cluster_lost_time, color='orange', linestyle='--', label='Cluster lost')
            plt.xscale('symlog', linthresh=1)
            plt.xticks(all_ticks, [str(tick) for tick in ticks_linear] + [str(tick) for tick in ticks_log])
            plt.xlim(0, 13.8)
            plt.xlabel('Age (Gyr)')
            plt.ylabel('No. of Particles')
            plt.title(f'Star Particle Ages in Cluster {cluster_idx} (Snapshot {snap_idx})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            # Save the plot in the cluster's directory with a consistent naming pattern for the movie
            plt.savefig(f"{cluster_dir}frame_{snap_idx:04d}.png", dpi=300)
            plt.close()


            # #plot Cdf of birth dates
            # plt.figure(figsize=(8, 6))
            # plt.hist(star_ages, bins=1000, density=True, cumulative=True, histtype='step', color='b', linewidth=2)
            # plt.xscale('symlog', linthresh=1)
            # plt.xticks(all_ticks, [str(tick) for tick in ticks_linear] + [str(tick) for tick in ticks_log])
            # plt.xlim(0, 13.8)
            # plt.xlabel('Age (Gyr)')
            # plt.ylabel('Cumulative Fraction')
            # plt.title(f'Star Particle Ages in Cluster {cluster_idx} (Snapshot {snap_idx})')
            # plt.grid(True, alpha=0.3)
            # # Save the plot in the cluster's directory with a consistent naming pattern for the movie
            # plt.savefig(f"{cluster_dir}cdf_frame_{snap_idx:04d}.png", dpi=300)
            # plt.close()
        # After processing all snapshots, create movies for each cluster
    print("Creating evolution movies for each cluster...")
    movies_dir = f"{plots_dir}movies/"
    os.makedirs(movies_dir, exist_ok=True)
    for folder in os.listdir(plots_dir):
        if not folder.startswith('cluster_'):
            continue
        cluster_idx = folder.split('_')[1]
        cluster_dir = f"{plots_dir}cluster_{cluster_idx}/"
        
        movie_path = f"{movies_dir}cluster_{cluster_idx}_movie.mp4"
        
        # Check if there are any frames to make a movie
        frames = glob.glob(f"{cluster_dir}frame_*.png")
        if not frames:
            print(f"No frames found for cluster {cluster_idx}, skipping movie creation")
            continue
        
        print(f"Creating movie for cluster {cluster_idx}...")
        try:
            (
                ffmpeg
                .input(f"{cluster_dir}frame_*.png", pattern_type='glob', framerate=frameRate)
                .output(movie_path)
                .run()
            )
            print(f"Movie saved to: {movie_path}")
        except Exception as e:
            print(f"Error creating movie for cluster {cluster_idx}: {e}")
    

   

    make_movie_of_fuzzy_star_forming_clusters(burst_clusters, analysis_dir, working_directory_path, snapshot_file_paths, axisLimits, frameRate)

def make_movie_of_fuzzy_star_forming_clusters(burst_clusters, analysis_dir, workingDirectoryPath, snapshotFilePaths, axisLimits, frameRate):
    #make a movie like in the makemovieoffuzzycatclusters function, but only for the best burst clusters

    # Sort the dataframe by t25_t75 in ascending order
    sorted_df = burst_clusters.sort_values('t25_t75')

    # Get the top 50 clusters with lowest t25_t75 values
    top_50_clusters = sorted_df.head(50)

    # If you only need the cluster_idx values
    top_50_cluster_ids = top_50_clusters['cluster_idx'].tolist()
    
    particleType = 'stars'

    print("Creating movies for the best burst clusters...")
    # Prepare output path for this particle type
    saveFileNameStem = f"{analysis_dir}Cluster_plots/plotted_clusters_{particleType}_"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(saveFileNameStem), exist_ok=True)

    print(f"Loading cluster metadata...")
    # Load cluster data
    clusterFileNames = np.load(workingDirectoryPath + 'clusterFileNames.npy')
    ordering = np.load(workingDirectoryPath + 'ordering.npy')
    fuzzyClusters = np.load(workingDirectoryPath + 'fuzzyClusters.npy')
    
    # Create mapping for fuzzy clusters
    whichCluster = -np.ones(clusterFileNames.size, dtype=np.int32)
    for i, clst in enumerate(fuzzyClusters):
        whichCluster[ordering[clst[0]:clst[1]]] = i



    for index, snapshotFilePath in enumerate(snapshotFilePaths):
        snapshotFileName = snapshotFilePath.split('/')[-1]
        print(f"Loading {snapshotFileName}                                                         \t\t", end = '\r')
        # Load the galaxy
        
        particleArr = loadGalaxyAsArrays(snapshotFilePath, particleName)[0]
            

        print(f"Loading clusters of {particleName} particles from snapshot {snapshotFileName}   \t\t", end = '\r')
        # Load AstroLink clusters (found in this snapshot) that belong to the fuzzy clusters from FuzzyCat
        clusters_raw, fuzzyLabels = [], []
        for clusterFileName, whichFuzzyClst in zip(clusterFileNames, whichCluster):
            clstSnapshot = int(clusterFileName.split('_')[0])
            if whichFuzzyClst != -1 and clstSnapshot == index and whichFuzzyClst in top_50_cluster_ids:
                cluster_raw = np.load(workingDirectoryPath + 'Clusters_raw/' + clusterFileName)
                clusters_raw.append(cluster_raw)
                fuzzyLabels.append(whichFuzzyClst)


        # Make plot of clusters if there are any
        print(f"Plotting {len(clusters_raw)} astrolink {particleType} clusters for {snapshotFileName}")
        paintLabelsOntoSnapshot(
            particleArr, 
            clusters_raw, 
            fuzzyLabels, 
            saveFileNameStem, 
            snapshotFileName, 
            axisLimits,
            plot_labels = True
        )

    # Make movie
    print(f"Creating movie for {particleType}...")
    (
        ffmpeg
        .input(f"{saveFileNameStem}*.png", pattern_type='glob', framerate=frameRate)
        .output(f"{analysis_dir}star_forming_clusters_movie.mp4")
        .run()
    )


def fast_identify_burst_clusters(snapshot_file_paths, working_directory_path, analysis_path, star_ages_dir, fuzzy_cluster_threshold_ranges, threshold_fraction=0.02):
    # Load fuzzy cluster data
    clustered_ids = np.load(f"{working_directory_path}clusteredIDs.npy")
    ordering = np.load(f"{working_directory_path}ordering.npy")
    fuzzy_clusters = np.load(f"{working_directory_path}fuzzyClusters.npy")
    clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")

    print("Finding potential star forming clusters...")

    burst_clusters = {}

    for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):

        print(f"Analyzing cluster {cluster_idx}/{len(fuzzy_clusters)}")
        cluster_start_snapshot = fuzzy_cluster_threshold_ranges[cluster_idx][0]
        cluster_end_snapshot = fuzzy_cluster_threshold_ranges[cluster_idx][1]

        for snap_idx in range(cluster_start_snapshot - 4, cluster_start_snapshot + 1):
            if snap_idx < 0 or snap_idx >= len(snapshot_file_paths):
                continue
            snapshot_path = snapshot_file_paths[snap_idx]
            print(f"Processing snapshot {snap_idx}/{len(snapshot_file_paths)}")
            star_data = np.load(f"{star_ages_dir}star_ages_{snap_idx:03d}.npy")
            star_ids = star_data[:,0]
            star_ages = star_data[:,1]

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

            # Calculate t25-t75 (time to form middle 50% of stars)
            t25 = np.percentile(sorted_ages, 25)
            t75 = np.percentile(sorted_ages, 75)
            t25_t75 = t75 - t25

            metrics = {
                'cluster_idx': cluster_idx,
                'star_count': len(star_ages_in_cluster),
                'age_range': age_range,
                't25_t75': t25_t75
            }

            if t25_t75 < threshold_fraction and burst_clusters.get(cluster_idx) is None:
                burst_clusters[cluster_idx] = metrics

    # Save all metrics for further analysis
    df = pd.DataFrame(burst_clusters.values())
    df.to_csv(f"{analysis_path}cluster_formation_metrics_3.csv", index=False)
    
    print(f"Found {len(burst_clusters)} burst clusters out of {len(fuzzy_clusters)} total clusters")
    return burst_clusters


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

    
    # Save all metrics for further analysis
    df = pd.DataFrame(burst_clusters.values())
    df.to_csv(f"{analysis_path}cluster_formation_metrics_2.csv", index=False)
    
    print(f"Found {len(burst_clusters)} burst clusters out of {len(fuzzy_clusters)} total clusters")
    return burst_clusters




def calculateFuzzyCatWindowSize(snapshotFilePaths, lastSnapshotNumber, ageOfTheUniverse):

    lastSnapshot = snapshotFilePaths[-1]

    # Load the simulation snapshot
    simulation = pb.load(lastSnapshot)

    # Take only the largest halo and make it face-on (stellar disk is in the x-y plane)
    mainHalo = simulation.halos()[0]
    pb.analysis.angmom.faceon(mainHalo)
    mainHalo.physical_units()


    # Get the virial radius and mass
    rmax = mainHalo.properties['Rmax']*10**3 #in pc
    vmax = mainHalo.properties['Vmax'] #in km/s

    orbital_time = (2*np.pi*rmax*3.1*10**13/vmax)/60/60/24/365/10**6 # in Myrs

    fuzzycat_window = int(np.ceil(orbital_time*lastSnapshotNumber/ageOfTheUniverse))
    print("Fuzzycat sliding window size: " + str(fuzzycat_window))
    return fuzzycat_window


def save_star_ages(snapshot_file_paths, working_directory_path):
    
    analysis_dir = f"{working_directory_path}further_Cluster_origin_analysis_2/"
    star_ages_dir = f"{analysis_dir}star_ages/"

    n_snapshots = len(snapshot_file_paths)

    #save star ages for faster access
    for snap_idx, snapshot_path in enumerate(snapshot_file_paths):
        if os.path.exists(f"{star_ages_dir}star_ages_{snap_idx:03d}.npy"):
            continue
        snapshot_name = os.path.basename(snapshot_path)
        print(f"Processing snapshot {snap_idx+1}/{n_snapshots}: {snapshot_name}")
        
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

        # Create a 2D array with star IDs and ages
        stars = np.column_stack((star_ids, star_ages.in_units('Gyr')))

        # Save ages for this snapshot
        np.save(f"{star_ages_dir}star_ages_{snap_idx:03d}.npy", stars)


if __name__ == "__main__":

    """Run phase-temporal clustering pipeline :)
    """
    import os
    import gc
    import glob

    import numpy as np
    import pynbody as pb
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    import ffmpeg

    from astrolink import AstroLink
    from fuzzycat import FuzzyCat, FuzzyPlots


    # Choose a particle from ['dm', 'stars', 'gas', 'stars_gas', 'stars_gas_dm'] to cluster
    particleName = 'stars'

    # Choose the number of snapshots to analyze from ['all', int] -> will get last x snapshots
    snapshots = 'all'

    #choose significance for astrolink from ['auto', digit]
    significance = 'auto'

    # Set up the working directory
    galaxyFolderName = '8.26e11_zoom_2_new_run'
    #workingDirectoryPath = f"/mnt/storage/samuel_data/nihao_uhd_{galaxyFolderName}_{particleName}_{snapshots}_snapshots_S={significance}/"
    workingDirectoryPath = f"/mnt/storage/samuel_data/1_2_nihao_uhd_8.26e11_zoom_2_new_run_stars_all_snapshots_S_default/"
    #workingDirectoryPath = f"/mnt/storage/samuel_data/nihao_uhd_{galaxyFolderName}_{particleName}_last_100_snapshots_S_5/"

    if not os.path.exists(workingDirectoryPath):
        os.makedirs(workingDirectoryPath)
        os.makedirs(f"{workingDirectoryPath}Clusters_raw/")
        os.makedirs(f"{workingDirectoryPath}Clusters_iord/")
        os.makedirs(f"{workingDirectoryPath}Clusters/")
        os.makedirs(f"{workingDirectoryPath}Cluster_plots/")



    # Extract the path pattern without the specific significance value
    path_parts = workingDirectoryPath.rsplit("S=", 1)
    path_pattern = path_parts[0] + "S=*"

    # Find all directories matching this pattern
    matching_dirs = glob.glob(path_pattern)

    # Filter out the current directory (if it exists)
    current_dir = workingDirectoryPath.rstrip("/")
    other_significance_dirs = [d for d in matching_dirs if d != current_dir]

    # Find directories that contain the Astrolink_objects folder
    dir_with_astrolink = ""
    rerun = False

    for dir_path in other_significance_dirs:
        astrolink_path = os.path.join(dir_path, "Astrolink_objects/")
        if os.path.exists(astrolink_path) and os.path.isdir(astrolink_path):
            if len(os.listdir(astrolink_path)) == snapshots + 1:
                dir_with_astrolink = astrolink_path
                rerun = True
                break
            else:
                print(f"Found Astrolink directory with {len(os.listdir(astrolink_path))} snapshots, expected {snapshots + 1}")

    if dir_with_astrolink == "":
        print("No other Astrolink directories found")
        dir_with_astrolink = f"{workingDirectoryPath}Astrolink_objects/"
        os.makedirs(dir_with_astrolink, exist_ok=True)
        rerun = True


    # Get the simulation snapshot file paths
    simulationDirectoryPath = f"/mnt/storage/_data/nihao/nihao_uhd/{galaxyFolderName}/"
    snapshotFilePrefix = '8.26e11.'
    if(snapshots == 'all'):
        snapshotNumberRange = range(1164, 2001)
    else:
        snapshotNumberRange = range(2000 - snapshots, 2001)
        #snapshotNumberRange = range(1900, 1900 + snapshots + 1)
    snapshotFilePaths = [f"{simulationDirectoryPath}{snapshotFilePrefix}{i:05}" for i in snapshotNumberRange]

    # Get merger tree info files for AHF comparison movie
    mtreeIdxFilePaths = [fileName for fileName in os.listdir(simulationDirectoryPath) if fileName.endswith('.AHF_mtree_idx') and int(fileName.split('.')[2]) in snapshotNumberRange]
    reorder = np.argsort([int(fileName.split('.')[2]) for fileName in mtreeIdxFilePaths])
    mtreeIdxFilePaths = [f"{simulationDirectoryPath}{mtreeIdxFilePaths[i]}" for i in reorder]

    # Info for the clustering pipeline
    nSamples = len(snapshotFilePaths)
    minLongevityOfFuzzyClusters = 10 # The minimum life-span of fuzzy clusters in Mega-years
    ageOfTheUniverse = 13800 # Age of the Universe in Mega-years
    # Calculate the minStability parameter so that fuzzy clusters live for at least `minLongevityOfFuzzyClusters` Myrs`
    minStability = minLongevityOfFuzzyClusters*(snapshotNumberRange.stop - 1)/(ageOfTheUniverse*snapshotNumberRange.step*nSamples)
    # Choose appropriate axis limits (in kpc) for the movie
    axisLimits = 100

    # Calculate movie frame rate so that 100 Myrs pass every second
    frameRate = 100*(snapshotNumberRange.stop - 1)/(ageOfTheUniverse*snapshotNumberRange.step)


    # Do clustering over snapshots with AstroLink
    #findAndSaveClustersFromSnapshots(snapshotFilePaths, workingDirectoryPath, particleName, nSamples, significance, rerun, dir_with_astrolink)

    #calculate fuzzycat window size
    #fuzzycat_window = calculateFuzzyCatWindowSize(snapshotFilePaths, snapshotNumberRange.stop - 1, ageOfTheUniverse)


    # Run FuzzyCat on AstroLink clusters

    #runFuzzyCatOnClustersFromSnapshots(workingDirectoryPath, nSamples, minStability, fuzzycat_window)

    # Make movie of stable clusters over time
    #makeMovieOfFuzzyClustersOverTime(snapshotFilePaths, workingDirectoryPath, particleName, axisLimits, frameRate)

    #do analysis on mixed gas and star clusters to see if stars are forming in the gas clusters
    #analyze_mixed_clusters(snapshotFilePaths, workingDirectoryPath, axisLimits, frameRate)
    #analysisresult(snapshotFilePaths, workingDirectoryPath, axisLimits, frameRate)

    #plotTemperatureCuts(snapshotFilePaths, workingDirectoryPath, particleName, axisLimits, frameRate)

    #star_cluster_analysis(snapshotFilePaths, workingDirectoryPath, axisLimits, frameRate)
    #identify_burst_clusters(snapshotFilePaths, workingDirectoryPath, threshold_fraction=0.8, time_window=0.2)

    #analyze_all_burst_clusters(snapshotFilePaths, workingDirectoryPath)

    #save_star_ages(snapshotFilePaths, workingDirectoryPath)

    analysis_dir = f"{workingDirectoryPath}further_Cluster_origin_analysis_2/"

    burst_clusters = pd.read_csv(f"{analysis_dir}/cluster_formation_metrics_3.csv")

    make_movie_of_fuzzy_star_forming_clusters(burst_clusters, analysis_dir, workingDirectoryPath, snapshotFilePaths, axisLimits, frameRate)