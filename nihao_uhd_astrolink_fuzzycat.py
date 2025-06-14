import os
import gc
import glob
import logging
import warnings

import numpy as np
import pynbody as pb
import matplotlib.pyplot as plt
import matplotlib.colors as col
import ffmpeg
import random

from astrolink import AstroLink
from fuzzycat import FuzzyCat, FuzzyPlots

from astrolink.io import loadAstroLinkObject
from astrolink.io import saveAstroLinkObject

def loadGalaxyAsArrays(snapshotFilePath, particleName, tagging):
    """Returns the main halo data, from the simulation file `snapshotFilePath`,
    for particle `particleName`, in the feature spaces specified by
    `featureSpaceNames`.
    """
    # Feature space names for the particles
    featureSpaceNames = ['pos', 'vel']
    chemical_abundances = ['FeMassFrac', 'OxMassFrac']

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
        # Dynamical features (pos + vel)
        dyn_features = np.column_stack([mainHalo.stars[feature] for feature in featureSpaceNames])
        dyn_features -= centre  # Centre only dynamics
        if tagging == 'chemical':
            featureSpace = np.column_stack([mainHalo.stars[feature] for feature in chemical_abundances])
        elif tagging == 'chemodynamical':
            chem_features = np.column_stack([mainHalo.stars[feature] for feature in chemical_abundances])
            featureSpace = np.hstack([dyn_features, chem_features])
        else:
            featureSpace = dyn_features  # Just dynamics
        starsIDs = mainHalo.stars['iord']
        starMasses = np.array(mainHalo.stars['mass'])
        return featureSpace, starsIDs, starMasses, simulation
    if particleName == 'gas':
        gas = np.column_stack([mainHalo.gas[feature] for feature in featureSpaceNames])
        gas -= centre
        gasIDs = mainHalo.gas['iord']
        gasMasses = np.array(mainHalo.gas['mass'])
        gasTemperatures = np.array(mainHalo.gas['temp'])
        return gas, gasIDs, gasMasses, gasTemperatures, simulation
    if particleName == 'stars_gas':
        gas = np.column_stack([mainHalo.gas[feature] for feature in featureSpaceNames])
        gas -= centre
        gasIDs = mainHalo.gas['iord']
        if tagging == 'chemical':
            featureSpaceNames = chemical_abundances
        elif tagging == 'chemodynamical':
            featureSpaceNames.extend(chemical_abundances)
        stars = np.column_stack([mainHalo.stars[feature] for feature in featureSpaceNames])
        stars -= centre
        starsIDs = mainHalo.stars['iord']
        starMasses = np.array(mainHalo.stars['mass'])
        gasMasses = np.array(mainHalo.gas['mass'])
        particles = np.vstack([stars, gas])
        particleIDs = np.hstack([starsIDs, gasIDs])
        allMasses = np.hstack([starMasses, gasMasses])

        return particles, particleIDs, allMasses, simulation
    
    if particleName == 'stars_gas_dm':
        gas = np.column_stack([mainHalo.gas[feature] for feature in featureSpaceNames])
        gas -= centre
        darkMatter = np.column_stack([mainHalo.dm[feature] for feature in featureSpaceNames])
        darkMatter -= centre
        if tagging == 'chemical':
            featureSpaceNames = chemical_abundances
        elif tagging == 'chemodynamical':
            featureSpaceNames.extend(chemical_abundances)
        stars = np.column_stack([mainHalo.stars[feature] for feature in featureSpaceNames])
        stars -= centre
        starsIDs = mainHalo.stars['iord']
        starMasses = np.array(mainHalo.stars['mass'])
        gasIDs = mainHalo.gas['iord']
        gasMasses = np.array(mainHalo.gas['mass'])
        darkMatterIDs = mainHalo.dm['iord']
        darkMatterWeights = np.array(mainHalo.dm['mass'])
        particles = np.vstack([stars, gas, darkMatter])
        particleIDs = np.hstack([starsIDs, gasIDs, darkMatterIDs])
        allMasses = np.hstack([starMasses, gasMasses, darkMatterWeights])

        return particles, particleIDs, allMasses, simulation
        

def findAndSaveClustersFromSnapshots(snapshotFilePaths, workingDirectoryPath, particleName, nSamples, significance, rerun, tagging, dir_with_astrolink):
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
            particleArr, particleIDs, weights,  _ = loadGalaxyAsArrays(snapshotFilePath, particleName, tagging=tagging)


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
    fc = FuzzyCat(nSamples, nPoints, workingDirectoryPath, minStability = minStability, checkpoint = True, verbose = 2)
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


def makeMovieOfFuzzyClustersOverTimeByParticleType(snapshotFilePaths, workingDirectoryPath, particleType, axisLimits, frameRate, plot_labels, tagging, color_mapping = None):
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
        
        particleArr = loadGalaxyAsArrays(snapshotFilePath, particleName, tagging)[0]
            

        print(f"Loading clusters of {particleName} particles from snapshot {snapshotFileName}   \t\t", end = '\r')
        # Load AstroLink clusters (found in this snapshot) that belong to the fuzzy clusters from FuzzyCat
        clusters_raw, fuzzyLabels = [], []
        for clusterFileName, whichFuzzyClst in zip(clusterFileNames, whichCluster):
            clstSnapshot = int(clusterFileName.split('_')[0])
            if whichFuzzyClst != -1 and clstSnapshot == index:
                cluster_raw = np.load(workingDirectoryPath + 'Clusters_raw/' + clusterFileName)
                clusters_raw.append(cluster_raw)
                fuzzyLabels.append(whichFuzzyClst)


        # Make plot of clusters
        print(f"Plotting {len(clusters_raw)} astrolink {particleType} clusters for {snapshotFileName}")
        paintLabelsOntoSnapshot(
            particleArr, 
            clusters_raw, 
            fuzzyLabels, 
            saveFileNameStem, 
            snapshotFileName, 
            axisLimits,
            plot_labels,
            color_mapping = color_mapping,
        )

    # Make movie
    print(f"Creating movie for {particleType}...")
    (
        ffmpeg
        .input(f"{saveFileNameStem}*.png", pattern_type='glob', framerate=frameRate)
        .output(f"{workingDirectoryPath}{workingDirectoryPath.split('/')[-2]}_{particleType}_movie_without_labels.mp4")
        .run()
    )


def makeMovieOfFuzzyClustersOverTime(snapshotFilePaths, workingDirectoryPath, particleName, axisLimits, frameRate, plot_labels, tagging):
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
            snapshotFilePaths, workingDirectoryPath, particleType, axisLimits, frameRate, plot_labels, tagging, color_mapping,
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
        
def burst_cluster_cdf_analysis(burst_clusters, snapshot_file_paths, working_directory_path, age_distribution_plots_dir, star_data_dir, analysis_dir, n_snapshots, ordering, fuzzy_clusters):
    """
    Analyzes burst clusters and creates CDF plots and a movie for each cluster over time. Then selects the top 50 clusters by burst size and creates a movie of them in the simulation.
    """
    for cluster_idx in burst_clusters:

        burst_snapshot = burst_clusters[cluster_idx]['burst_snapshot']
        #convert birth and death to relative years
        cluster_detected_time = burst_clusters[cluster_idx]['cluster_detected_snapshot']*13.8/2000
        cluster_lost_time = burst_clusters[cluster_idx]['cluster_lost_snapshot']*13.8/2000
        burst_snapshot_time = burst_snapshot*13.8/2000
        
        cluster_dir = f"{age_distribution_plots_dir}cluster_{cluster_idx}/"
        os.makedirs(cluster_dir, exist_ok=True)

        all_member_ids_until_now = np.array([], dtype=int)


        fuzzy_start_idx, fuzzy_end_idx = fuzzy_clusters[cluster_idx]
        astrolink_cluster_ids_in_fuzzycat_cluster = ordering[fuzzy_start_idx:fuzzy_end_idx]
        clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")
        index = f"{snap_idx:03d}"
        # Get AstroLink clusters for this snapshot that belong to the current FuzzyCat cluster
        cluster_filenames = [clusterFileNames[idx] for idx in astrolink_cluster_ids_in_fuzzycat_cluster if clusterFileNames[idx].startswith(index)]
        print(f"Found {len(cluster_filenames)} AstroLink clusters in FuzzyCat cluster {cluster_idx}")

   
        start_idx = max(0, burst_snapshot - 5)

        for snap_idx in range(start_idx, n_snapshots):
            snapshot_path = snapshot_file_paths[snap_idx]
            snapshot_name = os.path.basename(snapshot_path)
            print(f"Processing snapshot {snap_idx+1}/{n_snapshots}: {snapshot_name}")

            # Load the star data for the current snapshot
            star_data = np.load(f"{star_data_dir}star_data_{snap_idx:03d}.npy")
            star_ids = star_data[:,0]
            star_ages = star_data[:,1]
            
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

            #plot age distribution of stars in the cluster
            this_snapshot_star_ages = star_ages[np.isin(star_ids, member_ids)]
            star_ages_for_cluster_until_now = star_ages[np.isin(star_ids, all_member_ids_until_now)]

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
    movies_dir = f"{age_distribution_plots_dir}movies/"
    os.makedirs(movies_dir, exist_ok=True)
    for folder in os.listdir(age_distribution_plots_dir):
        if not folder.startswith('cluster_'):
            continue
        cluster_idx = folder.split('_')[1]
        cluster_dir = f"{age_distribution_plots_dir}cluster_{cluster_idx}/"
        
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


def mass_distribution_analysis(burst_clusters, working_directory_path, star_data_dir, mass_distributions_dir, n_snapshots, ordering, fuzzy_clusters):
    """
    Analyzes the mass distribution of stars in burst clusters over time and creates plots for each cluster.
    First plot is a 2D heatmap of mass over time, showing how the density of star masses in the cluster changes over time.
    Second plot is how the mean and median mass the cluster changes over time for each cluster.
    """
    burst_clusters_masses_all_snapshots = { c: {} for c in burst_clusters }

    #only use a few random burst clusters for now for faster analysis
    num_clusters_to_select = 5
    all_cluster_ids = list(burst_clusters.keys())
    selected_cluster_ids = random.sample(all_cluster_ids, num_clusters_to_select)
    burst_clusters = {cluster_id: burst_clusters[cluster_id] for cluster_id in selected_cluster_ids}

    for cluster_idx in burst_clusters:
        cluster_dir = f"{mass_distributions_dir}cluster_{cluster_idx}/"
        os.makedirs(cluster_dir, exist_ok=True)

        fuzzy_start_idx, fuzzy_end_idx = fuzzy_clusters[cluster_idx]
        astrolink_cluster_ids_in_fuzzycat_cluster = ordering[fuzzy_start_idx:fuzzy_end_idx]
        # Ensure the path to clusterFileNames.npy is correct
        clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy") 
        
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
                cluster_data_path = os.path.join(workingDirectoryPath, "Clusters_iord", cluster_filename)
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

        # --- Plot 1: 2D Heatmap of Mass Distribution over Time ---
    
        # Calculate histogram data for each snapshot
        # hist_data_list will store 1D arrays of histogram counts (densities)
        hist_data_list = []
        for current_masses in mass_list_with_data: # Already filtered for non-empty lists
            counts, _ = np.histogram(current_masses, bins=bins, density=True)
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
        plt.colorbar(label="Normalized Count (density)")
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
    
        outfn_heatmap = os.path.join(cluster_dir, f"mass_dist_heatmap.png")
        plt.savefig(outfn_heatmap, dpi=200)
        plt.close()
        print(f"  → Saved heatmap (axes swapped) for Cluster {cluster_idx} at {outfn_heatmap}")

        
        # --- Plot 2: Summary Statistics (Mean, Median, IQR) over Time ---
        
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

    #save all snapshots' masses for later use
    np.save(f"{mass_distributions_dir}burst_clusters_masses_all_snapshots.npy", burst_clusters_masses_all_snapshots, allow_pickle=True)
    
def power_law_analysis(power_law_analysis_dir, mass_distributions_dir, n_snapshots):
    """
    Analyses the power law slope of the burst clusters.
    We initially bin the cluster masses using the freedman-diaconis rule. 
    Then we assume a beta distribution for the number of clusters in each bin and perform a bootstrap analysis to estimate the slope and its error.
    We then plot the resulting slope and error for each snapshot.
    """
    
    all_snapshot_masses = np.load(f"{mass_distributions_dir}burst_clusters_masses_all_snapshots.npy", allow_pickle=True).item()
    nonempty_clusters = [c for c in all_snapshot_masses if any(len(m) > 0 for m in all_snapshot_masses[c].values())]
    results_snapshots = []
    results_mean_slopes = []
    results_slope_errors = []
    num_bootstraps = 1000
    make_individual_plots = True  # Set to False to skip individual plots and only save overview over slopes
    individual_plots_dir = f"{power_law_analysis_dir}individual_plots/"
    os.makedirs(individual_plots_dir, exist_ok=True)

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
        if len(cluster_masses) < 20:
            print(f"Not enough clusters found for snapshot {final_snapshot_idx}. Skipping power law analysis.")
            continue
        log_masses = np.log10(cluster_masses)  # Convert to log scale for power law fitting
        q75, q25 = np.percentile(log_masses, [75, 25])
        iqr = q75 - q25
        # Use IQR to determine bin_width
        if iqr > 0:
            bin_width = 2*iqr*(len(log_masses)**(-1/3))  # Freedman-Diaconis rule
            num_bins = int(np.ceil((log_masses.max() - log_masses.min()) / bin_width))
        else:
            num_bins = 10  # Fallback if IQR is zero
        num_bins = max(2, min(num_bins, 100))  # Ensure num_bins is between 5 and 50
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
            dM[dM==0] = 1  # Avoid division by zero
            resampled_dN_dM = resampled_dN / dM
            mask = (10**log_bin_centers >= 1e4) & (resampled_dN > 0)  # Filter for Mcl ≥ 1e6 and dN > 0
            x_fit_data = log_bin_centers[mask]
            y_fit_data = np.log10(resampled_dN_dM[mask])
            if x_fit_data.size > 1:
                slope, y_intercept = np.polyfit(x_fit_data, y_fit_data, 1)
                bootstrap_slopes.append(slope)
                bootstrap_y_intercepts.append(y_intercept)
                dN_dM_array.append(resampled_dN_dM)
        
        if(len(bootstrap_slopes) > 20):
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
            plt.plot(10**x_fit_line, 10**y_fit_line, 'r--', color='red', label=f"Fit: slope = {slope:.2f}")
            plt.grid(True, linestyle='--', alpha=0.5, axis='both')
            plt.legend()    
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Cluster Mass")
            plt.ylabel("Number of Clusters per delta log(M)")
            plt.title("Cluster Mass Function")
            plt.savefig(f"{individual_plots_dir}cluster_mass_function_snapshot{final_snapshot_idx}.png", dpi=200)
            plt.close()
            print(f"Saved cluster mass function plot for snapshot {final_snapshot_idx} to {individual_plots_dir}cluster_mass_function_snapshot{final_snapshot_idx}.png")
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
    plt.savefig(f"{power_law_analysis_dir}slope_evolution_plot2.png", dpi=300)


def star_cluster_analysis(snapshot_file_paths, working_directory_path, axisLimits, frameRate):
    # Create output directories
    analysis_dir = f"{working_directory_path}star_cluster_analysis/"
    age_distribution_plots_dir = f"{analysis_dir}age_distributions/"
    mass_distributions_dir = f"{analysis_dir}mass_distributions/"
    power_law_analysis_dir = f"{analysis_dir}power_law_analysis/"
    star_data_dir = f"{analysis_dir}star_data/"
    os.makedirs(star_data_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(age_distribution_plots_dir, exist_ok=True)
    os.makedirs(mass_distributions_dir, exist_ok=True)
    os.makedirs(power_law_analysis_dir, exist_ok=True)
    
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

    #------ load and save all star data for each snapshot (ids, ages, masses, positions)----------
    #save_star_data(snapshot_file_paths, star_data_dir)

    #------ identify burst clusters from fuzzy clusters and save their metrics ----------
    #fast_identify_burst_clusters(snapshot_file_paths, working_directory_path, analysis_dir, star_data_dir, n_snapshots, n_clusters, ordering, fuzzy_clusters, clustered_ids)
    burst_clusters = pd.read_csv(f"{analysis_dir}cluster_formation_metrics.csv").set_index('cluster_idx').to_dict(orient='index')
    print(f"Found {len(burst_clusters)} burst clusters out of {len(fuzzy_clusters)} total clusters")

    #------ plot the cdf of each birth cluster over time and create movies of cdf for every cluster, then create movie of simulation with top 50 star forming clusters ----------
    #burst_cluster_cdf_analysis(burst_clusters, snapshot_file_paths, working_directory_path, age_distribution_plots_dir, star_data_dir, analysis_dir, n_snapshots, ordering, fuzzy_clusters)
    
    #------ plot mass distributions of stars in each star forming cluster ----------
    #mass_distribution_analysis(burst_clusters, working_directory_path, star_data_dir, mass_distributions_dir, n_snapshots, ordering, fuzzy_clusters)

    #------ analyze the power law of the mass distribution of star forming clusters ----------
    power_law_analysis(power_law_analysis_dir, mass_distributions_dir, n_snapshots)


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

    #get the snapshot ranges for each fuzzycat cluster
    fuzzy_cluster_snapshot_ranges = np.zeros((n_clusters, 2), dtype=int)
    for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):
        astrolink_cluster_ids_in_fuzzycat_cluster = ordering[start_idx:end_idx]
        clusterFileNames = np.load(f"{working_directory_path}clusterFileNames.npy")
        cluster_filenames = [clusterFileNames[idx] for idx in astrolink_cluster_ids_in_fuzzycat_cluster]
        # Get the snapshot indices for this cluster
        snapshot_indices = set([int(filename.split('_')[0]) for filename in cluster_filenames])
        fuzzy_cluster_snapshot_ranges[cluster_idx] = [min(snapshot_indices), max(snapshot_indices)]

    csv_filepath = os.path.join(analysis_path, "cluster_formation_metrics.csv")
    header_columns = ['cluster_idx', 'cluster_start_snapshot', 'cluster_end_snapshot', 'star_count', 't25_t75', 'burst_snapshot']
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

        for snap_idx in range(cluster_start_snapshot - 4, cluster_start_snapshot + 1):
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

            # Calculate t25-t75 (time to form middle 50% of stars)
            t25 = np.percentile(sorted_ages, 25)
            t75 = np.percentile(sorted_ages, 75)
            t25_t75 = t75 - t25

            cluster_detected_time = (snap_idx - cluster_start_snapshot)*13.8/2000
            cluster_lost_time = (snap_idx - cluster_end_snapshot)*13.8/2000


            if t25_t75 < threshold_fraction and np.abs(cluster_detected_time - snap_idx*13.8/2000) < 1.0:
                metrics = {
                'cluster_idx': cluster_idx,
                'cluster_start_snapshot': cluster_start_snapshot,
                'cluster_end_snapshot': cluster_end_snapshot,
                'star_count': len(star_ages_in_cluster),
                't25_t75': t25_t75,
                'burst_snapshot': snap_idx,
                }
                df_row = pd.DataFrame([metrics])
                df_row.to_csv(csv_filepath, mode='a', header=False, index=False)
                burst_recorded_for_this_cluster_idx_this_run = True


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

def save_star_data(snapshot_file_paths, star_data_dir):

    n_snapshots = len(snapshot_file_paths)

    #save star ages for faster access
    for snap_idx, snapshot_path in enumerate(snapshot_file_paths):
        if os.path.exists(f"{star_data_dir}star_data_{snap_idx:03d}.npy"):
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
        star_masses = star_particles['mass']
        star_pos = star_particles['pos']

        # Create an array with the above parameters

        star_ages = star_ages.in_units('Gyr')  # Convert ages to Gyr
        star_masses = star_masses.in_units('Msol')  # Convert masses to Msun
        stars = np.column_stack((star_ids, star_ages, star_masses, star_pos))

        # Save ages for this snapshot
        np.save(f"{star_data_dir}star_data_{snap_idx:03d}.npy", stars)


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


if __name__ == "__main__":

    """Run phase-temporal clustering pipeline :)
    """
    import os
    import gc
    import glob

    import numpy as np
    import pynbody as pb
    import pandas as pd
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

    # should plots have labels in the movie?
    plot_labels = False

    # Choice of tagging from ['dynamical' (standard), 'chemical', 'chemodynamical']
    tagging = 'chemodynamical'

    # The minimum life-span of fuzzy clusters in Mega-years
    minLongevityOfFuzzyClusters = 230 
    
    # Age of the Universe in Mega-years
    ageOfTheUniverse = 13800 

    # Choose appropriate axis limits (in kpc) for the movie
    axisLimits = 100

    # Set up the working directory
    galaxyFolderName = '2.79e12_zoom_6_rerun'
    #workingDirectoryPath = f"/mnt/storage/samuel_data/nihao_uhd_{galaxyFolderName}_{particleName}_last_100_snapshots_S_5/"
    #workingDirectoryPath = f"/home/samuel_data/nihao_uhd_{galaxyFolderName}_{particleName}_{snapshots}_snapshots_{tagging}_tagging_S={significance}/"
    workingDirectoryPath = f"/home/samuel_data/nihao_uhd_{galaxyFolderName}_{particleName}_{snapshots}_snapshots_S={significance}_without_window/"

    if not os.path.exists(workingDirectoryPath):
        os.makedirs(workingDirectoryPath)
        os.makedirs(f"{workingDirectoryPath}Clusters_raw/")
        os.makedirs(f"{workingDirectoryPath}Clusters_iord/")
        os.makedirs(f"{workingDirectoryPath}Clusters/")
        os.makedirs(f"{workingDirectoryPath}Cluster_plots/")

    # Configure the logger
    logging.basicConfig(
        filename=f"{workingDirectoryPath}log.log",    # Specify the log file name
        level=logging.INFO,               # Set the logging level to INFO
        format='%(asctime)s - %(levelname)s - %(message)s'  # Define the log message format
    )

    logging.info(f"Starting the clustering pipeline for {particleName} with {snapshots} snapshots and significance {significance}, tagging {tagging}")
    logging.info(f"Parameters: minLongevityOfFuzzyClusters={minLongevityOfFuzzyClusters}, ageOfTheUniverse={ageOfTheUniverse}, axisLimits={axisLimits}")
    
    # Get the simulation snapshot file paths
    simulationDirectoryPath = f"/home/_data/nihao/nihao_uhd/{galaxyFolderName}/"
    snapshotFilePrefix = '2.79e12.'
    if(snapshots == 'all'):
        snapshotNumberRange = range(1800, 2001)
        snapshots = len(snapshotNumberRange)
    else:
        snapshotNumberRange = range(2000 - snapshots - 1, 2001)
        #snapshotNumberRange = range(1900, 1900 + snapshots + 1)
    snapshotFilePaths = [f"{simulationDirectoryPath}{snapshotFilePrefix}{i:05}" for i in snapshotNumberRange]



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

    # for dir_path in other_significance_dirs:
    #     astrolink_path = os.path.join(dir_path, "Astrolink_objects/")
    #     if os.path.exists(astrolink_path) and os.path.isdir(astrolink_path):
    #         if len(os.listdir(astrolink_path)) == snapshots:
    #             dir_with_astrolink = astrolink_path
    #             rerun = True
    #             break
    #         else:
    #             print(f"Found Astrolink directory with {len(os.listdir(astrolink_path))} snapshots, expected {snapshots + 1}")

    if dir_with_astrolink == "":
        print("No other Astrolink directories found")
        dir_with_astrolink = f"{workingDirectoryPath}Astrolink_objects/"
        os.makedirs(dir_with_astrolink, exist_ok=True)


    logging.info(f"Using Astrolink directory: {dir_with_astrolink}")

    # Info for the clustering pipeline
    nSamples = len(snapshotFilePaths)
    # Calculate the minStability parameter so that fuzzy clusters live for at least `minLongevityOfFuzzyClusters` Myrs`
    minStability = minLongevityOfFuzzyClusters*(snapshotNumberRange.stop - 1)/(ageOfTheUniverse*snapshotNumberRange.step*nSamples)

    # Calculate movie frame rate so that 100 Myrs pass every second
    frameRate = 100*(snapshotNumberRange.stop - 1)/(ageOfTheUniverse*snapshotNumberRange.step)

    #logging.info("Starting astrolink...")
    #Do clustering over snapshots with AstroLink
    #findAndSaveClustersFromSnapshots(snapshotFilePaths, workingDirectoryPath, particleName, nSamples, significance, rerun, tagging, dir_with_astrolink)

    # logging.info("Finished Astrolink, starting FuzzyCat...")
    # #calculate fuzzycat window size
    # #fuzzycat_window = calculateFuzzyCatWindowSize(snapshotFilePaths, snapshotNumberRange.stop - 1, ageOfTheUniverse)
    # fuzzycat_window = None
    # logging.info(f"FuzzyCat window size: {fuzzycat_window}")
    # #Run FuzzyCat on AstroLink clusters
    # runFuzzyCatOnClustersFromSnapshots(workingDirectoryPath, nSamples, minStability, fuzzycat_window)
    # logging.info("Finished FuzzyCat, starting plotting...")
    # #Make movie of stable clusters over time
    # makeMovieOfFuzzyClustersOverTime(snapshotFilePaths, workingDirectoryPath, particleName, axisLimits, frameRate, plot_labels, tagging)
    # logging.info("Finished plotting")

    #plotTemperatureCuts(snapshotFilePaths, workingDirectoryPath, particleName, axisLimits, frameRate)
    #logging.info("Saving star ages for analysis...")
    #save_star_data(snapshotFilePaths, workingDirectoryPath)

    logging.info("Starting star cluster analysis...")
    star_cluster_analysis(snapshotFilePaths, workingDirectoryPath, axisLimits, frameRate)
    logging.info("Finished star cluster analysis")