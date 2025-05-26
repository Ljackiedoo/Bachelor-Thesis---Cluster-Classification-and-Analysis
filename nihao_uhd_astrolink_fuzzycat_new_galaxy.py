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

from astrolink import AstroLink
from fuzzycat import FuzzyCat, FuzzyPlots

from astrolink.io import loadAstroLinkObject
from astrolink.io import saveAstroLinkObject

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


def makeMovieOfFuzzyClustersOverTimeByParticleType(snapshotFilePaths, workingDirectoryPath, particleType, axisLimits, frameRate, plot_labels, color_mapping = None):
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

    # # Pre-organize clusters by snapshot index for faster access
    # print(f"Organizing clusters by snapshot...")
    # clusters_by_snapshot = {}
    # for i, (fileName, fuzzyClustId) in enumerate(zip(clusterFileNames, whichCluster)):
    #     snapshot_idx = int(fileName.split('_')[0])
    #     if fuzzyClustId != -1:
    #         if snapshot_idx not in clusters_by_snapshot:
    #             clusters_by_snapshot[snapshot_idx] = []
    #         clusters_by_snapshot[snapshot_idx].append((fileName, fuzzyClustId))

    # # Cache for particle counts to avoid redundant calculations
    # particle_counts = {}

    # for index, snapshotFilePath in enumerate(snapshotFilePaths):
    #     snapshotFileName = snapshotFilePath.split('/')[-1]
        
    #     # Skip if no clusters for this snapshot
    #     if index not in clusters_by_snapshot:
    #         print(f"No clusters for {particleType} in snapshot {snapshotFileName}")
    #         continue
            
    #     print(f"Processing {particleType} for snapshot {snapshotFileName}...")
        
    #     # Load particle data once per snapshot
    #     particleArr = loadGalaxyAsArrays(snapshotFilePath, particleType)[0]
        
    #     # Get all particle counts for this snapshot if not already cached
    #     if index not in particle_counts:
    #         particle_counts[index] = {
    #             'stars': loadGalaxyAsArrays(snapshotFilePath, 'stars')[0].shape[0],
    #             'gas': loadGalaxyAsArrays(snapshotFilePath, 'gas')[0].shape[0],
    #             'dm': loadGalaxyAsArrays(snapshotFilePath, 'dm')[0].shape[0]
    #         }
        
    #     # Get proper offsets based on the order of your combined particle array
    #     # (Adjust these according to how your particles are actually ordered)
    #     star_offset = 0
    #     gas_offset = particle_counts[index]['stars']
    #     dm_offset = gas_offset + particle_counts[index]['gas']
        
    #     clusters_raw = []
    #     fuzzyLabels = []
        
    #     # Process only relevant clusters for this snapshot
    #     for clusterFileName, whichFuzzyClst in clusters_by_snapshot[index]:
    #         cluster_raw = np.load(workingDirectoryPath + 'Clusters_raw/' + clusterFileName)
            
    #         # Filter by particle type
    #         if particleType == 'stars':
    #             # Keep only indices that correspond to stars (between star_offset and gas_offset)
    #             valid_indices = cluster_raw[(cluster_raw >= star_offset) & (cluster_raw < gas_offset)]
    #             valid_indices = valid_indices - star_offset  # Adjust to start from 0
                
    #         elif particleType == 'gas':
    #             # Keep only indices that correspond to gas (between gas_offset and dm_offset)
    #             valid_indices = cluster_raw[(cluster_raw >= gas_offset) & (cluster_raw < dm_offset)]
    #             valid_indices = valid_indices - gas_offset  # Adjust to start from 0
                
    #         elif particleType == 'dm':
    #             # Keep only indices that correspond to dark matter (after dm_offset)
    #             valid_indices = cluster_raw[cluster_raw >= dm_offset]
    #             valid_indices = valid_indices - dm_offset  # Adjust to start from 0
            
    #         if len(valid_indices) > 0:
    #             clusters_raw.append(valid_indices)
    #             fuzzyLabels.append(whichFuzzyClst)


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


def makeMovieOfFuzzyClustersOverTime(snapshotFilePaths, workingDirectoryPath, particleName, axisLimits, frameRate, plot_labels):
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
            snapshotFilePaths, workingDirectoryPath, particleType, axisLimits, frameRate, plot_labels, color_mapping
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
        

def analyze_mixed_clusters(snapshot_file_paths, working_directory_path, axis_limits, frame_rate):
    """Analyzes mixed gas-star clusters to track star formation within clusters.
    
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
    # Identify significant clusters (more sophisticated criteria)
    # Consider both mass and activity in star formation



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
    
def star_cluster_analysis(snapshot_file_paths, working_directory_path, axisLimits, frameRate):
    # Create output directories
    analysis_dir = f"{working_directory_path}Cluster_origin_analysis/"
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

    n_snapshots = len(snapshot_file_paths)
    n_clusters = len(fuzzy_clusters)
    print(f"Processing {n_snapshots} snapshots for {n_clusters} fuzzy clusters")

    # First, create a directory for each fuzzy cluster
    for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):
        cluster_dir = f"{plots_dir}cluster_{cluster_idx}/"
        os.makedirs(cluster_dir, exist_ok=True)


    for snap_idx, snapshot_path in enumerate(snapshot_file_paths):
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

        for cluster_idx, (start_idx, end_idx) in enumerate(fuzzy_clusters):
            cluster_dir = f"{plots_dir}cluster_{cluster_idx}/"

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

            star_ids_mask = np.isin(member_ids, star_ids)
            if(not np.any(star_ids_mask)):
                print(f"No star particles found in cluster {cluster_idx}")
                continue
            member_ids = member_ids[star_ids_mask]

            #get birth dates of star particles
            star_ages = star_particles['age'][np.isin(star_ids, member_ids)]
            #plot cdf of birth dates
            plt.figure(figsize=(8, 6))
            plt.hist(star_ages, bins=100, cumulative=True, density=True, histtype='step', color='b', linewidth=2)
            plt.xlabel('Age (Gyr)')
            plt.ylabel('Cumulative Fraction')
            plt.title(f'Star Particle Ages in Cluster {cluster_idx} (Snapshot {snap_idx})')
            plt.grid(True, alpha=0.3)
            # Save the plot in the cluster's directory with a consistent naming pattern for the movie
            plt.savefig(f"{cluster_dir}frame_{snap_idx:04d}.png", dpi=300)
            plt.close()
    # After processing all snapshots, create movies for each cluster
    print("Creating evolution movies for each cluster...")
    movies_dir = f"{analysis_dir}movies/"
    os.makedirs(movies_dir, exist_ok=True)
    for cluster_idx, _ in enumerate(fuzzy_clusters):
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

    # The minimum life-span of fuzzy clusters in Mega-years
    minLongevityOfFuzzyClusters = 230 
    
    # Age of the Universe in Mega-years
    ageOfTheUniverse = 13800 

    # Choose appropriate axis limits (in kpc) for the movie
    axisLimits = 100

    # Set up the working directory
    galaxyFolderName = '2.79e12_zoom_6_rerun'
    workingDirectoryPath = f"/mnt/storage/samuel_data/nihao_uhd_{galaxyFolderName}_{particleName}_{snapshots}_snapshots_S={significance}_without_window/"
    #workingDirectoryPath = f"/mnt/storage/samuel_data/nihao_uhd_{galaxyFolderName}_{particleName}_last_100_snapshots_S_5/"

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

    logging.info(f"Starting the clustering pipeline for {particleName} with {snapshots} snapshots and significance {significance}")
    logging.info(f"Parameters: minLongevityOfFuzzyClusters={minLongevityOfFuzzyClusters}, ageOfTheUniverse={ageOfTheUniverse}, axisLimits={axisLimits}")
    
    # Get the simulation snapshot file paths
    simulationDirectoryPath = f"/mnt/storage/_data/nihao/nihao_uhd/{galaxyFolderName}/"
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

    for dir_path in other_significance_dirs:
        astrolink_path = os.path.join(dir_path, "Astrolink_objects/")
        if os.path.exists(astrolink_path) and os.path.isdir(astrolink_path):
            if len(os.listdir(astrolink_path)) == snapshots:
                dir_with_astrolink = astrolink_path
                rerun = True
                break
            else:
                print(f"Found Astrolink directory with {len(os.listdir(astrolink_path))} snapshots, expected {snapshots + 1}")

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

    logging.info("Starting astrolink...")
    # Do clustering over snapshots with AstroLink
    findAndSaveClustersFromSnapshots(snapshotFilePaths, workingDirectoryPath, particleName, nSamples, significance, rerun, dir_with_astrolink)

    logging.info("Finished Astrolink, starting FuzzyCat without window...")
    #calculate fuzzycat window size
    #fuzzycat_window = calculateFuzzyCatWindowSize(snapshotFilePaths, snapshotNumberRange.stop - 1, ageOfTheUniverse)
    #logging.info(f"FuzzyCat window size: {fuzzycat_window}")
    fuzzycat_window = 1

    # Run FuzzyCat on AstroLink clusters
    runFuzzyCatOnClustersFromSnapshots(workingDirectoryPath, nSamples, minStability, fuzzycat_window)
    logging.info("Finished FuzzyCat, starting plotting...")
    # Make movie of stable clusters over time
    makeMovieOfFuzzyClustersOverTime(snapshotFilePaths, workingDirectoryPath, particleName, axisLimits, frameRate, plot_labels)
    logging.info("Finished plotting")
    #do analysis on mixed gas and star clusters to see if stars are forming in the gas clusters
    #analyze_mixed_clusters(snapshotFilePaths, workingDirectoryPath, axisLimits, frameRate)
    #analysisresult(snapshotFilePaths, workingDirectoryPath, axisLimits, frameRate)

    #plotTemperatureCuts(snapshotFilePaths, workingDirectoryPath, particleName, axisLimits, frameRate)

    #star_cluster_analysis(snapshotFilePaths, workingDirectoryPath, axisLimits, frameRate)