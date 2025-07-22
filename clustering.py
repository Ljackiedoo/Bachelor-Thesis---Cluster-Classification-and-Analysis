import os
import glob


def performClustering(
    particleName,
    snapshots,
    significance,
    tagging,
    plot_labels,
    minLongevityOfFuzzyClusters,
    ageOfTheUniverse,
    axisLimits,
    workingDirectoryPath,
    simulationDirectoryPath,
    snapshotFilePaths
):
    # Check if astrolink files are available
    reuse_astrolink, astrolink_filepath = doAstrolinkFilesExist(workingDirectoryPath, snapshots, particleName, significance, tagging)
    # Info for the clustering pipeline
    nSamples = len(snapshotFilePaths)
    # Calculate the minStability parameter so that fuzzy clusters live for at least `minLongevityOfFuzzyClusters` Myrs`
    minStability = minLongevityOfFuzzyClusters*(snapshots.stop - 1)/(ageOfTheUniverse*snapshots.step*nSamples)

    # Calculate movie frame rate so that 100 Myrs pass every second
    frameRate = 100*(snapshots.stop - 1)/(ageOfTheUniverse*snapshots.step)

    #Do clustering over snapshots with AstroLink
    findAndSaveClustersFromSnapshots(snapshotFilePaths, workingDirectoryPath, particleName, nSamples, significance, reuse_astrolink, tagging, astrolink_filepath)

    # Run FuzzyCat to find fuzzy clusters
    runFuzzyCatOnClustersFromSnapshots(workingDirectoryPath, nSamples, minStability)

    makeMovieOfFuzzyClustersOverTime(snapshotFilePaths, workingDirectoryPath, particleName, axisLimits, frameRate, plot_labels, tagging, sample_rate)


def doAstrolinkFilesExist(workingDirectoryPath, snapshots, particlename, significance, tagging):
    
    # Extract the path pattern without the specific significance value
    path_parts = workingDirectoryPath.split("S=", 1)
    path_pattern = path_parts[0] + "S=*"

    # Find all directories matching this pattern
    matching_dirs = glob.glob(path_pattern)

    # Find directories that contain the Astrolink_objects folder
    dir_with_astrolink = ""
    rerun = False

    for dir_path in matching_dirs:
        astrolink_path = os.path.join(dir_path, f"astrolink_objects_{particlename}_{tagging}/")
        if os.path.exists(astrolink_path) and os.path.isdir(astrolink_path):
            if len(os.listdir(astrolink_path)) == snapshots:
                dir_with_astrolink = astrolink_path
                rerun = True
                break
            else:
                print(f"Found Astrolink directory with {len(os.listdir(astrolink_path))} snapshots, expected {snapshots + 1}")

    if dir_with_astrolink == "":
        print("No other Astrolink directories found")
        dir_with_astrolink = f"{workingDirectoryPath}Astrolink_objects_{particlename}_{tagging}/"
        os.makedirs(dir_with_astrolink, exist_ok=True)
    
    return rerun, dir_with_astrolink

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


def runFuzzyCatOnClustersFromSnapshots(workingDirectoryPath, nSamples, minStability):
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

def makeMovieOfFuzzyClustersOverTimeByParticleType(snapshotFilePaths, workingDirectoryPath, particleType, axisLimits, frameRate, plot_labels, tagging, color_mapping = None):
    """Makes a movie of the fuzzy clusters for a specific particle type (stars or gas) found by AstroLink and FuzzyCat
    as they evolve over time. Optimized for performance.
    """
    # Prepare output path for this particle type
    saveFileNameStem = f"{workingDirectoryPath}Cluster_plots/plotted_clusters_{particleType}_"
    
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

def paintLabelsOntoSnapshot(particleArr, clusters_raw, labels, saveFileNameStem, snapshotFileName, axisLimits, plot_labels, sample_rate=2, withDiskZoomIn = True, color_mapping = None):
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


    sampled_clusters = {}
    
    if sample_rate >= 1:
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
