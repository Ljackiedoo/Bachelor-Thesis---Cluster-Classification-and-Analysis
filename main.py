from clustering import performClustering
from clustering import makeMovieOfFuzzyClustersOverTime
from analysis import starClusterAnalysis

import os
import gc
import glob
import logging
import warnings

import numpy as np
import pynbody as pb
import matplotlib.pyplot as plt
import matplotlib.colors as col
import seaborn as sns
import ffmpeg
import random
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from scipy import stats
from collections import defaultdict

from astrolink import AstroLink
from fuzzycat import FuzzyCat, FuzzyPlots

from astrolink.io import loadAstroLinkObject
from astrolink.io import saveAstroLinkObject


# Choose a particle from ['dm', 'stars', 'gas', 'stars_gas', 'stars_gas_dm'] to cluster
particleName = 'stars'

# Choose the number of snapshots to analyze from ['all', int] -> int will get last 'int' snapshots of the simulation
snapshots = 'all'
first_snapshot = 1800  # The first snapshot number to consider
last_snapshot = 2000  # The latest snapshot number in the simulation
snapshot_frequency = 1  # Frequency of snapshots to consider
total_snapshots_in_simulation = 2000 # All snapshots in the full simulation, if we only look at a fraction of snapshots

#choose significance for astrolink from ['auto', float] -> float will set a fixed significance level
significance = 'auto'

# Choice of tagging from ['dynamical' (standard), 'chemical', 'chemodynamical']
tagging = 'chemodynamical'

# should plots have labels in the movie?
plot_labels = True

# The minimum life-span of fuzzy clusters in Mega-years
minLongevityOfFuzzyClusters = 230 

# Age of the Universe in Mega-years
ageOfTheUniverse = 13800 

# Choose appropriate axis limits (in kpc) for the movie
axisLimits = 100

#plotting sample rate (every nth particle will be plotted)
plotting_sample_rate = 2

# Set up the working directory
galaxyFolderName = '2.79e12_zoom_6_rerun'
#workingDirectoryPath = f"C:/Users/power/OneDrive/_Studium/_Veranstaltungen/8. Semester/Bachelorarbeit/Backups/Programming_beautify/nihao_uhd{galaxyFolderName}_{particleName}_{snapshots}_snapshots_{tagging}_tagging_S={significance}/"
workingDirectoryPath = f"/home/samuel_data/nihao_uhd_{galaxyFolderName}_{particleName}_{snapshots}_snapshots_{tagging}_tagging_S={significance}/"
simulationDirectoryPath = f"C:/Users/power/OneDrive/_Studium/_Veranstaltungen/8. Semester/Bachelorarbeit/Backups/Programming_beautify/{galaxyFolderName}/"
simulationDirectoryPath = f"/home/_data/nihao/nihao_uhd/{galaxyFolderName}/"
snapshotFilePrefix = '2.79e12.'

snapshots = range(first_snapshot,last_snapshot + 1, snapshot_frequency)
snapshotFilePaths = [f"{simulationDirectoryPath}{snapshotFilePrefix}{i:05}" for i in snapshots]

snapshots = range(0,last_snapshot - first_snapshot + 1, snapshot_frequency)
snapshot_conversion_factor = ageOfTheUniverse / total_snapshots_in_simulation  # Convert snapshot number to Mega-years


if not os.path.exists(workingDirectoryPath):
    os.makedirs(workingDirectoryPath)
    os.makedirs(f"{workingDirectoryPath}Clusters_raw/")
    os.makedirs(f"{workingDirectoryPath}Clusters_iord/")
    os.makedirs(f"{workingDirectoryPath}Clusters/")
    os.makedirs(f"{workingDirectoryPath}Cluster_plots/")


# perform the clustering and make movie of fuzzy clusters over time
# performClustering(
#     particleName=particleName,
#     snapshots=snapshots,
#     significance=significance,
#     tagging=tagging,
#     plot_labels=plot_labels,
#     minLongevityOfFuzzyClusters=minLongevityOfFuzzyClusters,
#     ageOfTheUniverse=ageOfTheUniverse,
#     axisLimits=axisLimits,
#     workingDirectoryPath=workingDirectoryPath,
#     simulationDirectoryPath=simulationDirectoryPath,
#     snapshotFilePaths=snapshotFilePaths
# )

starClusterAnalysis(snapshotFilePaths, workingDirectoryPath, snapshots, snapshot_conversion_factor)



