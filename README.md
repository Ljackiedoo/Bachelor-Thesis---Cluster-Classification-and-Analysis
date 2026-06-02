<div align="center">

# Classification and Analysis of Star-Forming Clusters in a Milky Way-Like Galaxy Simulation

**A computational pipeline for identifying and analyzing stellar structures using chemodynamical tagging.**

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)

📄 **[Read the full Bachelor Thesis (PDF)](docs/Thesis.pdf)**

</div>

---

## Overview

The formation and dissolution of stellar clusters are fundamental processes in galactic evolution. This repository contains the data pipeline and analytical code used to identify and evaluate these structures within a simulated galactic environment.

* **The Problem:** Identifying stellar structures—especially young clusters in a galactic disk—is notoriously difficult because purely kinematic signatures (positions and velocities) are erased within just a few orbital periods.
* **The Solution:** We built a data pipeline applying two general-purpose clustering algorithms ('[AstroLink](https://github.com/william-h-oliver/astrolink)' and '[FuzzyCat](https://github.com/william-h-oliver/fuzzycat)' to a high-resolution cosmological simulation ([NIHAO-UHD](https://arxiv.org/abs/1909.05864)). We compared purely dynamical tagging against a chemodynamical approach that incorporated chemical abundances (`[Fe/H]` and `[O/H]`).
* **The Result:** The chemodynamical approach proved vastly superior, identifying ~40% more clusters overall and doubling the detection rate of young, star-forming clusters in the disk. The pipeline also predicted a 4:1 ratio of native to pollution stars in a typical Milky Way-like cluster, providing a new baseline for observational studies.

---

## Visualizations

<div align="center">
  <img src="results/galaxy_animation.gif" alt="Simulation Plot" width="800"/>
  <p><i>Figure 1: Animation of clusters identified by applying AstroLink and FuzzyCat to the NIHAO-UHD 2.79e12 galaxy using chemodynamical tagging</i></p>
</div>
