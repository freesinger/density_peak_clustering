# Density Peak Clustering 
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![check](https://img.shields.io/github/status/s/pulls/freesinger/Density_Peak_Cluster/3.svg?style=flat)
![top_language](https://img.shields.io/github/languages/top/freesinger/Density_Peak_Cluster.svg?colorB=blue&logo=top_language&style=flat)
![release](https://img.shields.io/github/release/freesinger/Density_Peak_Cluster.svg?colorB=orange&style=flat)
![size](https://img.shields.io/github/repo-size/freesinger/Density_Peak_Cluster.svg?colorB=red&style=flat)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Ffreesinger%2Fdensity_peak_clustering.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Ffreesinger%2Fdensity_peak_clustering?ref=badge_shield)

## 1. Dependencies

**Python:**  `3.6.7`

**Lib:**  `numpy, matplotlib`

## 2. Files

- **data:** Given data from this essay
- **images:** Generated images for constructing framework
- **references:** Essays used for this projects
- **report:** Records and report
- **data_process.py:** Data processing
- **cluster.py:** Find cluster center and classify points
- **setup.py:** Process given data and visualizing
- **generatePoints.py:** Generating testing points dataset
- **plot.py:** Process generated data and visualizing

## 3. Usage

- Show the performance on given dataset:

`>_ python3.6 setup.py`

- Show the performance on generate dataset:

`>_ python3.6 generatePoints.py`

`>_ python3.6 plot.py`

## 4. Performances

- Dataset to be clustering:

![generatedPoints](images/generatedPoints.png)

- After clustering:

![result](images/result.png)


## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Ffreesinger%2Fdensity_peak_clustering.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Ffreesinger%2Fdensity_peak_clustering?ref=badge_large)
