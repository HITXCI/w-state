# w-state

This repository is the source of a hybrid vehicl state estimator based on CfC and UKF. We have released the main code of our work.
<p align="center">
  <img src="https://github.com/HITXCI/w-state/blob/main/Fig-1.png" width="80%">
</p>

# Environment Setup
1. Setup the environment on Ubuntu 20.04
2. Setup the environment as follows:
```
conda env create -f environment.yaml
```
You can download the KITTI Odometry benchmark, and use the cfc-kitti.py to train and validate;
For your carsim dataset and real vehicle dataset, you can run the cfc-data.py to train. 
