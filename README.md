# tangle-network-particle-filter
By [Or Tslil](https://github.com/ortslil64), [Tal Feiner](https://github.com/TalFeiner) and Avishy Carmi
This package is part of the work titled "Distributed Information Fusion in Tangle Networks", currently under review.

## Algorithm
The package contains a fusion scheme for distributed particle filters and a demo for sensor network. The fusion scheme is based on the generalization of the Covariance Intersection algorithm. The fusion algorithm is fed with a network transition matrix that contains the connectivity of the network (see `demo.py`).

## Demo
The demo implemented here is a sensor network for object positioning using range and bearing observations. The target stochastic model is of a sircular walk, and the estimated state is the target center of mass and orientation. An example for the target trajectory and a position of the sensor is shown in the next figure.

![demo](https://github.com/ortslil64/tangle-network-particle-filter/blob/master/images/nodes.png?raw=true "Sensors positions and target trajectory")

For this demo we use a random network transition matrix as follows.

![demo](https://github.com/ortslil64/tangle-network-particle-filter/blob/master/images/transition_matrix.png?raw=true "visualization of the transition matrix")

## Installation
1) Clone the repository:
```bash
git clone https://github.com/ortslil64/tangle-network-particle-filter.git
```

2) Install the dependencies in `requirments.txt` file

3) Install the package using pip:
```bash
pip install -e tangle-network-particle-filter
```

## Usage
Run the demo file:
```bash
python3 demo/demo.py  
```

## API tutorial
TO DO
