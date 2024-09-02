# Cloud Particles: A Carleton University and NRC collaboration
This repository contains the final implementation and training script.

Below is a _brief_ explanation of the repository.

### Main directory:

single_gpu_train_v2.py → Trains vision transformer models using the MoCo V3 self-supervised learning framework on a single GPU.

training.py → Implements some helper functions needed for training (right now, just learning rate scheduler).

dataview.ipynb → Jupyter notebook that demonstrates creating the 384x384px images from NetCDF files.

### models directory:

mocov3.py → Implementation of the MoCo V3 framework.

vision_transformer.py → Implementation of the vision transformer network.

### v2 directory (data preprocessing and loading):

PixelsFilterV2.py → Accepts the frame buffer from a NetCDF and filters it to remove streaks.

DataView.py → Constructs 384x384px image "clips" from a frame buffer, and provides an interface to index them.

ChannelDataset.py → Aggregates dataviews across multiple flight files, for a specific sensor channel, into a single dataset.

TorchDataset.py → A torch dataset wrapper for the channel dataset. Generates positive pairs, and transforms images to be ready for training.
