# Clouds-SSL

Official PyTorch codebase for _Soft Contrastive Representation Learning for Cloud-particle Images Captured In-Flight by the New HVPS-4 Airborne Probe_.

Available on arXiv: Link available soon

## Usage

This repository contains code to train and evaluate all the models described in the paper. We have also released pretrained weights for the best performing hyperparameters under each model, alongside pre-processed training and annotated data.

### Installation

To get started using the repository, install the dependencies specified in the `requirements.txt` file (some sort of virtual environment is recommended):
```bash
$ python -m venv ./venv
$ ./venv/Scripts/activate
$ pip install requirements.txt
```

Then install the data, and pretrained models using git-lfs (make sure you have git-lfs installed):
```bash
$ sudo apt install git-lfs -y
$ git lfs install
$ git lfs pull
```
### Code Structure

```
.
├── annotated_images                    # contains images used for evaluation
├── annotations.csv                     # contains the labels for select images from the folder above
├── eval                                # saved evaluation results (embeddings, and probing MAEs)
├── filtered_data                       # contains pre-processed training data
├── saved_models                        # contains pre-trained weights for each of the 4 models in the paper
├── v2                                  # dataloading helpers. Only TorchDataset is really relevant.
├── eval_util.py, probe.py, embed.py    # evaluation helpers
└── single_gpu_train_*.py               # training scripts for moco (v2), soft-moco (soft), and remoco (jepa).
```

### Training

To start training a model, run the according `single_gpu_train_*.py` script:

```bash
$ python single_gpu_train_v2.py
```

which will iterate over, and train a model for each member of the specified cartesian product of hyperparameters. Models will be saved in `saved_models`.

### Evaluation

To evaluate a model, we first extract its embeddings/representations for each image in evaluation set using the `embed.py` script. These embeddings are saved in the `eval/embeddings` directory.

We then run the `probe.py` script, pointing to these embeddings to train a KNN regressor on the embeddings. The class-wise, and average MAEs are saved under `eval/probe`.

Running `embed.py`, followed by `probe.py` will generate the probing results for each model saved under `saved_models`.


## Please Cite

```
@inproceedings{yassin2024clouds,
  title={Soft Contrastive Representation Learning for Cloud-particle Images Captured In-Flight by the New HVPS-4 Airborne Probe},
  author={Yassin, Yousef and Fuller, Anthony and Ranjbar, Keyvan and Bala, Kenny and Nichman, Leonid and Green, James R},
  year={2024}
}
```