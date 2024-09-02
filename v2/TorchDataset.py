from torch.utils.data import Dataset
import torch
import numpy as np

"""
Module for creating Torch dataset from Channel dataset.
"""

class TorchDataset(Dataset):
    def __init__(self, channel_dataset, sims=None, transform=None, weak_aug=None, strong_aug=None, use_self=False, use_single=False, range=1):
        """
        Initializes the TorchDataset object.

        Args:
            channel_dataset: The channel dataset.
            sims: The similarity feature vectors.
            transform: Transformation to apply to the data.
            weak_aug: Weak augmentation to apply to the data.
            strong_aug: Strong augmentation to apply to the data.
            use_self: Whether to use the same image as positive pair.
            use_single: Whether to return only one view.
            range: The offset of the positive pair.
        """
        self.channel_dataset = channel_dataset
        self.transform = transform
        self.weak_aug = weak_aug
        self.strong_aug = strong_aug
        self.use_self = use_self
        self.use_single = use_single
        self.range = range
        if sims is not None:
            self.sims = sims["sim_vectors"]
            self.sim_max = sims["max"]
            self.sim_min = sims["min"]
            self.sims_range = torch.from_numpy(self.sim_max - self.sim_min)
        else:
            self.sims = None

    def __len__(self):
        """
        Returns the total length of the dataset.

        Returns:
            int: Total length of the dataset.
        """
        # The last entry will be the last index
        # if the combined dataset was a 1d array.
        return len(self.channel_dataset)
    
    def __getitem__(self, idx):
        """
        Retrieves the item at the provided index from the dataset.

        Args:
            idx (int): Index.

        Returns:
            tuple: Transformed image, and its positive pair.
        """
        img1_, dv, timestep_idx, flight_idx = self.channel_dataset[idx]

        if self.strong_aug:
            img1 = self.strong_aug(img1_)
        if self.use_single:
            return img1

        if self.use_self:
            img2 = img1_
        else:
            # offset = np.random.randint(-self.range, self.range + 1)
            # # Positive sample is next (few)
            # next_idx = idx + offset
            # if (timestep_idx + offset >= len(dv)) or (timestep_idx + offset < 0):
            #     next_idx -= 2 * offset # Set it to previous if it's last

            # Positive sample is next
            next_idx = idx + 1
            if next_idx >= len(dv):
                next_idx -= 2 # Set it to previous if it's last
            img2, _, _, _ = self.channel_dataset[next_idx]
    
        if self.weak_aug:
            img2 = self.weak_aug(img2)
        else:
            img2 = self.strong_aug(img2)

        if self.sims is not None:
            return img1, img2, timestep_idx, flight_idx, self.sims[idx]
        else:
            return img1, img2, timestep_idx, flight_idx
