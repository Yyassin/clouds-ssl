import bisect

"""
Module to aggregate datasets composed of multiple data views,
from different flights/files.
"""


class ChannelDataset:
    """
    Attributes:
        dvs (list): List of data views.
        dv_bounds (list): List of bounds for data views.
    """

    def __init__(self, dataviews):
        """
        Initializes the ChannelDataset object.

        Args:
            dataviews (list): List of data views.
        """
        self.dvs = dataviews
        self.dv_bounds = []
        self._set_bounds()

    def __len__(self):
        """
        Returns the total length of the dataset.

        Returns:
            int: Total length of the dataset.
        """
        # The last entry will be the last index
        # if the combined dataset was a 1d array.
        return self.dv_bounds[-1] + 1

    def _set_bounds(self):
        """
        Sets the bounds for data views.

        Given a 2D array, returns an array used for 1d indexing. For example,
        the array [[1, 2], [3, 4, 5], [1, 7, 9]] returns [-1, 1, 4, 7]. Now, if we
        access an index between [0, 1] inclusive this'll map to index 0. [2, 4]
        inclusive maps to 1, greater than 4 maps to 2.

        So to acces the last 9 in the 2d array, we want index 7. That maps to
        2 first -- so we access the third array. To get the index of the 9,
        we subtract the prior bound so (7 - 4) - 1 = 2.
        """

        # Start the count with -1 to offset everything left by 1. Reason is
        # if we have [-1, a, b] for two bins (one of size a, second of size b)
        # then index `a` will fall into bin 0 when we do binary search -- if we
        # used starting indices [-1, a + 1] then index `a+1` will fall into bin 0
        # incorrectly, since it belongs to the second bin.
        bounds = [-1]
        for dv in self.dvs:
            bounds.append(bounds[-1] + len(dv))
        self.dv_bounds = bounds

    def _find_bin(self, arr, number):
        """
        Finds the dataview index for a given "global" index
        across the entire channel dataset. So [1, 4, 7] and 3
        would return 1 since 3 is between 1 and 4. The fourth
        image is in the second dataview, at index 3 - 1 = 2.

        Args:
            arr (list): List of numbers.
            number (int): Number to find.

        Returns:
            int: Bin index.
        """
        # Use bisect to find the insertion point for the number in the array
        # This insertion point corresponds to the first element greater than the number
        # and serves as the bin index
        bin_index = bisect.bisect_left(arr, number)
        # It'll give us 1-index bins, so sub 1.
        return bin_index - 1

    def __getitem__(self, idx):
        """
        Retrieves an item at "global" index `idx` from the dataset.

        Args:
            idx (int): Index.

        Returns:
            tuple: The clip image and the data view it belongs in.
        """
        dv_idx = self._find_bin(self.dv_bounds, idx)
        dv = self.dvs[dv_idx]

        im_idx = idx - self.dv_bounds[dv_idx] - 1
        return dv[im_idx], dv, im_idx, dv_idx
