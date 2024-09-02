import numpy as np
from tqdm import tqdm

"""
Module for filtering pixel data.

This module defines the `PixelsFilterV2` class, which implements methods 
to filter out streaks from pixel data based on the method described in the report.
"""


class PixelsFilterV2:
    """
    Class to filter pixel data.

    Attributes:
        pixels (numpy.ndarray): Array of pixel data.
        img_lens (numpy.ndarray): Array of image lengths.
        filtered_pixels (list): List to store filtered pixel data.
        streak_filtered_count (int): Count of filtered streaks.
        img_bound_lower (numpy.ndarray): Lower bounds of filtered single-particle image indices.
        img_bound_upper (numpy.ndarray): Upper bounds of filtered single-particle image indices.
    """

    def __init__(self, pixels, img_lens):
        """
        Initializes the PixelsFilterV2 object.

        Args:
            pixels (numpy.ndarray): Array of pixel data.
            img_lens (numpy.ndarray): Array of image lengths.
        """
        self.pixels = pixels
        self.img_lens = img_lens
        self.filtered_pixels = []

        self.streak_filtered_count = 0

        # Image bounds (img start : image end)
        image_inds = np.zeros(len(self.img_lens) + 1, dtype=np.int64)
        image_inds[1:] = self.img_lens
        image_inds = np.cumsum(image_inds)
        self.img_bound_lower = image_inds[0:-1]
        self.img_bound_upper = image_inds[1:]

    def _get_image_ranges_batch(self, start_indices, end_indices, query_ranges):
        """
        Given a range of corresponding start, and end indices pairs for *columns* in the
        pixel array -- this method returns the index of the first and last single-particle
        image in each.

        Args:
            start_indices (numpy.ndarray): Array of start indices.
            end_indices (numpy.ndarray): Array of end indices.
            query_ranges (numpy.ndarray): Query ranges.

        Returns:
            numpy.ndarray: Image ranges.
        """
        start_search = np.searchsorted(start_indices, query_ranges[:, 0], side="right")
        end_search = np.searchsorted(end_indices, query_ranges[:, 1], side="left")

        # Disclude the last index from end, it's not end-inclusive
        # return np.array(list(zip(start_search, end_search - 1)))

        # This is faster; start needs to begin at 0
        return np.c_[start_search - 1, end_search]

    def _divisions_to_ranges(self, arr):
        # [0 30 60 90] -> [[0, 30], [30, 60], [60, 90]] **
        return np.column_stack((arr[:-1], arr[1:]))

    def find_first_last_1_for_rows(self, im):
        """
        Find first and last row with a 1 in the image, used for filtering.
        We assume the image is not transposed.

        Args:
            im (numpy.ndarray): Image data.

        Returns:
            numpy.ndarray: First and last 1 indices.
        """
        first_one_indices = np.argmax(im == 1, axis=1)
        last_one_indices = im.shape[1] - 1 - np.argmax(im[:, ::-1] == 1, axis=1)

        result = np.column_stack((first_one_indices, last_one_indices))

        return result

    def x_same_and_width_below_129(self, ranges, maj):
        """
        Check if there are 'maj' count of equal ranges in the provided ranges,
        and ensure the width is below 129 (which means a 1 is not in the range).

        Args:
            ranges (numpy.ndarray): Ranges.
            maj (int): Majority count.

        Returns:
            bool: True if conditions met, False otherwise.
        """
        unique_ranges, counts = np.unique(ranges, axis=0, return_counts=True)
        return (
            np.any(counts >= maj)
            and 0 <= np.abs(unique_ranges[0, 1] - unique_ranges[0, 0]) < 129
        )

    # Iterate over 32 width from image
    # Keep a list of img_indices and reverse check to filter.
    def filter(self, width=32, checks=15, p_match=0.75):
        """
        Filter streaks out of pixel data.

        Args:
            width (int): Width of the filter. Default is 32.
            checks (int): Number of checks. Default is 15.
            p_match (float): Percentage of checks that must match exactly to filter. Default is 0.75.

        Returns:
            tuple: Filtered pixel data and filtered images in slices.
        """
        checkpoints = np.linspace(width // checks, width - width // checks, checks)
        checkpoints = np.round(checkpoints).astype(int)

        # The total number of slices in the pixels array
        num_slices = len(self.pixels) // width
        # The indices marking  the start and end of each slice
        division_indices = np.arange(0, (num_slices + 1) * width, width)
        end_indices = np.roll(division_indices, -1)

        query_ranges = self._divisions_to_ranges(division_indices)

        # Images in each 32 slice
        images_in_slices = self._get_image_ranges_batch(
            self.img_bound_lower, self.img_bound_upper, query_ranges
        )
        # Stores the indices that should be filtered out
        self.should_filter = np.zeros((num_slices,))

        for idx in tqdm(range(num_slices - 1)):
            start_col = division_indices[idx]
            end_col = end_indices[idx]
            slice = (self.pixels[start_col:end_col] - 1) / 255

            first_last_1s = self.find_first_last_1_for_rows(slice)
            mask = first_last_1s[checkpoints]
            self.should_filter[idx] = self.x_same_and_width_below_129(
                mask, int(checks * p_match)
            )

        self.streak_filtered_count = np.count_nonzero(self.should_filter)
        remaining_indices = np.where(self.should_filter == 0)[0]

        filtered_lower = division_indices[remaining_indices]
        filtered_upper = end_indices[remaining_indices]
        pre_concat_pixels = [
            self.pixels[low:up]
            for low, up in list(zip(filtered_lower, filtered_upper))[:-1]
        ]
        # We could filter everything
        filtered_pixels = (
            np.concatenate(pre_concat_pixels)
            if len(pre_concat_pixels) > 0
            else np.array([])
        )
        filtered_images_in_slices = images_in_slices[remaining_indices]
        return filtered_pixels, filtered_images_in_slices
