import numpy as np

"""
Module for and constructing 384x384 images from pixel data.
"""

IM_HEIGHT = 128


class DataView:
    """
    Class for constructing 384x384 images from pixel data.

    Attributes:
        stacks (int): Number of rows to stack to create an image.
        im_size (int): Size of the image.
        pixels (numpy.ndarray): Array of pixel data.
        timestamps (numpy.ndarray): Array of timestamps. UNUSED.
        filtered_images_in_slices (numpy.ndarray): Array of filtered images in slices.
        slices_size (int): Size of slices.
        images_in_view (list): List to store single-particle image range in a given clip/image from the dataview.
        images_in_view_granular (list): List to store each single-particle image in a given clip/image from the dataview.
        img_to_square_lut (numpy.ndarray): Lookup table that goes from single-particle image to clip index (which 384x384 has this particle).
    """

    def __init__(
        self,
        im_size,
        pixels,
        filtered_images_in_slices=None,
        slices_size=32,
        timestamps=None,
    ):
        """
        Initializes the DataView object.

        Args:
            im_size (int): Size of the image.
            pixels (numpy.ndarray): Array of pixel data.
            timestamps (numpy.ndarray): Array of timestamps.
            filtered_images_in_slices (numpy.ndarray): Array of filtered images in slices. Default is None.
            slices_size (int): Size of slices. Default is 32.
        """
        assert im_size % IM_HEIGHT == 0, "im_size must be a multiple of 128"

        # Number of stacked rows
        self.stacks = im_size // IM_HEIGHT
        # Square (384)
        self.im_size = im_size
        self.pixels = pixels  # (width, 128)
        self.timestamps = timestamps

        # Each image from the view has shape
        # (128 * stacks, 128 * stacks). This is `stacks`
        # rows of length 128 * stacks. Each row is
        # 128 * stacks columns, so we need 128 * stacks * stacks
        # columns per image.
        self.cols_per_im = IM_HEIGHT * self.stacks * self.stacks
        self.filtered_images_in_slices = filtered_images_in_slices
        self.slices_size = slices_size
        self.images_in_view = []
        self.images_in_view_granular = []

        self.img_to_square_lut = []

    def _find_ranges(self, value, ranges):
        """
        Finds indices in provided ranges that contain the specified value.

        Args:
            value (numpy.ndarray): The value to query.
            ranges (numpy.ndarray): Ranges.

        Returns:
            numpy.ndarray: Index of range containing value.
        """
        valid_ranges = np.where((value >= ranges[:, 0]) & (value <= ranges[:, 1]))[0]
        return valid_ranges

    def pixel_image_idx_to_square_indices(self, idx):
        """
        Converts single-particle image index to square image / clip index.

        Args:
            idx (int): Single-particle image index.

        Returns:
            numpy.ndarray: Clip index.
        """
        return self._find_ranges(idx, self.img_to_square_lut)

    def fill_lut(self):
        """
        Fills the lookup table (LUT) with ranges of single-particle images
        in each clip / image.
        """
        size = len(self)
        self.img_to_square_lut = np.zeros((size, 2))
        for idx in range(size):
            self.img_to_square_lut[idx] = self._get_image_indices(idx)

    def _get_image_indices(self, idx):
        """
        Gets the indices of the single-particle images within the clip
        at the provided index.

        Args:
            idx (int): Clip Index.

        Returns:
            numpy.ndarray: Image indices.
        """
        slices_per_image = self.stacks * self.stacks * (IM_HEIGHT // self.slices_size)
        slice_ptr = idx * slices_per_image

        start_imgs = self.filtered_images_in_slices[slice_ptr]
        end_imgs = self.filtered_images_in_slices[slice_ptr + slices_per_image]
        self.images_in_view = [start_imgs[0], end_imgs[-1]]
        self.images_in_view_granular = []
        for i in range(slices_per_image):
            self.images_in_view_granular.append(
                self.filtered_images_in_slices[slice_ptr + i]
            )

        return np.array([start_imgs[0], end_imgs[-1]]).astype(int)

    def compute_slice_times(self):
        """
        Computes the average timestamp for each clip image.
        """
        self.img_to_avg_time = np.zeros((len(self),))
        slices_per_image = self.stacks * self.stacks * (IM_HEIGHT // self.slices_size)
        for slice_ptr in range(len(self)):
            start_imgs = self.filtered_images_in_slices[slice_ptr]
            end_imgs = self.filtered_images_in_slices[slice_ptr + slices_per_image]
            start_time, end_time = (
                self.timestamps[start_imgs[0]],
                self.timestamps[end_imgs[-1]],
            )
            self.img_to_avg_time[slice_ptr] = (start_time + end_time) / 2

    def __len__(self):
        """
        Returns the number of images in the view.
        """
        return self.pixels.shape[0] // self.cols_per_im

    def __getitem__(self, idx):
        """
        Constructs and retrieves the 384x384 clip image at the provided index.

        Args:
            idx (int): Index.

        Returns:
            numpy.ndarray: Resulting image.
        """
        result_image = np.zeros((self.im_size, self.im_size))
        img_ptr = idx * self.cols_per_im

        # Loop to extract the slices
        for i in range(self.stacks):
            start_ptr = img_ptr + i * self.stacks * IM_HEIGHT
            end_ptr = img_ptr + (i + 1) * self.stacks * IM_HEIGHT
            im_slice = np.abs(self.pixels[start_ptr:end_ptr] - 1)
            result_image[i * IM_HEIGHT : i * IM_HEIGHT + IM_HEIGHT, :] = im_slice.T

        if self.filtered_images_in_slices is not None:
            self._get_image_indices(idx)

        return result_image
