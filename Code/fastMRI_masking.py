# importing necessary packages
import h5py
import numpy as np
import fastmri
from matplotlib import pyplot as plt

from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc
from fastMRI.fastmri.data.subsample import EquiSpacedMaskFunc
from fastmri.data.subsample import EquispacedMaskFractionFunc
from fastmri.data.subsample import MagicMaskFunc
from fastmri.data.subsample import MagicMaskFractionFunc

# Shows what the absolute value of a k-space looks like:
def show_coils(data, slice_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i+1)
        plt.imshow(data[num], cmap=cmap)
    plt.show(block=True)

# The fastMRI dataset is distributed as a set of HDF5 files and can be read with the h5py package.
# Here, we show how to open a file from the multi-coil dataset. Each file corresponds to one MRI
# scan and contains the k-space data, ground truth and some metadata related to the scan.

file_name = 'file1000167.h5' # file must be from the training set
hf = h5py.File(file_name)

# In multi-coil MRIs, k-space has the following shape: (number of slices, number of coils, height, width)
# For single coil MRIs, k-space has the following shape: (number of slices, height, width)

# MRIs are acquired as 3D volumes, the first dimension is the number of 2D slices.

volume_kspace = hf['kspace'][()]

slice_kspace = volume_kspace[20] # choosing the 20th slices of this volume
print('Shape of the k-spaces:', slice_kspace.shape)

show_coils(np.log(np.abs(slice_kspace) + 1e-9), [0, 5, 10]) # This shows the k-spaces of coils 0, 5 and 10

# The fastMRI repo contains some utility functions to convert k-space into image space.
# These functions work on PyTorch Tensors. The to_tensor function can covert NumPy arrays
# to PyTorch Tensors.

slice_kspace2 = T.to_tensor(slice_kspace) # convert from numpy array to pytorch tensor
print('PyTorch shape of the k-spaces:', slice_kspace2.shape)
slice_image = fastmri.ifft2c(slice_kspace2) # apply inverse fourier transform to get the complex image
slice_image_abs = fastmri.complex_abs(slice_image) # compute absolute value to get real image

show_coils(slice_image_abs, [0, 5, 10], cmap='gray')

# Results will show that each coil in a multi-coil MRI scan focuses on a different region of the image.
# These coils can be combined into the full image using the Root-Sum-of-Squares (RSS) transform.

slice_image_rss = fastmri.rss(slice_image_abs, dim=0)

plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray')
plt.show(block=True)

'''
We can simulate under-sampled data by creating a mask and applying it to k-space.
Here we show a few different types of masking on a k-space and then their images
after applying the inverse Fourier transform.

1. create the mask object
2. next apply the mask to the kspace using apply_mask
3. show the masked kspace of coils 0,5 and 10 from slice 20
4. apply the inverse fast fourier transform to get the complex image
5. compute absolute value to get the real image
6. display the subsampled image
7. combine coils into the full image using rss
8. show the full image
'''

# Randomly generated mask
random_mask = RandomMaskFunc(center_fractions=[0.04], accelerations=[8])
random_mask_kspace, mask, _ = T.apply_mask(slice_kspace2, random_mask)
show_coils(np.log(np.abs(T.tensor_to_complex_np(random_mask_kspace)) + 1e-9), [0, 5, 10])

random_mask_slice_image = fastmri.ifft2c(random_mask_kspace)
random_mask_slice_image_abs = fastmri.complex_abs(random_mask_slice_image)
show_coils(random_mask_slice_image_abs, [0, 5, 10], cmap='gray')

random_mask_image_rss = fastmri.rss(random_mask_slice_image_abs, dim=0)
plt.imshow(np.abs(random_mask_image_rss.numpy()), cmap='gray')
plt.show(block=True)

# Equispaced mask
equispaced_mask = EquiSpacedMaskFunc(center_fractions=[0.04], accelerations=[8])
equispaced_mask_kspace, mask, _ = T.apply_mask(slice_kspace2, equispaced_mask)
show_coils(np.log(np.abs(T.tensor_to_complex_np(equispaced_mask_kspace)) + 1e-9), [0, 5, 10])

equispaced_mask_slice_image = fastmri.ifft2c(equispaced_mask_kspace)
equispaced_mask_slice_image_abs = fastmri.complex_abs(equispaced_mask_slice_image)
show_coils(equispaced_mask_slice_image_abs, [0, 5, 10], cmap='gray')

equispaced_mask_image_rss = fastmri.rss(equispaced_mask_slice_image_abs, dim=0)
plt.imshow(np.abs(equispaced_mask_image_rss.numpy()), cmap='gray')
plt.show(block=True)

# Equispaced mask fraction
equispaced_mask_fraction = EquispacedMaskFractionFunc(center_fractions=[0.04], accelerations=[8])
equispaced_mask_fraction_kspace, mask, _ = T.apply_mask(slice_kspace2, equispaced_mask_fraction)
show_coils(np.log(np.abs(T.tensor_to_complex_np(equispaced_mask_fraction_kspace)) + 1e-9), [0, 5, 10])

equispaced_mask_fraction_slice_image = fastmri.ifft2c(equispaced_mask_fraction_kspace)
equispaced_mask_fraction_slice_image_abs = fastmri.complex_abs(equispaced_mask_fraction_slice_image)
show_coils(equispaced_mask_fraction_slice_image_abs, [0, 5, 10], cmap='gray')

equispaced_mask_fraction_image_rss = fastmri.rss(equispaced_mask_fraction_slice_image_abs, dim=0)
plt.imshow(np.abs(equispaced_mask_fraction_image_rss.numpy()), cmap='gray')
plt.show(block=True)

# Magic mask
magic_mask = MagicMaskFunc(center_fractions=[0.04], accelerations=[8])
magic_mask_kspace, mask, _ = T.apply_mask(slice_kspace2, magic_mask)
show_coils(np.log(np.abs(T.tensor_to_complex_np(magic_mask_kspace)) + 1e-9), [0, 5, 10])

magic_mask_slice_image = fastmri.ifft2c(magic_mask_kspace)
magic_mask_slice_image_abs = fastmri.complex_abs(magic_mask_slice_image)
show_coils(magic_mask_slice_image_abs, [0, 5, 10], cmap='gray')

magic_mask_image_rss = fastmri.rss(magic_mask_slice_image_abs, dim=0)
plt.imshow(np.abs(magic_mask_image_rss.numpy()), cmap='gray')
plt.show(block=True)

# Magic mask fraction
magic_mask_fraction = MagicMaskFractionFunc(center_fractions=[0.04], accelerations=[8])
magic_mask_fraction_kspace, mask, _ = T.apply_mask(slice_kspace2, magic_mask_fraction)
show_coils(np.log(np.abs(T.tensor_to_complex_np(magic_mask_fraction_kspace)) + 1e-9), [0, 5, 10])

magic_mask_fraction_slice_image = fastmri.ifft2c(magic_mask_fraction_kspace)
magic_mask_fraction_slice_image_abs = fastmri.complex_abs(magic_mask_fraction_slice_image)
show_coils(magic_mask_fraction_slice_image_abs, [0, 5, 10], cmap='gray')

magic_mask_fraction_image_rss = fastmri.rss(magic_mask_fraction_slice_image_abs, dim=0)
plt.imshow(np.abs(magic_mask_fraction_image_rss.numpy()), cmap='gray')
plt.show(block=True)