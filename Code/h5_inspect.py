import h5py
import numpy as np
from matplotlib import pyplot as plt

def h5file_inspect(file_path, preview_lim=5):
    def print_info(name, obj):
        obj_type = "Group" if isinstance(obj, h5py.Group) else "Dataset"
        print(f"{name} ({obj_type})")

        # display attributes
        for attr, val in obj.attrs.items():
            print(f"  ├─ Attr: {attr} = {val}")

        if isinstance(obj, h5py.Dataset):
            print(f"  ├─ Shape: {obj.shape}")
            print(f"  ├─ Dtype: {obj.dtype}")

            # preview data if it's small enough
            try:
                data = obj[()]
                if data.size < preview_lim:
                    print(f"  └─ Data: {data}")
                else:
                    preview = data.flat[:preview_lim]
                    print(f"  └─ Preview: {preview} ...")
            except Exception as e:
                print(f"  └─ (Could not read data: {e})")

    with h5py.File(file_path, "r") as f:
        f.visititems(print_info)

def show_coils(data, slice_nums, cmap=None):
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i+1)
        plt.imshow(data[num], cmap=cmap)
    plt.show(block=True)

h5file_inspect("file1000167.h5")

'''
The ismrmrd_header contains information like this:

| Section                        | Description                                                         |
| ------------------------------ | ------------------------------------------------------------------- |
| `encoding`                     | Matrix size, FOV, parallel imaging parameters                       |
| `acquisitionSystemInformation` | Number of receiver coils, system details, etc.                      |
| `subjectInformation`           | Patient age, weight, gender which have been anonymised              |
| `experimentalConditions`       | Field strength of the MRI machine (e.g. 1.5T, 3T)                   |
| `sequenceParameters`           | TR, TE, flip angle, etc.                                            |
and more...
'''

file = h5py.File("file1000167.h5", "r")
xml_str = file['ismrmrd_header'][()].decode('utf-8')
print(xml_str) # this shows a lot of information use xml_str[:1000] to show just a snippet

# show one of the k-spaces of which this data is for
volume_kspace = file['kspace'][()]
slice_kspace = volume_kspace[20] # choosing the 20th slices of this volume
show_coils(np.log(np.abs(slice_kspace) + 1e-9), [0]) # this shows the k-spaces of coils 0
