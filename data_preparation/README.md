# Description

From a zip file creates hdf5 file with group `images` with N datasets of dog images converted to bytearray (1d vector).
Dataset names are indexed from 0 to N-1 and correspond to the rows in `annotations` dataset.

Annotations dataset has structure like this:

```
breed_id, xmin, ymin, xmax, ymax
```

Last 4 columns are the coordinates of the bounding box.

# Example

```bash
python data_preparation/prepare.py --dataset_path="data/archive.zip" --unpacked_path="data/archive" --h5path="data/dataset.h5"
```
If you want to run those commands on our machine, use /data instead of data, to allowe them to be used globally.
```bash
python data_preparation/prepare.py --dataset_path="/data/archive.zip" --unpacked_path="/data/archive" --h5path="/data/dataset.h5"
```

# Usage

```
prepare.py [-h]
    [--dataset_path DATASET_PATH]
    [--unpacked_path UNPACKED_PATH]
    [--h5path H5PATH] 
    [--convert_shape_x CONVERT_SHAPE_X] 
    [--convert_shape_y CONVERT_SHAPE_Y] 
    [--print_stats]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path        DATASET_PATH - path to ZIP File
  --unpacked_path       UNPACKED_PATH - where to unpack the ZIP File
  --h5path              H5PATH - where to save the HDF5 File
  --convert_shape_x     CONVERT_SHAPE_X - optional
  --convert_shape_y     CONVERT_SHAPE_Y - optional
  --print_stats         Print statistics about the breeds

```
