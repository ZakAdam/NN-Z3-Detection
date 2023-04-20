
from torch.utils.data import Dataset
import os
import h5py
import json
import cv2
import numpy as np
import torch
from backstage import utils

import torch.nn.functional as TNF


dir_path = os.path.dirname(os.path.realpath(__file__))


class DogDataset(Dataset):
    """
    Args:
        folder (string): Directory with with h5 file.
        h5_filename (string): Name of h5 file.
        scale_shape (tuple): Shape to scale image to.
        as_tensor (bool): Whether to convert image to torch tensor.
        breed_map_path (string): Path to json file containing breed map.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(
        self,
        folder,
        h5_filename,
        detectionGridSize=None,
        imageSize=None,
        scale_shape=None,
        as_tensor=True,
        breed_map_path=dir_path + "/../../data_eda/breed_map.json",
        transform=None,
        localization=False,
        apply_random_sized_bbox_safe_crop=False,
    ):
        if os.path.exists(breed_map_path):
            with open(breed_map_path, "r") as f:
                self.breed_map = json.load(f)
                self.num_classes = len(self.breed_map)
        else:
            self.breed_map = None
            self.num_classes = 120

        self.folder = folder
        self.h5file = None
        self.filename = os.path.join(folder, h5_filename)
        self.info = {}
        self.labels = None
        self.scale_shape = scale_shape
        self.len = 0
        self.as_tensor = as_tensor
        self.transform = transform
        self.localization = localization
        self.apply_random_sized_bbox_safe_crop = apply_random_sized_bbox_safe_crop
        self.detectionGridSize = detectionGridSize
        self.imageSize = imageSize
        self.assert_open()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        self.assert_open()
        image, labels = self.load_raw(idx)
        label, bbox = labels[0], labels[1:]


        if self.apply_random_sized_bbox_safe_crop:
            image, bbox = utils.random_sized_bbox_safe_crop(image, bbox)

        # Convert to torch tensor
        if self.as_tensor:
            image = DogDataset.to_tensor(image)

        orig_image = image
        if self.transform is not None:
            image = self.transform(image)

        # Break down the label
        if not self.localization:
            label = utils.encode_fully_convolutional_label(bbox, image.shape, self.detectionGridSize)

            return image, label
        


        return image, torch.tensor(self.convert_bounding_box(orig_image, bbox))

    def convert_bounding_box(self, image, bbox):
        xmin, ymin, xmax, ymax = bbox

        _, h, w = image.shape

        if h < 0 or w < 0:
            raise ValueError("Invalid height or width of image")

        # TODO: add checks for validity

        if xmax < 0 or xmin < 0 or ymin < 0 or ymax < 0:
            raise ValueError("Invalid bounding box positions")

        center_x = (xmax + xmin) / (2.0 * w)
        center_y = (ymax + ymin) / (2.0 * h)

        bound_height = (ymax - ymin) / h
        bound_width = (xmax - xmin) / w

        dog_persence = 1.0

        return np.array([dog_persence, center_x, center_y, bound_height, bound_width], dtype=np.float)

    def assert_open(self):
        if not self.h5file:
            self.h5file = h5py.File(self.filename, mode="r")
            group_images = self.h5file["images"]
            self.len = len(group_images)
            self.labels = self.h5file.get("labels")

    def load_raw(self, idx):
        img_raw = self.h5file["images"][str(idx)]
        # because file is stored as bytes, we need to decode it
        bImage = np.fromstring(np.array(img_raw), np.uint8)
        image = cv2.imdecode(bImage, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.scale_shape:
            image = cv2.resize(image, self.scale_shape, interpolation=cv2.INTER_AREA)

        label = np.array(self.h5file["labels"][idx])
        return image, label

    @staticmethod
    def to_numpy_image(tx: torch.Tensor):
        y = tx.cpu().detach().numpy()
        y = (255 * y).astype(np.uint8)
        y = np.transpose(y, (1, 2, 0))
        return y

    @staticmethod
    def to_tensor(image: np.ndarray):
        image = image.transpose((2, 0, 1))
        image = image.astype("float32") / 255.0
        return torch.from_numpy(image)
