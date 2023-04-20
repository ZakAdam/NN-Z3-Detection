import glob
import logging
import os
import sys
import zipfile
from collections import defaultdict

import argparse as argparse
import h5py
import numpy as np
import xmltodict
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import convert_image_to_bytearray
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = [20, 10]

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True


class Annotation:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __str__(self):
        return f"{self.xmin=},{self.ymin=},{self.xmax=},{self.ymax=}"

    def __repr__(self) -> str:
        return str(self)


class Image:
    def __init__(self, img_path, annotation_path):
        self.img_path = img_path
        self.annotation_path = annotation_path
        self.img_name = os.path.basename(img_path)
        self.annotation_name = os.path.basename(annotation_path)
        self.folder_name = os.path.basename(os.path.dirname(img_path))
        self.breed_name = self.folder_name.split("-")[1]

        self.breed_id = None
        self.annotations = []
        self.np_arr = None

    def __str__(self):
        return f"{self.img_path=}, {self.annotation_path=}, " \
               f"{self.img_name=}, {self.annotation_name=}, {self.folder_name=}, " \
               f"{self.breed_name=}, {self.breed_id=}, {self.annotations=}"

    def __repr__(self) -> str:
        return str(self)


def unzip_dataset(dataset_path, unpacked_path):
    if not os.path.exists(dataset_path):
        logging.error(f'File {dataset_path} does not exist')
        sys.exit(1)
    if os.path.isdir(unpacked_path):
        logging.info(f'Unpacked dataset already exists at {unpacked_path}')
        return

    logging.info(f'Unpacking dataset {dataset_path} to {unpacked_path}')
    with zipfile.ZipFile(dataset_path) as zf:
        for member in tqdm(zf.infolist()):
            try:
                zf.extract(member, unpacked_path)
            except zipfile.error as e:
                logging.error(f'Could not extract {member.filename} from {dataset_path}. Error: {e}')


def get_images_and_annotations_paths(unpacked_path):
    """
    Creates two lists of paths to images and annotations.
    """
    image_paths = []
    annotations_paths = []
    logging.info(f'Extracting images and annotations paths from {unpacked_path}')
    for file_path in tqdm(glob.glob(unpacked_path + '/**/*', recursive=True)):
        if os.path.isfile(file_path):
            if file_path.endswith(".jpg"):
                image_paths.append(file_path)
            elif os.path.isfile(file_path):
                annotations_paths.append(file_path)
    # sort paths to have the same order
    image_paths.sort()
    annotations_paths.sort()
    assert len(image_paths) == len(annotations_paths)
    logging.info(f"Number of images: {len(image_paths)}")
    logging.info(f"Number of annotations: {len(annotations_paths)}")
    return list(zip(image_paths, annotations_paths))


def validate_paths(image_and_annotations):
    """
    Validate if image and annotation have the same name and their index in the list is the same
    """
    for image_path, annotation_path in tqdm(image_and_annotations, desc="Validating paths"):
        image_name = os.path.basename(image_path)
        annotation_name = os.path.basename(annotation_path)
        if annotation_name != image_name.replace(".jpg", ""):
            logging.error("Image and annotation names do not match!")
            logging.error(image_name)
            logging.error(annotation_name)
            sys.exit(1)


def parse_images(image_and_annotations, print_stats=False):
    def create_annotation(annotation_dict):
        return Annotation(
            int(annotation_dict['bndbox']['xmin']),
            int(annotation_dict['bndbox']['ymin']),
            int(annotation_dict['bndbox']['xmax']),
            int(annotation_dict['bndbox']['ymax'])
        )

    breed_id = 0
    breed_map = defaultdict(int)
    image_object_counts = defaultdict(int)
    breed_map_ids = {}

    images = []

    logging.info(f'Parsing images')
    for image_path, annotation_path in tqdm(image_and_annotations):
        image = Image(image_path, annotation_path)
        with open(annotation_path) as fd:
            doc = xmltodict.parse(fd.read())

        if isinstance(doc['annotation']['object'], list):
            image.annotations = [create_annotation(annotation) for annotation in doc['annotation']['object']]
        else:
            annotation_object = doc['annotation']['object']
            image.annotations.append(create_annotation(annotation_object))

        image_object_counts[len(image.annotations)] += 1

        if image.breed_name not in breed_map_ids:
            breed_map_ids[image.breed_name] = breed_id
            breed_id += 1
        image.breed_id = breed_map_ids[image.breed_name]
        breed_map[image.breed_name] += 1

        images.append(image)

    logging.info(f"Number of images: {len(images)}")
    logging.info(f"Number of breeds: {len(breed_map)}")

    if print_stats:
        logging.info(f"Generating plots.")
        # plot number of images per object count
        bar = plt.bar(image_object_counts.keys(), image_object_counts.values())
        plt.title("Number of images per object count")
        plt.xlabel("Object count")
        plt.ylabel("Number of images")
        # add labels to the bars
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, str(height), ha='center', va='bottom')

        plt.savefig("images_per_object_count.png")
        plt.clf()
        plt.close()

        total_breeds = len(breed_map)
        # plot number of images per breed
        plt.bar(list(range(total_breeds)), breed_map.values())
        plt.title("Number of images per breed")
        plt.xlabel("Breed")
        plt.ylabel("Number of images")
        plt.xticks(list(range(total_breeds)), rotation=90)

        plt.savefig("images_per_breed.png")
        plt.clf()
        plt.close()
    return images


def read_image(image_path):
    with open(image_path, 'rb') as f:
        return f.read()


def save_to_hdf5(images, output_path, convert_shape=None):
    """
    Create h5 file with:
        - Group images where each image is a dataset with name id of file and value is byte array of image
        - Dataset annotations with value is np array of breed id and bounding box
    """
    logging.info(f'Saving images to {output_path}')

    for image in images:
        assert len(image.annotations) == 1  # assert that there is only one annotation per image
        annotation = image.annotations[0]  # take first
        # create np array as breed id and bounding box
        image.np_arr = np.array([
            image.breed_id,
            annotation.xmin, annotation.ymin, annotation.xmax, annotation.ymax
        ], dtype=int)

    with h5py.File(output_path, 'w') as h5file:
        group_images = h5file.create_group('images')
        h5file.create_dataset('labels',
                              data=np.array([image.np_arr for image in images], dtype=int))
        for idx, image in tqdm(enumerate(images), total=len(images)):
            if convert_shape:
                img_bytes = convert_image_to_bytearray(image.img_path, convert_shape)
                group_images.create_dataset(str(idx), data=img_bytes)
            else:
                img_bytes = read_image(image.img_path)
                binary_blob = np.void(img_bytes)
                group_images.create_dataset(str(idx), data=binary_blob)

    logging.info("Successfully created dataset")


def plot_breed_distribution(images, img_name):
    breeds = [image.breed_id for image in images]
    # save hist plot to file
    plt.bar(list(range(len(set(breeds)))), np.bincount(breeds))
    plt.title("Breed distribution")
    plt.xlabel("Breed")
    plt.ylabel("Number of images")
    plt.xticks(list(range(len(set(breeds)))), rotation=90)
    plt.savefig(img_name)
    plt.clf()
    plt.close()


def prepare_dataset(dataset_path, unpacked_path, h5path, convert_shape=None,
                    print_stats=False, split=True, train_size=0.9):
    if type(convert_shape) == tuple and len(convert_shape) == 2 \
            and (convert_shape[0] is None or convert_shape[1] is None):
        convert_shape = None

    unzip_dataset(dataset_path, unpacked_path)
    image_and_annotations = get_images_and_annotations_paths(unpacked_path)
    validate_paths(image_and_annotations)
    images = parse_images(image_and_annotations, print_stats)

    # Filter out images with more than one object
    images_with_one_object = [image for image in images if len(image.annotations) == 1]
    logging.info(f"Number of images with one object: {len(images_with_one_object)}")

    if split:
        logging.info(f"Splitting dataset into train and test. Train size: {train_size}")
        train_images, test_images = train_test_split(images_with_one_object, train_size=train_size, random_state=42)
        logging.info(f"Train size: {len(train_images)}")
        logging.info(f"Test size: {len(test_images)}")
        plot_breed_distribution(train_images, 'train_breed_distribution.png')
        plot_breed_distribution(test_images, 'test_breed_distribution.png')

        save_to_hdf5(train_images, h5path + '_train.h5', convert_shape)
        save_to_hdf5(test_images, h5path + '_test.h5', convert_shape)
    else:
        save_to_hdf5(images_with_one_object, h5path + '.h5', convert_shape)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--dataset_path", type=str, default="archive.zip")
    argparse.add_argument("--unpacked_path", type=str, default="archive")
    argparse.add_argument("--h5path", type=str, default="dataset.h5")
    argparse.add_argument("--convert_shape_x", type=int, default=None)
    argparse.add_argument("--convert_shape_y", type=int, default=None)
    argparse.add_argument("--print_stats", action="store_true", default=True)
    argparse.add_argument("--split_train_test", action="store_true", default=True)
    argparse.add_argument("--train_size", type=float, default=0.9)

    args = argparse.parse_args()

    prepare_dataset(
        args.dataset_path,
        args.unpacked_path,
        args.h5path,
        (args.convert_shape_x, args.convert_shape_y),
        args.print_stats,
        args.split_train_test,
        args.train_size
    )
