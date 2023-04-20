import cv2
import h5py
import numpy as np
import argparse


def read_and_save_image_from_h5(h5path, image_id, result_name):
    with h5py.File(h5path, 'r') as h5file:
        img_raw = h5file['images'][str(image_id)]
        nparr = np.array(img_raw)
        data = cv2.imdecode(nparr, cv2.IMWRITE_JPEG_QUALITY)
        cv2.imwrite(result_name, data)


def read_and_save_raw_image_from_h5(h5path, image_id, result_name):
    with h5py.File(h5path, 'r') as h5file:
        img_raw = h5file['images'][str(image_id)]
        nparr = np.array(img_raw)
        with open(result_name, 'wb') as f:
            f.write(nparr)


def convert_image_to_bytearray(file_path, convert_shape=None):
    """
    Converts image to byte array, 1d array of pixels
    """
    src = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if convert_shape and len(convert_shape) == 2 and convert_shape[0] and convert_shape[1]:
        img = cv2.resize(src, convert_shape, interpolation=cv2.INTER_AREA)
    else:
        img = src
    _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return img_encoded


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--image_id", type=int)
    argparse.add_argument("--h5path", type=str, default="dataset.h5")
    argparse.add_argument("--output", type=str, default="result.jpg")
    argparse.add_argument("--raw", action="store_true", default=True)
    args = argparse.parse_args()
    if args.raw:
        read_and_save_raw_image_from_h5(args.h5path, args.image_id, args.output)
    else:
        read_and_save_image_from_h5(args.h5path, args.image_id, args.output)
