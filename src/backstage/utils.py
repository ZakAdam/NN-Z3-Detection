import cv2
import numpy as np
import matplotlib.pyplot as plt

import backstage.dataset as dataset
import torch
import random



def showImage(x, h=5):
    plt.figure(figsize=(h * 16.0 / 9.0, h))
    plt.axis("off")
    plt.imshow(x)
    plt.show()
    plt.clf()
    plt.close()


def get_image_with_bounding_box(
    image=None,
    xmin=None,
    ymin=None,
    xmax=None,
    ymax=None,
    h5_index=None,
    h5_file=None,
    image_path=None,
):
    """
    Given an image, or an index into an h5 file, or a path to an image,
    show the image with bounding box's coordinates.

    Example image path with bounding box:
        get_image_with_bounding_box(image_path=path, xmin=0, ymin=0, xmax=100, ymax=100)

    Example h5 index in a h5 file. This expects file structure as follows:
        h5 file with group "images" and group "labels" where each row corresponds to id in group "images".
        Labels is a numpy array with 5 columns class, xmin, ymin, xmax, ymax
        get_image_with_bounding_box(h5_index=0, h5_file="data/train.h5")

    Example image with bounding box:
        image is read from h5 file which was saved as raw bytes, so fromstring is used to convert to numpy array,
        then cv2.imdecode is used to convert to image.
        image, labels = train[0] # needs to implement getitem in dataset
        category, xmin, ymin, xmax, ymax = labels
        get_image_with_bounding_box(image, xmin, ymin, xmax, ymax)

    Args:
        image: raw bytes of image to bound (optional)
        xmin: minimum x coordinate of bounding box (optional)
        ymin: minimum y coordinate of bounding box (optional)
        xmax: maximum x coordinate of bounding box (optional)
        ymax: maximum y coordinate of bounding box (optional)
        h5_index: index into h5 file (optional)
        h5_file: path to h5 file (optional)
        image_path: path to image (optional)

    Returns:
        Image with bounding box visible
    """
    if h5_index and h5_file:
        img_raw = h5_file["images"][str(h5_index)]
        cat, xmin, ymin, xmax, ymax = np.array(h5_file["labels"][h5_index])
    elif image_path:
        with open(image_path, "rb") as f:
            img_raw = f.read()

    if image is None:
        bImage = np.fromstring(np.array(img_raw), np.uint8)
        image = cv2.imdecode(bImage, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    BOX_COLOR = (255, 255, 0)  # Yellow
    thickness = 2

    return cv2.rectangle(
        image, (xmin, ymin), (xmax, ymax), color=BOX_COLOR, thickness=thickness
    )

def draw_image_with_bbox(image, bbox, color=(255, 255, 0), thickness=2):
    # TODO: add checks for validity
    h, w, _ = image.shape

    if isinstance(bbox, torch.Tensor):
        bbox = bbox.cpu().detach().numpy()

    if h < 0 or w < 0:
        raise ValueError("Invalid height or width of image")

    center_x, center_y, bound_height, bound_width = bbox
    left = (center_x - bound_width / 2.0) * w
    right = left + (bound_width * w)

    top = (center_y - bound_height / 2.0) * h
    bottom = top + (bound_height * h)

    return cv2.rectangle(
        image, (int(left), int(top)), (int(right), int(bottom)), color, thickness
    )

def draw_bbox_from_tensor(image_as_tensor, bbox_tensor):
    image = dataset.DogDataset.to_numpy_image(image_as_tensor)
    return draw_image_with_bbox(image, bbox_tensor)

    

def draw_bbox_numpy_image(image_as_numpy, bbox):
    return draw_image_with_bbox(image_as_numpy, bbox)


def draw_detection_result(image_as_tensor, detection_results):
    image = dataset.DogDataset.to_numpy_image(image_as_tensor)
    
    for detection_result in detection_results:
        image = draw_bbox_numpy_image(image, detection_result[1:5])
    return image


def encode_fully_convolutional_label(orig_label, image_size, grid_size):
    grid_height, grid_width = grid_size

    result = np.zeros((grid_height, grid_width, 5), dtype=float)

    for i in range(grid_height):
        for j in range(grid_width):
            result[i, j] = encode_fully_convolutional_label_cell(
                i, j, orig_label, image_size, grid_size
            )
    result = np.transpose(result, (2, 0, 1))

    return torch.tensor(result)


def encode_fully_convolutional_label_cell(i, j, orig_label, image_size, grid_size):
    xmin, ymin, xmax, ymax = orig_label
    _, image_height, image_width = image_size
    grid_height, grid_width = grid_size

    center_x = (xmax + xmin) / (2.0 * image_width)
    center_y = (ymax + ymin) / (2.0 * image_height)


    bound_height = (ymax - ymin) / image_height
    bound_width = (xmax - xmin) / image_width


    grid_relative_height = 1 / grid_height
    grid_relative_width = 1 / grid_width

    cell_bottom = (i + 1) * grid_relative_height
    cell_top = i * grid_relative_height
    cell_left = j * grid_relative_width
    cell_right = (j + 1) * grid_relative_width


    if cell_left < center_x <= cell_right and cell_top < center_y <= cell_bottom:
        cell_relative_center_x = (center_x - cell_left) / grid_relative_width
        cell_relative_center_y = (center_y - cell_top) / grid_relative_height
        cell_relative_width = bound_width / grid_relative_width
        cell_relative_height = bound_height / grid_relative_height

        return np.array(
            [
                1.0,
                cell_relative_center_x,
                cell_relative_center_y,
                cell_relative_height,
                cell_relative_width,
            ],
            dtype=np.float,
        )
    else:
        return np.array([0] * 5, dtype=float)


def extract_bounding_boxes(detection_result, threshold=0.5):
    result = []
    detection_result_rows = detection_result.size(1)
    detection_result_cols = detection_result.size(2)
    grid_relative_height = 1 / detection_result_rows
    grid_relative_width = 1 / detection_result_cols

    for i in range(detection_result_rows):
        for j in range(detection_result_cols):
            detection = detection_result[:, i,j]
            if detection[0] >= threshold:
                cell_top = i * grid_relative_height
                cell_left = j * grid_relative_width
                cx, cy = detection[1], detection[2]

                center_x = cx * grid_relative_width + cell_left
                center_y = cy * grid_relative_height + cell_top
                bound_height = detection[3] * grid_relative_height
                bound_width = detection[4] * grid_relative_width

                result.append((detection[0],center_x, center_y, bound_height, bound_width))
    return result


def random_sized_bbox_safe_crop(numpy_image, bbox, width=224, height=224):
    xmin, ymin, xmax, ymax = bbox

    h, w, _ = numpy_image.shape


    bound_height = (ymax - ymin)
    bound_width = (xmax - xmin)


    max_dim = min(w,h)
    min_dim = max(bound_width, bound_height)
    if max_dim < min_dim:
        max_dim = min_dim


    dim = random.randint(min_dim, max_dim+1)
    L = random.randint(0, w - dim) if dim < w else 0
    R = L + dim
    T = random.randint(0, h - dim) if dim < h else 0
    B = T + dim

    # move bbox to new location
    xmin = np.clip(xmin - L, 0, dim)
    xmax = np.clip(xmax - L, 0, dim)
    ymin = np.clip(ymin - T, 0, dim)
    ymax = np.clip(ymax - T, 0, dim)
    bbox = [xmin, ymin, xmax, ymax]

    return numpy_image[T:B, L:R, :], bbox