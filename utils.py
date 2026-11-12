import numpy as np
from matplotlib import pyplot as plt


def get_image(image_path):
    image = plt.imread(image_path)
    # JPEGs load as uint8 in [0, 255]; PNGs already load as floats in [0, 1]
    if image.dtype == np.uint8:
        image = image / 255.0
    return image[..., :3]  # drop the alpha channel if there is one


def show_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def save_image(image, image_path):
    plt.imsave(image_path, np.clip(image, 0.0, 1.0))


def make_blobs(num_points: int = 600, seed: int = 0):
    """Three 2-D Gaussian blobs: a toy dataset where K-means works well."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [4.0, 1.0], [1.5, 4.0]])
    labels = rng.integers(len(centers), size=num_points)
    X = centers[labels] + rng.normal(scale=0.8, size=(num_points, 2))
    return X, labels
