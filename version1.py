"""
version1.py: colour quantization ("cartoonify") of an image with K-means.

What is NEW compared to version0.py
- The same algorithm on real data: every pixel is a point in RGB space (d = 3),
  so a 512x512 image is a dataset of 262,144 points.
- After fitting, every pixel is replaced by its closest prototype, so the image
  is redrawn with only k colours. Small k gives a flat, poster/cartoon look.
- Scale matters: model.py computes distances with ||x||^2 - 2x.c + ||c||^2
  instead of broadcasting an (n, k, 3) array.

Usage: python version1.py --k 5 [--image image.jpg]
"""
import argparse

from model import KMeans
from utils import get_image, show_image, save_image


def main():
    parser = argparse.ArgumentParser(description="Colour quantization with K-means")
    parser.add_argument("--image", default="image.jpg", help="input image (jpg or png)")
    parser.add_argument("--k", type=int, default=50, help="number of colours (prototypes)")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="also display the result")
    args = parser.parse_args()

    # get image
    image = get_image(args.image)
    img_shape = image.shape

    # reshape image: (H, W, 3) -> (H*W, 3), one row per pixel
    image = image.reshape(img_shape[0] * img_shape[1], img_shape[2])

    # create and fit the model
    kmeans = KMeans(args.k, seed=args.seed)
    kmeans.fit(image, max_iter=args.max_iter)
    print(f"k = {args.k}: {len(kmeans.history)} iterations, objective = {kmeans.history[-1]:.5f}")

    # replace each pixel with its closest prototype, and reshape back to an image
    image = kmeans.replace_with_prototypes(image).reshape(img_shape)

    out_path = f"image_cartoonified_{args.k}.jpg"
    save_image(image, out_path)
    print(f"Saved {out_path}")
    if args.show:
        show_image(image)


if __name__ == "__main__":
    main()
