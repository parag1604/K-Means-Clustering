"""
version2.py: better initialization (k-means++) and choosing k (the elbow plot).

What is NEW compared to version1.py
- K-means only finds a local minimum, and which one depends on the start.
  k-means++ seeds the prototypes far apart (each new seed is sampled with
  probability proportional to its squared distance from the seeds chosen so
  far). We compare the final objective of random vs k-means++ starts over
  several seeds.
- The objective always decreases as k grows, so it cannot pick k by itself.
  Plotting it against k and looking for the "elbow", where extra colours stop
  paying off, is a simple heuristic.

To keep this fast, K-means is fitted on a random subsample of the pixels.
"""
import argparse

import numpy as np
from matplotlib import pyplot as plt

from model import KMeans
from utils import get_image


def main():
    parser = argparse.ArgumentParser(description="k-means++ vs random init, and the elbow plot")
    parser.add_argument("--image", default="image.jpg")
    parser.add_argument("--k", type=int, default=10, help="k used for the initialization comparison")
    parser.add_argument("--seeds", type=int, default=5, help="number of restarts per initialization")
    parser.add_argument("--subsample", type=int, default=20_000, help="pixels used for fitting")
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 5, 8, 10, 15, 20, 30, 50])
    args = parser.parse_args()

    image = get_image(args.image)
    pixels = image.reshape(-1, image.shape[-1])
    rng = np.random.default_rng(0)
    X = pixels[rng.choice(len(pixels), min(args.subsample, len(pixels)), replace=False)]

    # 1. Initialization: same k, several seeds
    results = {}
    for init in ["random", "kmeans++"]:
        objs = []
        for seed in range(args.seeds):
            km = KMeans(args.k, init=init, seed=seed)
            km.fit(X)
            objs.append(km.history[-1])
        results[init] = objs
        print(f"{init:>9}: final objective mean = {np.mean(objs):.5f}, "
              f"worst = {np.max(objs):.5f}, iterations (last run) = {len(km.history)}")

    # 2. Elbow plot: objective vs k (k-means++ init)
    objectives = []
    for k in args.ks:
        km = KMeans(k, init="kmeans++", seed=0)
        km.fit(X)
        objectives.append(km.history[-1])
        print(f"k = {k:>3}: objective = {objectives[-1]:.5f}")

    plt.figure(figsize=(5, 4))
    plt.plot(args.ks, objectives, marker="o")
    plt.xlabel("k (number of colours)")
    plt.ylabel("objective (mean squared distance)")
    plt.title("Elbow plot")
    plt.tight_layout()
    plt.savefig("elbow.png")
    print("Saved elbow.png")


if __name__ == "__main__":
    main()
