"""
version0.py: K-means (Lloyd's algorithm) on a toy 2-D dataset.

What this file teaches
- K-means alternates two steps: (1) assign every point to its closest
  prototype, (2) move every prototype to the mean of the points assigned to it.
- Neither step can increase the objective J = mean_i ||x_i - mu_{c(i)}||^2, and
  there are finitely many partitions, so the algorithm always stops. We plot J
  per iteration to see the monotone decrease.

This is the first file; the algorithm itself lives in model.py.
"""
import argparse

import numpy as np
from matplotlib import pyplot as plt

from model import KMeans
from utils import make_blobs


def main():
    parser = argparse.ArgumentParser(description="K-means on 2-D blobs")
    parser.add_argument("--k", type=int, default=3, help="number of prototypes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, _ = make_blobs(seed=args.seed)

    kmeans = KMeans(args.k, seed=args.seed)
    assignments = kmeans.fit_predict(X)
    print(f"Converged after {len(kmeans.history)} iterations, objective = {kmeans.history[-1]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(X[:, 0], X[:, 1], c=assignments, s=10, cmap="viridis")
    ax1.scatter(kmeans.prototypes[:, 0], kmeans.prototypes[:, 1], c="red", marker="x", s=100, label="prototypes")
    ax1.set_title(f"K-means clusters (k = {args.k})")
    ax1.legend()
    ax2.plot(range(1, len(kmeans.history) + 1), kmeans.history, marker="o")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("objective (mean squared distance)")
    ax2.set_title("The objective never increases")
    plt.tight_layout()
    plt.savefig("kmeans_blobs.png")
    print("Saved kmeans_blobs.png")


if __name__ == "__main__":
    main()
