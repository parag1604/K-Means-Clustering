import numpy as np


def squared_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance between every row of X (n, d) and every row of C (k, d).

    Uses ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, which needs an (n, k) matrix only.
    The naive X[:, None] - C builds an (n, k, d) array: for a 512x512 image and
    k = 50 that is ~300 MB, which is why the expanded form is used here.
    """
    d2 = (X ** 2).sum(axis=1, keepdims=True) - 2 * X @ C.T + (C ** 2).sum(axis=1)
    return np.maximum(d2, 0.0)  # clip tiny negatives caused by round-off


class KMeans:
    def __init__(self, k: int, eps: float = 1e-6, init: str = "random", seed: int = None) -> None:
        self.num_prototypes = k
        self.prototypes = None
        self.eps = eps            # stop when the prototypes move less than this
        self.init = init          # "random" (Forgy) or "kmeans++"
        self.rng = np.random.default_rng(seed)
        self.history = []         # objective (mean squared distance) after every iteration

    def _init_random(self, X: np.ndarray) -> np.ndarray:
        # Forgy initialization: k distinct data points chosen uniformly at random
        return X[self.rng.choice(X.shape[0], self.num_prototypes, replace=False)].copy()

    def _init_kmeans_pp(self, X: np.ndarray) -> np.ndarray:
        # k-means++ (Arthur & Vassilvitskii, 2007): the first prototype is a random
        # point; every next one is sampled with probability proportional to the
        # squared distance to the closest prototype chosen so far. Prototypes
        # therefore start spread out over the data.
        prototypes = [X[self.rng.integers(X.shape[0])]]
        closest = squared_distances(X, np.array(prototypes))[:, 0]
        for _ in range(1, self.num_prototypes):
            probs = closest / closest.sum()
            prototypes.append(X[self.rng.choice(X.shape[0], p=probs)])
            closest = np.minimum(closest, squared_distances(X, prototypes[-1][None])[:, 0])
        return np.array(prototypes)

    def objective(self, X: np.ndarray) -> float:
        # the K-means objective (distortion), averaged over the samples
        return squared_distances(X, self.prototypes).min(axis=1).mean()

    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        X = X.astype(np.float64)
        # Initialization matters: with a bad start some prototypes may end up with
        # no assigned pixels at all (an "empty cluster"), and its mean is undefined.
        self.prototypes = self._init_kmeans_pp(X) if self.init == "kmeans++" else self._init_random(X)
        self.history = []

        for _ in range(max_iter):
            # Assignment step: each sample goes to its closest prototype
            assignments = self.predict(X)

            # Update step: each prototype moves to the mean of its samples
            new_prototypes = self.prototypes.copy()
            worst_first = None
            for i in range(self.num_prototypes):
                members = X[assignments == i]
                if len(members) > 0:
                    new_prototypes[i] = members.mean(axis=0)
                else:
                    # empty cluster: re-seed it at the samples that are currently worst fitted
                    if worst_first is None:
                        worst_first = list(np.argsort(-squared_distances(X, self.prototypes).min(axis=1)))
                    new_prototypes[i] = X[worst_first.pop(0)]

            shift = np.abs(new_prototypes - self.prototypes).max()
            self.prototypes = new_prototypes
            self.history.append(self.objective(X))
            if shift < self.eps:  # converged: assignments can no longer change
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmin(squared_distances(X, self.prototypes), axis=1)

    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)

    def replace_with_prototypes(self, X: np.ndarray) -> np.ndarray:
        return self.prototypes[self.predict(X)]
