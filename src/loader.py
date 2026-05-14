from pathlib import Path

import numpy as np


class ResourceLoader:
    def __init__(self) -> None:
        self.resources_dir = Path(__file__).resolve().parent.parent / "resources"

    def _resource_path(self, name: str) -> Path:
        return self.resources_dir / name

    def _load_optional_array(
        self, name: str, mmap_mode: str = "r"
    ) -> np.ndarray | None:
        path = self._resource_path(name)
        if not path.exists():
            return None
        return np.load(path, mmap_mode=mmap_mode)

    def load_vectors(self) -> np.ndarray:
        return np.load(self._resource_path("vectors.npy"), mmap_mode="r")

    def load_vectors_f16(self) -> np.ndarray | None:
        return self._load_optional_array("vectors_f16.npy")

    def load_vectors_q8(self) -> np.ndarray | None:
        return self._load_optional_array("vectors_q8.npy")

    def load_labels(self) -> np.ndarray:
        return np.load(self._resource_path("labels.npy"), mmap_mode="r")

    def load_norms(self) -> np.ndarray:
        return np.load(self._resource_path("norms.npy"), mmap_mode="r")

    def load_centroids(self) -> np.ndarray:
        return np.load(self._resource_path("centroids.npy"), mmap_mode="r")

    def load_cluster_indices(self) -> np.ndarray:
        return np.load(self._resource_path("cluster_indices.npy"), mmap_mode="r")

    def load_cluster_offsets(self) -> np.ndarray:
        return np.load(self._resource_path("cluster_offsets.npy"), mmap_mode="r")

    def load_tree(self) -> dict[str, np.ndarray] | None:
        path = self._resource_path("tree.npz")
        if not path.exists():
            return None
        tree_file = np.load(path)
        return {key: tree_file[key] for key in tree_file.files}
