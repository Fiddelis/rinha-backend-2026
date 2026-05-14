from pathlib import Path
import os
import sys

import numpy as np
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

N_CLUSTERS_DEFAULT = 1024
BATCH_SIZE_DEFAULT = 8192
RANDOM_STATE_DEFAULT = 42
SAMPLE_SIZE_DEFAULT = 400_000
INIT_ITERATIONS_DEFAULT = 100
REFINE_PASSES_DEFAULT = 2


def assign_centroids(
    vectors: np.ndarray,
    centroids: np.ndarray,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> np.ndarray:
    assignments = np.empty(vectors.shape[0], dtype=np.int32)
    centroid_norms = np.sum(centroids * centroids, axis=1, dtype=np.float32)

    for start in range(0, vectors.shape[0], batch_size):
        end = min(start + batch_size, vectors.shape[0])
        batch = vectors[start:end].astype(np.float32, copy=False)
        batch_norms = np.sum(batch * batch, axis=1, dtype=np.float32)
        distances = batch_norms[:, None] + centroid_norms[None, :] - 2.0 * (
            batch @ centroids.T
        )
        assignments[start:end] = np.argmin(distances, axis=1)

    return assignments


def refine_centroids(
    vectors: np.ndarray,
    centroids: np.ndarray,
    passes: int = REFINE_PASSES_DEFAULT,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    assignments = np.empty(vectors.shape[0], dtype=np.int32)

    for _ in range(passes):
        assignments = assign_centroids(vectors, centroids, batch_size=batch_size)
        sums = np.zeros_like(centroids, dtype=np.float64)
        counts = np.bincount(assignments, minlength=centroids.shape[0])

        for start in range(0, vectors.shape[0], batch_size):
            end = min(start + batch_size, vectors.shape[0])
            batch = vectors[start:end].astype(np.float32, copy=False)
            batch_assignments = assignments[start:end]
            np.add.at(sums, batch_assignments, batch)

        populated = counts > 0
        centroids[populated] = (
            sums[populated] / counts[populated, None]
        ).astype(np.float32)

    return centroids, assignments


def main() -> None:
    resources_dir = PROJECT_ROOT / "resources"
    n_clusters = int(os.getenv("RINHA_IVF_CLUSTERS", str(N_CLUSTERS_DEFAULT)))
    batch_size = int(os.getenv("RINHA_IVF_BATCH_SIZE", str(BATCH_SIZE_DEFAULT)))
    random_state = int(os.getenv("RINHA_IVF_SEED", str(RANDOM_STATE_DEFAULT)))
    sample_size_limit = int(
        os.getenv("RINHA_IVF_SAMPLE_SIZE", str(SAMPLE_SIZE_DEFAULT))
    )
    init_iterations = int(
        os.getenv("RINHA_IVF_INIT_ITERATIONS", str(INIT_ITERATIONS_DEFAULT))
    )
    refine_passes = int(
        os.getenv("RINHA_IVF_REFINE_PASSES", str(REFINE_PASSES_DEFAULT))
    )

    vectors_f32 = np.load(resources_dir / "vectors.npy", mmap_mode="r")
    vectors_f16 = np.load(resources_dir / "vectors_f16.npy", mmap_mode="r")
    vectors_q8 = np.load(resources_dir / "vectors_q8.npy", mmap_mode="r")
    labels = np.load(resources_dir / "labels.npy", mmap_mode="r")
    norms = np.load(resources_dir / "norms.npy", mmap_mode="r")

    sample_size = min(vectors_f32.shape[0], sample_size_limit)
    rng = np.random.default_rng(random_state)
    sample_ids = rng.choice(vectors_f32.shape[0], size=sample_size, replace=False)
    sample = np.asarray(vectors_f32[sample_ids], dtype=np.float32)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=init_iterations,
        algorithm="lloyd",
    )
    kmeans.fit(sample)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    centroids, cluster_ids = refine_centroids(
        vectors_f32,
        centroids,
        passes=refine_passes,
        batch_size=batch_size,
    )

    cluster_indices = np.argsort(cluster_ids, kind="stable").astype(np.int32)
    counts = np.bincount(cluster_ids, minlength=n_clusters)
    cluster_offsets = np.zeros(n_clusters + 1, dtype=np.int64)
    cluster_offsets[1:] = np.cumsum(counts, dtype=np.int64)

    reordered_f32 = np.asarray(vectors_f32[cluster_indices], dtype=np.float32)
    reordered_q8 = np.asarray(vectors_q8[cluster_indices], dtype=np.uint8)
    reordered_f16 = np.asarray(vectors_f16[cluster_indices], dtype=np.float16)
    reordered_labels = np.asarray(labels[cluster_indices], dtype=np.uint8)
    reordered_norms = np.asarray(norms[cluster_indices], dtype=np.float32)
    contiguous_indices = np.arange(cluster_indices.shape[0], dtype=np.int32)

    np.save(resources_dir / "centroids.npy", centroids)
    np.save(resources_dir / "cluster_indices.npy", contiguous_indices)
    np.save(resources_dir / "cluster_offsets.npy", cluster_offsets)
    np.save(resources_dir / "vectors.npy", reordered_f32)
    np.save(resources_dir / "vectors_q8.npy", reordered_q8)
    np.save(resources_dir / "vectors_f16.npy", reordered_f16)
    np.save(resources_dir / "labels.npy", reordered_labels)
    np.save(resources_dir / "norms.npy", reordered_norms)

    print(
        f"saved {resources_dir / 'centroids.npy'} "
        f"shape={centroids.shape} dtype={centroids.dtype}"
    )
    print(
        "ivf params "
        f"seed={random_state} sample_size={sample_size} "
        f"clusters={n_clusters} max_iter={init_iterations} refine_passes={refine_passes}"
    )
    print(
        f"saved {resources_dir / 'cluster_indices.npy'} "
        f"shape={contiguous_indices.shape} dtype={contiguous_indices.dtype}"
    )
    print(
        f"saved {resources_dir / 'cluster_offsets.npy'} "
        f"shape={cluster_offsets.shape} dtype={cluster_offsets.dtype}"
    )
    print(
        f"reordered {resources_dir / 'vectors.npy'} "
        f"shape={reordered_f32.shape} dtype={reordered_f32.dtype}"
    )
    print(
        f"reordered {resources_dir / 'vectors_q8.npy'} "
        f"shape={reordered_q8.shape} dtype={reordered_q8.dtype}"
    )
    print(
        f"reordered {resources_dir / 'vectors_f16.npy'} "
        f"shape={reordered_f16.shape} dtype={reordered_f16.dtype}"
    )
    print(
        f"reordered {resources_dir / 'labels.npy'} "
        f"shape={reordered_labels.shape} dtype={reordered_labels.dtype}"
    )
    print(
        f"reordered {resources_dir / 'norms.npy'} "
        f"shape={reordered_norms.shape} dtype={reordered_norms.dtype}"
    )


if __name__ == "__main__":
    main()
