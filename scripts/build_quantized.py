from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.index_utils import quantize_vectors, train_confident_tree


def main() -> None:
    resources_dir = PROJECT_ROOT / "resources"

    vectors = np.load(resources_dir / "vectors.npy", mmap_mode="r")
    labels = np.load(resources_dir / "labels.npy", mmap_mode="r")

    vectors_f16 = vectors.astype(np.float16)
    vectors_q8 = quantize_vectors(vectors)
    tree = train_confident_tree(
        np.asarray(vectors, dtype=np.float32),
        np.asarray(labels, dtype=np.uint8),
    )

    np.save(resources_dir / "vectors_f16.npy", vectors_f16)
    np.save(resources_dir / "vectors_q8.npy", vectors_q8)
    np.savez_compressed(resources_dir / "tree.npz", **tree)

    print(
        f"saved {resources_dir / 'vectors_f16.npy'} "
        f"shape={vectors_f16.shape} dtype={vectors_f16.dtype}"
    )
    print(
        f"saved {resources_dir / 'vectors_q8.npy'} "
        f"shape={vectors_q8.shape} dtype={vectors_q8.dtype}"
    )
    print(
        f"saved {resources_dir / 'tree.npz'} "
        f"nodes={tree['scores'].shape[0]}"
    )


if __name__ == "__main__":
    main()
