import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.index_utils import quantize_vectors, train_confident_tree


def main() -> None:
    resources_dir = PROJECT_ROOT / "resources"

    references_path = resources_dir / "references.json"
    vectors_path = resources_dir / "vectors.npy"
    vectors_f16_path = resources_dir / "vectors_f16.npy"
    vectors_q8_path = resources_dir / "vectors_q8.npy"
    tree_path = resources_dir / "tree.npz"
    labels_path = resources_dir / "labels.npy"
    norms_path = resources_dir / "norms.npy"

    with references_path.open() as file:
        references = json.load(file)

    vectors = np.array(
        [item["vector"] for item in references],
        dtype=np.float32,
    )
    labels = np.array(
        [1 if item["label"] == "fraud" else 0 for item in references],
        dtype=np.uint8,
    )
    norms = np.sum(vectors * vectors, axis=1, dtype=np.float32)
    vectors_f16 = vectors.astype(np.float16)
    vectors_q8 = quantize_vectors(vectors)
    tree = train_confident_tree(vectors, labels)

    np.save(vectors_path, vectors)
    np.save(vectors_f16_path, vectors_f16)
    np.save(vectors_q8_path, vectors_q8)
    np.savez_compressed(tree_path, **tree)
    np.save(labels_path, labels)
    np.save(norms_path, norms)

    print(f"saved {vectors_path} shape={vectors.shape} dtype={vectors.dtype}")
    print(f"saved {vectors_f16_path} shape={vectors_f16.shape} dtype={vectors_f16.dtype}")
    print(f"saved {vectors_q8_path} shape={vectors_q8.shape} dtype={vectors_q8.dtype}")
    print(f"saved {tree_path} nodes={tree['scores'].shape[0]}")
    print(f"saved {labels_path} shape={labels.shape} dtype={labels.dtype}")
    print(f"saved {norms_path} shape={norms.shape} dtype={norms.dtype}")


if __name__ == "__main__":
    main()
