"""Vector group data structure."""

import hashlib
from typing import Any, Dict, List

import numpy as np

try:
    from .utils import normalize_description, normalize_descriptions
except ImportError:
    from utils import normalize_description, normalize_descriptions


class VectorGroup:
    def __init__(
        self,
        group_name: str,
        representative: np.ndarray,
        rep_description: str,
        vectors: List[np.ndarray],
        descriptions: List[Dict[str, Any]],
        vector_type: str = "general",
        group_type: str = "default",
    ):
        """
        Store a set of related vectors under a shared representative.

        Args:
            group_name: Human-readable group name.
            representative: Representative vector for this group.
            rep_description: Description of the representative vector.
            vectors: Member vectors in the group.
            descriptions: Metadata for each member vector.
            vector_type: Vector modality such as "image", "text", or "general".
            group_type: Structural role such as "topic", "content", or "question".
        """
        self.group_name = group_name
        self.representative = representative
        self.rep_description = rep_description
        self.vectors = vectors
        self.descriptions = normalize_descriptions(descriptions)
        self.vector_type = vector_type
        self.group_type = group_type

        assert len(vectors) == len(self.descriptions), "vectors and descriptions must have the same length"
        assert len(vectors) > 0, "vectors must not be empty"
        assert representative.shape == vectors[0].shape, "representative and member vectors must have the same shape"

        self.group_id = self._generate_group_id()

    def append(self, vector: np.ndarray, description: Dict[str, Any] = None) -> int:
        """
        Append one vector to the group.

        Returns:
            The inserted vector index, or -1 when the vector dimension is invalid.
        """
        if vector.shape != self.vectors[0].shape:
            print(f"warning: vector dimension mismatch: {vector.shape} != {self.vectors[0].shape}", flush=True)
            return -1

        self.vectors.append(vector)
        self.descriptions.append(normalize_description(description))
        return len(self.vectors) - 1

    def _generate_group_id(self) -> str:
        content = f"{self.group_name}_{self.vector_type}_{self.group_type}"
        return hashlib.md5(content.encode()).hexdigest()

    def size(self) -> int:
        return len(self.vectors)

    def is_empty(self) -> bool:
        return len(self.vectors) == 0

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "group_name": self.group_name,
            "group_id": self.group_id,
            "vector_type": self.vector_type,
            "group_type": self.group_type,
            "vector_count": len(self.vectors),
            "vector_dimension": self.vectors[0].shape[0] if self.vectors else 0,
            "rep_description": self.rep_description,
        }


if __name__ == "__main__":
    print("VectorGroup smoke test", flush=True)

    np.random.seed(42)
    test_vectors = [np.random.rand(128) for _ in range(10)]
    test_descriptions = [{"text": f"vector_{i + 1}"} for i in range(10)]
    representative = np.mean(test_vectors, axis=0)

    group1 = VectorGroup(
        group_name="test_group",
        representative=representative,
        rep_description="test group representative",
        vectors=test_vectors,
        descriptions=test_descriptions,
    )

    group1.append(np.random.rand(128), {"text": "extra test vector"})

    print(f"group: {group1}")
    print(f"statistics: {group1.get_statistics()}")
    print("VectorGroup smoke test completed", flush=True)
