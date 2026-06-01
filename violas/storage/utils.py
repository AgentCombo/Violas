"""Utility functions and relation helpers for vector storage."""

from typing import Any, Dict, List, Union

import numpy as np
from scipy.spatial.distance import cdist


# ============= Distance and similarity functions =============

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two vectors."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if np.isclose(norm_a, 0.0) or np.isclose(norm_b, 0.0):
        return 0.0

    return np.dot(a, b) / (norm_a * norm_b)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine distance, defined as 1 - cosine similarity."""
    return max(0.0, 1.0 - cosine_similarity(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Euclidean distance between two vectors."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    return np.linalg.norm(a - b)


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return a bounded similarity score derived from Euclidean distance."""
    distance = euclidean_distance(a, b)
    return 1.0 / (1.0 + distance)


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Manhattan distance between two vectors."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    return np.sum(np.abs(a - b))


def manhattan_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return a bounded similarity score derived from Manhattan distance."""
    distance = manhattan_distance(a, b)
    return 1.0 / (1.0 + distance)


def get_distance_function(distance_method: str):
    """
    Return a distance function by name.

    Args:
        distance_method: One of "cosine", "euclidean", or "manhattan".
    """
    if distance_method == "cosine":
        return cosine_distance
    elif distance_method == "euclidean":
        return euclidean_distance
    elif distance_method == "manhattan":
        return manhattan_distance
    else:
        return cosine_distance


# ============= Vector-set functions =============

def hausdorff_distance(
    set_a: List[np.ndarray],
    set_b: List[np.ndarray],
    metric: str = "euclidean",
) -> float:
    """Return the Hausdorff distance between two vector sets."""
    if len(set_a) == 0 or len(set_b) == 0:
        return float("inf")

    a_matrix = np.array([vec.flatten() for vec in set_a])
    b_matrix = np.array([vec.flatten() for vec in set_b])

    distances = cdist(a_matrix, b_matrix, metric=metric)
    return max(distances.min(axis=1).max(), distances.min(axis=0).max())


def hausdorff_similarity(
    set_a: List[np.ndarray],
    set_b: List[np.ndarray],
    metric: str = "euclidean",
) -> float:
    """Return a bounded similarity score derived from Hausdorff distance."""
    distance = hausdorff_distance(set_a, set_b, metric)
    return 1.0 / (1.0 + distance)


def max_similarity(
    set_a: List[np.ndarray],
    set_b: List[np.ndarray],
    similarity_func=cosine_similarity,
) -> float:
    """Return the maximum pairwise similarity between two vector sets."""
    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0

    max_sim = 0.0

    for vec_a in set_a:
        for vec_b in set_b:
            sim = similarity_func(vec_a, vec_b)
            max_sim = max(max_sim, sim)

    return max_sim


# ============= Description normalization =============

def normalize_description(description: Union[Dict[str, Any], str, None] = None) -> Dict[str, Any]:
    """
    Normalize a vector description into a dictionary.

    Args:
        description: A dictionary, string, None, or any other object.

    Returns:
        A dictionary suitable for storage in a VectorGroup.
    """
    if description is None:
        return {}
    elif isinstance(description, dict):
        return description
    elif isinstance(description, str):
        return {"text": description}
    else:
        return {"default": description}


def normalize_descriptions(descriptions: Union[List[Dict[str, Any]], List[str], List[Any]]) -> List[Dict[str, Any]]:
    """Normalize every item in a description list."""
    return [normalize_description(desc) for desc in descriptions]


# ============= Relation data structures =============

class VectorRef:
    """Reference a concrete vector stored in a VectorMap."""

    def __init__(self, key: str, group_name: str, vector_idx: int):
        self.key = key
        self.group_name = group_name
        self.vector_idx = vector_idx

    def to_string(self) -> str:
        """Serialize the reference as key%group_name%vector_idx."""
        return f"{self.key}%{self.group_name}%{self.vector_idx}"

    @staticmethod
    def from_string(ref_str: str) -> "VectorRef":
        """Parse a key%group_name%vector_idx reference string."""
        parts = ref_str.split("%")
        if len(parts) != 3:
            raise ValueError(f"Invalid ref string format: {ref_str}")
        return VectorRef(parts[0], parts[1], int(parts[2]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "group_name": self.group_name,
            "vector_idx": self.vector_idx,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "VectorRef":
        return VectorRef(d["key"], d["group_name"], d["vector_idx"])

    def __eq__(self, other):
        if not isinstance(other, VectorRef):
            return False
        return (
            self.key == other.key
            and self.group_name == other.group_name
            and self.vector_idx == other.vector_idx
        )

    def __hash__(self):
        return hash((self.key, self.group_name, self.vector_idx))

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return f"VectorRef({self.key}, {self.group_name}, {self.vector_idx})"


class VectorRelation:
    """
    Describe a relation from one vector to another vector.

    Common relation types include context_prev, context_next, image_caption,
    caption_image, similar, same_entity, qa_pair, parent, and child.
    """

    CONTEXT_PREV = "context_prev"
    CONTEXT_NEXT = "context_next"
    IMAGE_CAPTION = "image_caption"
    CAPTION_IMAGE = "caption_image"
    SIMILAR = "similar"
    SAME_ENTITY = "same_entity"
    QA_PAIR = "qa_pair"
    PARENT = "parent"
    CHILD = "child"

    def __init__(
        self,
        ref: Union[VectorRef, str, Dict],
        relation_type: str,
        weight: float = 1.0,
        metadata: Dict[str, Any] = None,
    ):
        """
        Args:
            ref: Related vector reference, as VectorRef, string, or dictionary.
            relation_type: Semantic relation type.
            weight: Relation weight clamped to [0.0, 1.0].
            metadata: Optional relation metadata.
        """
        if isinstance(ref, VectorRef):
            self.ref = ref
        elif isinstance(ref, str):
            self.ref = VectorRef.from_string(ref)
        elif isinstance(ref, dict):
            self.ref = VectorRef.from_dict(ref)
        else:
            raise ValueError(f"Invalid ref type: {type(ref)}")

        self.relation_type = relation_type
        self.weight = max(0.0, min(1.0, weight))
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the relation to a dictionary for storage in descriptions."""
        result = {
            "ref": self.ref.to_string(),
            "relation_type": self.relation_type,
            "weight": self.weight,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "VectorRelation":
        """Create a relation from its dictionary representation."""
        return VectorRelation(
            ref=d["ref"],
            relation_type=d["relation_type"],
            weight=d.get("weight", 1.0),
            metadata=d.get("metadata"),
        )

    def __str__(self):
        return f"VectorRelation({self.ref} <- {self.relation_type}, w={self.weight})"

    def __repr__(self):
        return self.__str__()


# ============= Relation helper functions =============

def add_relation_to_description(
    description: Dict[str, Any],
    relation: VectorRelation,
) -> Dict[str, Any]:
    """Append a relation to a vector description."""
    if "related_vectors" not in description:
        description["related_vectors"] = []

    description["related_vectors"].append(relation.to_dict())
    return description


def get_relations_from_description(
    description: Dict[str, Any],
    relation_type: str = None,
) -> List[VectorRelation]:
    """Read vector relations from a description, optionally filtered by type."""
    related = description.get("related_vectors", [])
    relations = [VectorRelation.from_dict(r) for r in related]

    if relation_type:
        relations = [r for r in relations if r.relation_type == relation_type]

    return relations


def create_bidirectional_relation(
    desc1: Dict[str, Any],
    ref1: VectorRef,
    desc2: Dict[str, Any],
    ref2: VectorRef,
    relation_type_1_to_2: str,
    relation_type_2_to_1: str,
    weight: float = 1.0,
) -> None:
    """Create two directed relations between a pair of vector descriptions."""
    add_relation_to_description(desc1, VectorRelation(ref2, relation_type_1_to_2, weight))
    add_relation_to_description(desc2, VectorRelation(ref1, relation_type_2_to_1, weight))


def create_context_chain(
    descriptions: List[Dict[str, Any]],
    refs: List[VectorRef],
    weight_decay: float = 0.8,
    max_distance: int = 2,
) -> None:
    """
    Create local context relations for an ordered vector sequence.

    Each vector is linked to preceding and following vectors within
    max_distance. Relation weights decay with distance.
    """
    n = len(descriptions)
    assert len(refs) == n, "descriptions and refs must have the same length"

    for i in range(n):
        for dist in range(1, max_distance + 1):
            if i - dist >= 0:
                weight = weight_decay ** dist
                add_relation_to_description(
                    descriptions[i],
                    VectorRelation(refs[i - dist], VectorRelation.CONTEXT_PREV, weight),
                )

        for dist in range(1, max_distance + 1):
            if i + dist < n:
                weight = weight_decay ** dist
                add_relation_to_description(
                    descriptions[i],
                    VectorRelation(refs[i + dist], VectorRelation.CONTEXT_NEXT, weight),
                )
