"""
Unified vector storage primitives for Violas.

Exports VectorGroup, VectorMap, VectorRef, VectorRelation, and relation helper
functions used by the benchmark pipelines.
"""

from .vectorgroup import VectorGroup
from .vectormap import VectorMap
from .utils import (
    VectorRef,
    VectorRelation,
    add_relation_to_description,
    get_relations_from_description,
    create_bidirectional_relation,
    create_context_chain,
)

__all__ = [
    "VectorGroup",
    "VectorMap",
    "VectorRef",
    "VectorRelation",
    "add_relation_to_description",
    "get_relations_from_description",
    "create_bidirectional_relation",
    "create_context_chain",
]
