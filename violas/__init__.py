"""Public package entry point for the core Violas retrieval primitives."""

from violas.storage import (
    VectorGroup,
    VectorMap,
    VectorRef,
    VectorRelation,
    add_relation_to_description,
    create_bidirectional_relation,
    create_context_chain,
    get_relations_from_description,
)

__all__ = [
    "VectorGroup",
    "VectorMap",
    "VectorRef",
    "VectorRelation",
    "add_relation_to_description",
    "create_bidirectional_relation",
    "create_context_chain",
    "get_relations_from_description",
]

__version__ = "0.0.1"
