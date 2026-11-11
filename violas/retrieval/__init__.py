"""Intent-aware retrieval over the public VectorMap storage classes.

This package does not import benchmark drivers or database client libraries.
"""
from .index import IntentIndex
from .scoring import Corpus, representative_scan
from .hdmg import AdaptiveHDMG
from .objective import select_exact, objective as evaluate_objective
from .directory import build_semantic_directory
from .incremental import IncrementalVGStore

__all__ = ['IntentIndex', 'Corpus', 'AdaptiveHDMG', 'select_exact',
           'evaluate_objective', 'build_semantic_directory',
           'IncrementalVGStore', 'representative_scan']
