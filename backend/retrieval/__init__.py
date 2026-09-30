"""
Retrieval package for Diagnos AI.
Provides BM25 retrieval over Samsung Galaxy Settings deeplink catalogs.
"""

from .bm25_retriever import BM25Retriever, get_retriever, tokenize
from .label_match import match_by_label

__all__ = ["BM25Retriever", "get_retriever", "tokenize", "match_by_label"]
