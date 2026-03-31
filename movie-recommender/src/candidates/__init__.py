"""Public API for src.candidates."""

from src.candidates.base_candidate_generator import BaseCandidateGenerator
from src.candidates.collaborative import CFCandidateGenerator
from src.candidates.hybrid import HybridCandidateGenerator
from src.candidates.matrix_factorization import ALSCandidateGenerator
from src.candidates.popularity import PopularityCandidateGenerator

__all__ = [
    "BaseCandidateGenerator",
    "ALSCandidateGenerator",
    "CFCandidateGenerator",
    "HybridCandidateGenerator",
    "PopularityCandidateGenerator",
]
