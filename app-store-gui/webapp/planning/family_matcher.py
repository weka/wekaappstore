from __future__ import annotations

import re
from dataclasses import replace
from typing import Dict, Iterable, List

from .models import SupportedFamilyMatch, SupportedFamilyMetadata


_DEFAULT_REQUIRED_DOMAINS = [
    "cpu",
    "memory",
    "gpu",
    "namespaces",
    "storage_classes",
    "weka",
]

_SUPPORTED_FAMILY_CATALOG: Dict[str, SupportedFamilyMetadata] = {
    "ai-agent-enterprise-research": SupportedFamilyMetadata(
        family="ai-agent-enterprise-research",
        display_name="AI Agent Enterprise Research",
        description="Research assistant stack with vector storage and vLLM-backed inference services.",
        keywords=[
            "ai agent",
            "enterprise research",
            "research assistant",
            "domain specific data",
            "vector db",
            "vector database",
            "vllm chat model",
            "vllm embed model",
        ],
        required_domains=list(_DEFAULT_REQUIRED_DOMAINS),
    ),
    "nvidia-vss": SupportedFamilyMetadata(
        family="nvidia-vss",
        display_name="NVIDIA VSS",
        description="GPU-backed NVIDIA VSS deployment with vLLM-backed chat and embedding services.",
        keywords=[
            "nvidia vss",
            "vss",
            "visual search",
            "video search",
            "vision search",
            "nvidia",
            "vllm chat model",
            "vllm embed model",
        ],
        required_domains=list(_DEFAULT_REQUIRED_DOMAINS),
    ),
    "openfold": SupportedFamilyMetadata(
        family="openfold",
        display_name="OpenFold Protein Prediction",
        description="Protein folding workflow backed by WEKA CSI filesystem and Argo workflows.",
        keywords=[
            "openfold",
            "protein",
            "protein folding",
            "protein prediction",
            "alphafold",
            "msa",
            "argo workflow",
            "filesystem",
            "weka cluster filesystem",
        ],
        required_domains=list(_DEFAULT_REQUIRED_DOMAINS),
    ),
}

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def supported_family_catalog() -> Dict[str, SupportedFamilyMetadata]:
    return {family: replace(metadata) for family, metadata in _SUPPORTED_FAMILY_CATALOG.items()}


class SupportedFamilyMatcher:
    def __init__(self, *, catalog: Dict[str, SupportedFamilyMetadata] | None = None) -> None:
        self._catalog = catalog or supported_family_catalog()

    def match(self, request_text: str) -> SupportedFamilyMatch:
        normalized_text = _normalize_text(request_text)
        if not normalized_text:
            return SupportedFamilyMatch(
                status="no_supported_family",
                reason="No request text was provided for supported-family matching.",
            )

        scores: List[tuple[int, SupportedFamilyMetadata, List[str]]] = []
        for metadata in self._catalog.values():
            matched_terms = _matched_terms(normalized_text, metadata.keywords, metadata.family, metadata.display_name)
            score = _score_match(metadata.family, matched_terms)
            scores.append((score, metadata, matched_terms))

        scores.sort(key=lambda item: (item[0], item[1].family), reverse=True)
        best_score, best_metadata, best_terms = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else -1

        if best_score <= 0:
            supported = ", ".join(sorted(self._catalog))
            return SupportedFamilyMatch(
                status="no_supported_family",
                reason=(
                    "The request did not match any supported blueprint family keywords. "
                    f"Supported families: {supported}."
                ),
                metadata={"request_text": request_text},
            )

        if best_score == second_score:
            return SupportedFamilyMatch(
                status="no_supported_family",
                reason=(
                    f"The request matched multiple supported families equally ({best_metadata.family}); "
                    "backend routing failed closed instead of guessing."
                ),
                matched_terms=best_terms,
                score=best_score,
                metadata={"request_text": request_text},
            )

        return SupportedFamilyMatch(
            status="matched",
            family=best_metadata.family,
            matched_terms=best_terms,
            reason=(
                f"Matched the supported family '{best_metadata.family}' using repo-owned keywords and blueprint metadata."
            ),
            required_domains=list(best_metadata.required_domains),
            score=best_score,
            metadata=best_metadata.to_dict(),
        )


def _normalize_text(value: str) -> str:
    return " ".join(part for part in _TOKEN_RE.sub(" ", value.lower()).split() if part)


def _matched_terms(
    normalized_text: str,
    keywords: Iterable[str],
    family: str,
    display_name: str,
) -> List[str]:
    matches: list[str] = []
    phrases = [family, family.replace("-", " "), display_name, *keywords]
    for phrase in phrases:
        normalized_phrase = _normalize_text(phrase)
        if normalized_phrase and normalized_phrase in normalized_text and normalized_phrase not in matches:
            matches.append(normalized_phrase)
    return matches


def _score_match(family: str, matched_terms: Iterable[str]) -> int:
    score = 0
    exact_tokens = {_normalize_text(family), _normalize_text(family.replace("-", " "))}
    for term in matched_terms:
        if term in exact_tokens:
            score += 6
        elif " " in term:
            score += 3
        else:
            score += 1
    return score
