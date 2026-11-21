import numpy as np
from typing import Optional


def compute_slate_item_coverage(
    slates: np.ndarray,
    total_items: int,
    padding_id: Optional[int] = None,
    no_choice_id: Optional[int] = None,
) -> float:
    unique_items = np.unique(slates.flatten())

    if padding_id is not None:
        unique_items = unique_items[unique_items != padding_id]
    if no_choice_id is not None:
        unique_items = unique_items[unique_items != no_choice_id]

    coverage = len(unique_items) / total_items
    return coverage


def compute_slate_diversity(
    slates: np.ndarray,
    item_embeddings: Optional[np.ndarray] = None,
    padding_id: Optional[int] = None,
    no_choice_id: Optional[int] = None,
    random_item_tuples_sims: Optional[dict] = None,
) -> dict:
    metrics = {}

    if item_embeddings is not None:
        dissimilarities = []
        for slate in slates:
            valid_items = slate.copy()
            if padding_id is not None:
                valid_items = valid_items[valid_items != padding_id]
            if no_choice_id is not None:
                valid_items = valid_items[valid_items != no_choice_id]

            if len(valid_items) > 1:
                slate_embeddings = item_embeddings[valid_items]

                norms = np.linalg.norm(slate_embeddings, axis=1, keepdims=True)
                normalized_embeddings = slate_embeddings / (norms + 1e-10)
                similarity_matrix = normalized_embeddings @ normalized_embeddings.T

                triu_indices = np.triu_indices_from(similarity_matrix, k=1)
                similarities = similarity_matrix[triu_indices]

                similarities = (similarities + 1.0) / 2.0

                individual_dissimilarities = 1.0 - similarities

                avg_dissimilarity = individual_dissimilarities.mean()
                dissimilarities.append(avg_dissimilarity)

        if random_item_tuples_sims is not None:
            random_item_tuples_sims = 1 - (1 + np.array(random_item_tuples_sims)) / 2.0
            dissimilarities = np.array(dissimilarities)
            percentiles = []
            for dissim in dissimilarities:
                percentile = np.mean(
                    random_item_tuples_sims <= dissim
                ) * 100.0
                percentiles.append(percentile)

            metrics['embedding_diversity_percentile'] = np.mean(percentiles) if percentiles else 0.0

    return metrics
