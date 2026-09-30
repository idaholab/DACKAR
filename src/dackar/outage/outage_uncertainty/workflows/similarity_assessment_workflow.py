from __future__ import annotations

from outage_uncertainty.domain.activity import ActivityCase
from outage_uncertainty.domain.result_types import ActivityEstimate


class SimilarityAssessmentWorkflow:
    def __init__(
        self,
        index,
        similarity_engine,
        neighbor_selector,
        duration_estimator,
        prescorer_top_k: int = 200,
    ):
        self.index = index
        self.similarity_engine = similarity_engine
        self.neighbor_selector = neighbor_selector
        self.duration_estimator = duration_estimator
        self.prescorer_top_k = prescorer_top_k

    def run(self, query_activity: ActivityCase, historical_activities: list[ActivityCase]) -> ActivityEstimate:
        self.index.build(historical_activities)
        candidate_ids = self.index.search(query_activity, top_k=self.prescorer_top_k)
        # index.get() returns None for unknown IDs (e.g. if build() was called
        # concurrently between search() and get()).  Filter defensively.
        candidates = [
            a for activity_id in candidate_ids
            if (a := self.index.get(activity_id)) is not None
        ]

        matches = [self.similarity_engine.compare(query_activity, candidate) for candidate in candidates]
        neighbors = self.neighbor_selector.select(matches)
        return self.duration_estimator.estimate(query_activity, neighbors, historical_activities)
