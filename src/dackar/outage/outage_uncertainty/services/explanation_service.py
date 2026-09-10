from __future__ import annotations

from outage_uncertainty.domain.result_types import ActivityEstimate


class ExplanationService:
    def explain_estimate(self, estimate: ActivityEstimate) -> dict:
        return {
            "activity_id": estimate.activity_id,
            "support_count": estimate.support_count,
            "confidence": estimate.confidence_score,
            "distribution": {
                "type": estimate.estimated_distribution.distribution_type,
                "p10": estimate.estimated_distribution.p10,
                "p50": estimate.estimated_distribution.p50,
                "p80": estimate.estimated_distribution.p80,
                "p90": estimate.estimated_distribution.p90,
            },
            "top_matches": [
                {
                    "candidate_id": match.candidate_activity_id,
                    "score": match.total_score,
                    "duration": match.candidate_duration_hours,
                }
                for match in estimate.matched_cases[:5]
            ],
            "warnings": estimate.warnings,
        }
