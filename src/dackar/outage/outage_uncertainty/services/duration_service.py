from __future__ import annotations

from outage_uncertainty.domain.result_types import ActivityEstimate


class DurationUncertaintyService:
    def __init__(self, ingestion_workflow, similarity_workflow):
        self.ingestion_workflow = ingestion_workflow
        self.similarity_workflow = similarity_workflow

    def estimate_activity(self, query_row: dict, historical_rows) -> ActivityEstimate:
        historical = self.ingestion_workflow.run(historical_rows)
        query = self.ingestion_workflow.run([query_row])[0]
        return self.similarity_workflow.run(query, historical)
