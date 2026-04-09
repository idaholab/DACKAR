from __future__ import annotations


class ActivityService:
    def __init__(self, ingestion_workflow):
        self.ingestion_workflow = ingestion_workflow

    def ingest(self, rows):
        return self.ingestion_workflow.run(rows)
