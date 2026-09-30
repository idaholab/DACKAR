from __future__ import annotations


class Neo4jRepository:
    def __init__(self, driver=None):
        self.driver = driver

    def save_activity_graph(self, activities) -> None:
        # Placeholder for future graph persistence.
        del activities

    def fetch_neighbors(self, activity_id: str) -> list[str]:
        del activity_id
        return []
