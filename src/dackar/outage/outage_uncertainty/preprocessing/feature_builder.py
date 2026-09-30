from __future__ import annotations

from outage_uncertainty.domain.activity import ActivityCase


class SimpleEmbedder:
    def encode(self, text: str) -> list[float]:
        # Placeholder embedding
        return [float(len(text or ""))]


class ActivityFeatureBuilder:
    def __init__(self, embedder=None):
        self.embedder = embedder or SimpleEmbedder()

    def build_features(self, activity: ActivityCase) -> dict:
        text = activity.cleaned_description or activity.raw_description
        return {
            "text_embedding": self.embedder.encode(text),
            "discipline": activity.discipline,
            "task_family": activity.task_family,
            "component_family": activity.component_family,
            "system_name": activity.system_name,
            "is_emergent": activity.is_emergent,
            "crew_size": activity.crew_size,
            "planned_duration": activity.planned_duration_hours,
            "outage_phase": activity.outage_phase,
        }
