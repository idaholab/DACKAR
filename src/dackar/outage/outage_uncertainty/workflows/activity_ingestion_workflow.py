from __future__ import annotations

from outage_uncertainty.domain.activity import ActivityCase


class ActivityIngestionWorkflow:
    def __init__(self, repository, cleaner, label_mapper, feature_builder, validator=None):
        self.repository = repository
        self.cleaner = cleaner
        self.label_mapper = label_mapper
        self.feature_builder = feature_builder
        self.validator = validator

    def run(self, rows) -> list[ActivityCase]:
        activities = self.repository.load_activities(rows)
        processed: list[ActivityCase] = []

        for activity in activities:
            if self.validator:
                errors = self.validator.validate(activity)
                if errors:
                    activity.metadata["validation_errors"] = errors

            activity = self.cleaner.clean(activity)
            activity = self.label_mapper.map(activity)
            activity.metadata["features"] = self.feature_builder.build_features(activity)
            processed.append(activity)

        return processed
