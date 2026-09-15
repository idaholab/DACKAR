from __future__ import annotations

import json
from outage_uncertainty.api.facade import build_duration_uncertainty_service
from outage_uncertainty.services.explanation_service import ExplanationService


def main() -> None:
    service = build_duration_uncertainty_service()

    query_row = {
        "activity_id": "Q1",
        "outage_id": "O-PLANNED",
        "plant_id": "PLANT-A",
        "raw_description": "CALIBRATE PRESSURE TRANSMITTER",
        "planned_duration_hours": 8.0,
    }

    historical_rows = [
        {
            "activity_id": "H1",
            "outage_id": "O1",
            "plant_id": "PLANT-A",
            "raw_description": "CAL PRESS TX",
            "actual_duration_hours": 7.5,
            "discipline": "I&C",
        },
        {
            "activity_id": "H2",
            "outage_id": "O2",
            "plant_id": "PLANT-A",
            "raw_description": "PRESSURE TRANSMITTER CALIBRATION",
            "actual_duration_hours": 9.0,
            "discipline": "I&C",
        },
    ]

    estimate = service.estimate_activity(query_row, historical_rows)
    explanation = ExplanationService().explain_estimate(estimate)
    print(json.dumps(explanation, indent=2))


if __name__ == "__main__":
    main()
