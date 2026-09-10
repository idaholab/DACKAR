from outage_uncertainty.api.facade import build_duration_uncertainty_service
from outage_uncertainty.services.explanation_service import ExplanationService


def run_demo() -> None:
    service = build_duration_uncertainty_service(
        abbreviations={"CAL": "CALIBRATION", "PRESS": "PRESSURE", "TX": "TRANSMITTER"},
        taxonomy_rules={
            "transmitter": {"discipline": "I&C", "task_family": "calibration", "component_family": "transmitter"}
        },
    )

    query_row = {
        "activity_id": "Q-001",
        "outage_id": "OUT-PLANNED",
        "plant_id": "PLANT-X",
        "raw_description": "CAL PRESS TX",
        "planned_duration_hours": 8.0,
    }

    historical_rows = [
        {
            "activity_id": "H-001",
            "outage_id": "OUT-001",
            "plant_id": "PLANT-X",
            "raw_description": "PRESSURE TRANSMITTER CALIBRATION",
            "actual_duration_hours": 8.5,
            "discipline": "I&C",
            "task_family": "calibration",
            "component_family": "transmitter",
        },
        {
            "activity_id": "H-002",
            "outage_id": "OUT-002",
            "plant_id": "PLANT-X",
            "raw_description": "CALIBRATE PRESSURE TRANSMITTER",
            "actual_duration_hours": 7.0,
            "discipline": "I&C",
            "task_family": "calibration",
            "component_family": "transmitter",
        },
    ]

    estimate = service.estimate_activity(query_row, historical_rows)
    print(ExplanationService().explain_estimate(estimate))


if __name__ == "__main__":
    run_demo()
