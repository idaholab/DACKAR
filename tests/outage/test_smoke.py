from outage_uncertainty.api.facade import build_duration_uncertainty_service


def test_smoke_estimate_activity():
    service = build_duration_uncertainty_service()
    estimate = service.estimate_activity(
        query_row={
            "activity_id": "Q1",
            "outage_id": "OQ",
            "plant_id": "P1",
            "raw_description": "TEST ACTIVITY",
            "planned_duration_hours": 4.0,
        },
        historical_rows=[
            {
                "activity_id": "H1",
                "outage_id": "OH1",
                "plant_id": "P1",
                "raw_description": "TEST ACTIVITY",
                "actual_duration_hours": 5.0,
            }
        ],
    )
    assert estimate.activity_id == "Q1"
    assert estimate.estimated_distribution is not None
