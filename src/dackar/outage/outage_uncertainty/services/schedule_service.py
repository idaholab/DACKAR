from __future__ import annotations


class ScheduleRiskService:
    def __init__(self, outage_risk_workflow):
        self.outage_risk_workflow = outage_risk_workflow

    def assess_outage(self, planned_outage, historical_activities):
        return self.outage_risk_workflow.run(planned_outage, historical_activities)
