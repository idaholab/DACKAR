from __future__ import annotations

from outage_uncertainty.domain.outage import OutageRecord


class OutageRiskWorkflow:
    def __init__(self, estimator_workflow, schedule_builder, scenario_runner):
        self.estimator_workflow = estimator_workflow
        self.schedule_builder = schedule_builder
        self.scenario_runner = scenario_runner

    def run(self, planned_outage: OutageRecord, historical_activities):
        estimates = {}
        for activity in planned_outage.activities:
            estimates[activity.activity_id] = self.estimator_workflow.run(activity, historical_activities)

        network, baseline_cp_time = self.schedule_builder.build(planned_outage, estimates)
        risk_summary = self.scenario_runner.run(network, baseline_cp_time=baseline_cp_time)

        return {
            "activity_estimates": estimates,
            "risk_summary": risk_summary,
        }
