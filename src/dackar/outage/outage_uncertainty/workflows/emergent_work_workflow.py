from __future__ import annotations


class EmergentWorkWorkflow:
    def __init__(self, insertion_strategy):
        self.insertion_strategy = insertion_strategy

    def run(self, emergent_activity, schedule_network):
        return self.insertion_strategy.insert(emergent_activity, schedule_network)
