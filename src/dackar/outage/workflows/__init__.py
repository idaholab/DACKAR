"""
Outage analytics higher-level workflows.

Workflows coordinate multiple pipeline components and orchestrator calls
to implement complete operational procedures that span the A–G stage pipeline.

Available workflows
-------------------
CompletionFeedbackWorkflow
    Post-completion ingestion of actual execution data back into the
    historical analog index (M5 — learning loop closure).
"""
