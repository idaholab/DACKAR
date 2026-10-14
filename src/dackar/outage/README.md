# Outage Uncertainty Skeleton

Starter Python skeleton for outage activity similarity, duration uncertainty estimation,
and schedule risk analysis.

## Suggested next steps
1. Create a virtual environment and install dependencies you choose.
2. Replace placeholder logic in preprocessing, retrieval, and uncertainty modules.
3. Add unit tests under `tests/`.
4. Wire the workflow to DACKAR and your outage datasets.

## Package layout
- `outage_uncertainty/domain`: core domain models
- `outage_uncertainty/preprocessing`: text cleaning and feature preparation
- `outage_uncertainty/retrieval`: similarity search
- `outage_uncertainty/uncertainty`: distribution estimation and confidence
- `outage_uncertainty/schedule_risk`: schedule network and Monte Carlo logic
- `outage_uncertainty/workflows`: orchestration workflows
- `outage_uncertainty/adapters`: adapters for DACKAR, RAVEN, pandas, etc.
- `outage_uncertainty/services`: higher-level service interfaces
- `outage_uncertainty/api`: config and CLI entry points
