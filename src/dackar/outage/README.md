# Outage Uncertainty Skeleton

Starter Python skeleton for outage activity similarity, duration uncertainty estimation,
and schedule risk analysis.

## Suggested next steps
1. From the repo root, run `uv sync --group dev` (the outage subpackage currently has no extra deps beyond core; this will change as the skeleton is filled in).
2. Replace placeholder logic in preprocessing, retrieval, and uncertainty modules.
3. Add unit tests under `tests/`.
4. Wire the workflow to DACKAR and your outage datasets.

> **TODO:** populate an `outage` dependency group in the root `pyproject.toml` once the outage modules declare their concrete dependencies. The standalone `src/dackar/outage/pyproject.toml` is retained as a skeleton; it is not currently a uv workspace member.

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
