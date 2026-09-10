from __future__ import annotations


class ActivityService:
    def __init__(self, ingestion_workflow):
        self.ingestion_workflow = ingestion_workflow

    def ingest(self, rows):
        return self.ingestion_workflow.run(rows)

    def ingest_from_p6(
        self,
        dataset,   # OutageDataset  (P6_adapter.outage_model.dataset)
        outage,    # Outage          (P6_adapter.outage_model.models)
        *,
        skip_milestones: bool = True,
        emergent_change_types: set[str] | None = None,
        contractor_org_units: set[str] | None = None,
    ):
        """Ingest a P6 ``OutageDataset`` through the standard ingestion pipeline.

        Converts ``dataset.schedule_tasks`` (and related join tables) to
        ``ActivityCase`` objects via :class:`~outage_uncertainty.adapters.p6_dataset_adapter.P6DatasetAdapter`,
        then runs each activity through the same clean → label → embed pipeline
        as :meth:`ingest`.

        Parameters
        ----------
        dataset:
            A populated ``OutageDataset`` (from ``P6_adapter``).
        outage:
            The ``Outage`` record that owns the dataset — provides ``plant_id``
            and ``unit_id``.
        skip_milestones:
            Omit zero-duration milestone tasks (default ``True``).
        emergent_change_types:
            Override the set of ``ScopeChangeEvent.change_type`` values treated
            as emergent work.  ``None`` uses the adapter's default
            ``{"emergent", "scope_addition"}``.
        contractor_org_units:
            Set of ``Resource.org_unit`` values that identify contractor
            resources.  ``None`` (default) leaves ``contractor_flag`` as
            unknown (``None``) for all tasks.

        Returns
        -------
        list[ActivityCase]
        """
        from outage_uncertainty.adapters.p6_dataset_adapter import P6DatasetAdapter

        adapter = P6DatasetAdapter(
            emergent_change_types=emergent_change_types,
            contractor_org_units=contractor_org_units,
        )
        rows = adapter.iter_activity_rows(
            dataset, outage, skip_milestones=skip_milestones
        )
        return self.ingest(rows)

    def build_outage_record(
        self,
        dataset,   # OutageDataset  (P6_adapter.outage_model.dataset)
        outage,    # Outage          (P6_adapter.outage_model.models)
        *,
        skip_milestones: bool = True,
        emergent_change_types: set[str] | None = None,
        contractor_org_units: set[str] | None = None,
    ):
        """Build an ``OutageRecord`` ready for ``OutageRiskWorkflow``.

        This is the bridge between the P6 ingestion path and the UQ/RCPSP
        pipeline.  It combines :meth:`ingest_from_p6` (which produces the
        NLP-enriched ``ActivityCase`` list) with ``OutageRecord`` assembly
        so that a single call takes P6 data all the way to the object
        accepted by ``OutageRiskWorkflow.run()``.

        Parameters
        ----------
        dataset:
            A populated ``OutageDataset`` (from ``P6_adapter``).
        outage:
            The ``Outage`` record that owns the dataset.  Provides
            ``outage_id``, ``plant_id``, ``unit_id``, ``start_actual`` /
            ``start_planned`` (for ``start_date``), and ``finish_actual`` /
            ``finish_planned`` (for ``end_date``).
        skip_milestones:
            Forwarded to ``ingest_from_p6``.
        emergent_change_types:
            Forwarded to ``ingest_from_p6``.
        contractor_org_units:
            Forwarded to ``ingest_from_p6``.

        Returns
        -------
        OutageRecord
            Populated with NLP-enriched ``ActivityCase`` objects.

        Raises
        ------
        ValueError
            When neither ``outage.start_actual`` nor ``outage.start_planned``
            is set — ``OutageRecord.start_date`` is required.
        """
        from outage_uncertainty.domain.outage import OutageRecord

        start_date = getattr(outage, "start_actual", None) or getattr(
            outage, "start_planned", None
        )
        if start_date is None:
            raise ValueError(
                f"Cannot build OutageRecord for outage '{outage.outage_id}': "
                "neither start_actual nor start_planned is set on the Outage object."
            )

        end_date = getattr(outage, "finish_actual", None) or getattr(
            outage, "finish_planned", None
        )

        activities = self.ingest_from_p6(
            dataset,
            outage,
            skip_milestones=skip_milestones,
            emergent_change_types=emergent_change_types,
            contractor_org_units=contractor_org_units,
        )

        return OutageRecord(
            outage_id=outage.outage_id,
            plant_id=getattr(outage, "plant_id", None) or "",
            unit_id=getattr(outage, "unit_id", None),
            start_date=start_date,
            end_date=end_date,
            activities=activities,
        )
