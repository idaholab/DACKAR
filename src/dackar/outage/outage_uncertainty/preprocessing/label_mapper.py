from __future__ import annotations

import re

from outage_uncertainty.domain.activity import ActivityCase
from outage_uncertainty.preprocessing.default_taxonomy import DEFAULT_TAXONOMY_RULES


class TaskLabelMapper:
    """Map free-text activity descriptions to controlled taxonomy labels.

    Scans the cleaned (or raw) description for known keywords and populates
    ``discipline``, ``task_family``, and ``component_family`` on the
    :class:`ActivityCase`.  Activities that already carry all three labels are
    returned unchanged.

    Matching uses whole-word boundaries (``\\b``), so ``"test"`` fires on
    ``"test"`` and ``"pressure test"`` but *not* on ``"protesting"`` or
    ``"latest"``.

    Iteration order: single-word entries are visited before phrase entries,
    so more-specific phrases win via last-match semantics.

    Args:
        taxonomy_rules: Extra or override ``{keyword: {field: value}}`` mappings
            merged *on top of* the defaults.  Takes precedence over the built-in
            vocabulary.
        use_defaults: When ``True`` (default) the built-in
            :data:`DEFAULT_TAXONOMY_RULES` vocabulary is loaded first.  Set to
            ``False`` to rely solely on ``taxonomy_rules``.
    """

    def __init__(
        self,
        taxonomy_rules: dict[str, dict[str, str]] | None = None,
        *,
        use_defaults: bool = True,
    ) -> None:
        rules: dict[str, dict[str, str]] = {}
        if use_defaults:
            rules.update(DEFAULT_TAXONOMY_RULES)
        rules.update(taxonomy_rules or {})
        self.taxonomy_rules = rules

        # Pre-compile patterns once for performance.
        # Each pattern is  r'\b<escaped keyword>\b'  with IGNORECASE.
        self._patterns: list[tuple[re.Pattern, dict[str, str]]] = [
            (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), labels)
            for kw, labels in self.taxonomy_rules.items()
        ]

    def map(self, activity: ActivityCase) -> ActivityCase:
        """Apply taxonomy labels inferred from the activity description.

        If all three label fields are already populated the activity is
        returned as-is (caller-supplied metadata is never overwritten).
        """
        if activity.discipline and activity.task_family and activity.component_family:
            return activity

        text = activity.cleaned_description or activity.raw_description or ""

        # Collect inferred labels from all matching keywords.
        # Iteration order is insertion order (single words → phrases), so each
        # subsequent match overwrites the previous one: phrases win over the
        # single-word entries they contain.
        inferred: dict[str, str] = {}
        for pattern, labels in self._patterns:
            if pattern.search(text):
                inferred.update(labels)

        # Apply inferred labels only to fields that are still unset.
        # Caller-supplied data (from P6 or CSV) is never overwritten.
        if not activity.discipline:
            activity.discipline = inferred.get("discipline")
        if not activity.task_family:
            activity.task_family = inferred.get("task_family")
        if not activity.component_family:
            activity.component_family = inferred.get("component_family")

        return activity
