"""
cmms_integration — Live CMMS context retrieval for RCA reasoning.

Fetches Corrective Action records (CRs) and Work Orders (WOs) from the
plant CMMS at RCA invocation time.  Structured data flows into the
``cmms_context`` artifact; narrative text is embedded and injected into
the run-scoped Chroma collection for semantic evidence retrieval.

See CMMS_INTEGRATION_GUIDE.md for architecture details and the
implementation skeleton for live adapters.
"""
