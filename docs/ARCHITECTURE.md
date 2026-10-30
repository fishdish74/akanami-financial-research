# Architecture

The system separates untrusted source and model content from trusted control logic.

```text
source adapter
  -> Candidate schema
  -> deterministic relevance and quality filters
  -> duplicate and event classification
  -> optional structured analysis
  -> code-owned identifiers, confidence, and metadata
  -> full semantic validation
  -> deterministic Markdown renderer
  -> rendered-output validation
  -> dry-run report
  -> optional guarded create-only write
```

## Boundaries

**Source boundary.** Each adapter owns its timeout, retry policy, headers, parsing, and warning isolation. A failed source cannot silently become trusted input.

**Analysis boundary.** The optional model returns structured data rather than Markdown, paths, or filenames. Code validates the response and derives control fields.

**Persistence boundary.** SQLite access is versioned and transactional. Point-in-time queries retain filing and availability dates so later information is not silently used in an earlier analysis.

**Filesystem boundary.** Rendered content must pass validation before an atomic create-only write inside the configured root. Existing files are not overwritten.

## Offline portfolio path

The bundled fixtures exercise normalization, filtering, deduplication, schema validation, deterministic analysis, analytics, and rendering. The public demonstration does not require the private database or any provider credentials.
