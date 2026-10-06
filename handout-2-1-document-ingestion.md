# Handout 2.1 — Document ingestion

Paste this prompt into your coding assistant from the capstone repository:

---

Create `docs/stories/story-2-1-document-ingestion.md`. Create the story only; do **not** implement it now. Read `docs/config.yaml`, `docs/architecture.md`, the seed story, existing code, and supplied `data/raw/` first. Preserve existing work.

Write a focused implementation story in plain English with purpose, prerequisites, work to do, completion checks, and handover. It follows Story 1.1 and must reuse its application, documents, and existing project `.env.example` contract; do not add or rename settings. Its only goal is to turn the supplied BNS and IPC PDFs into a small, inspectable section-level corpus. MongoDB, embeddings, vector indexes, retrieval, and answer generation belong to later stories.

Require the implementer to preserve the original PDFs and `data/raw/PROVENANCE.md`, report any missing input, choose a suitable PDF parsing dependency through UV when needed, and extract sections without rewriting their meaning. Make JSONL records compatible with the classroom project’s later importer: `section_id`, `act`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `heading`, `text`, `source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version`, and `needs_review`. Use exactly these values: `act` is `BNS_2023` or `IPC_1860`; `status` is `in_force` (BNS) or `repealed` (IPC); `section_id` is `bns:<number>` or `ipc:<number>`; `section_number` is an integer; `chapter` is a string (for example `I`). Write the files to `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl`. Page references may be retained as additional extraction data but are not a substitute for these fields. Do not invent unknown provenance. Keep BNS and IPC records separate even where section numbers match. If source hashes change, regenerate the derived corpus safely as one replacement rather than mixing records from versions.

The story must produce a JSONL corpus in the agreed project data location, one record per line, without overwriting a supplied valid corpus unnecessarily. It should handle repeated page headers, spacing, multi-page sections, and obvious duplicate extraction conservatively; flag uncertain or scanned content rather than silently repairing it. Re-running must not append duplicate records. Record the parser choice, output format, command, and known limitations in `docs/architecture.md`.

Ask for only small, directly relevant verification: inspect representative records against each PDF, including a section spanning pages if present; check required fields; ensure no empty section text; and run the extraction a second time safely. Do not require a comprehensive corpus-validation framework, a reference-difference report, broad security tests, or synthetic permission records.

The completed story must name its inputs and output, explain how later chunking will preserve the parent section and source information, and report what was generated and checked. After creating the story, report its path only.
