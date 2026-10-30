# Local Generated Data

This directory is reserved for local SQLite databases and provider caches.

Expected runtime layout:

```text
data/
├── market_data.sqlite3
└── sec/
    ├── raw/
    └── metadata/
```

Generated databases, raw SEC responses, metadata caches, WAL files, and temporary files are ignored by Git. Only this README is intended to be version controlled.

Raw source documents are content-addressed and immutable. Normalized database rows retain their source hash, taxonomy concept, accession number, filing date, and availability date.
