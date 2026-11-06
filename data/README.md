# data/

Local corpus and working files. Everything here except this README is git-ignored
(decision D3): original documents are preserved unchanged under `raw/`, and their
provenance (URL, timestamp, SHA-256) is in `manifests/`.

| Path | Contents |
|---|---|
| `raw/<source>/...` | Original downloaded files, never transformed |
| `raw/openfda_bulk/<export_date>/` | openFDA bulk exports used for discovery |
| `discovery/` | Inventory and candidate-selection outputs |
| `manifests/batches/` | Batch plans, downloaded/failed file manifests, collection reports |

Rebuild with the scripts in `scripts/phase1/` (see the top-level README).
