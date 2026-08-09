# labels/

`labels.jsonl` — the 0–9 aesthetic labels, one JSON object per line,
**append-only** and **committed to git**.

`tags.jsonl` — which *aesthetic universe* each image belongs to, same rules.
A score says how GOOD an image is; a tag says what KIND it is, and the two are
independent: a 9 in one universe and a 9 in another are both keepers, but one
report card or one fitted distribution spanning both describes an image that
does not exist. Each record carries the **complete** tag set for one image at
one moment, so assigning and un-assigning are the same operation, and the
vocabulary is *derived* — a tag exists because an image wears it.

This is the one output of the project that is not regenerable. Every image can
be re-rendered from its seed; the five seconds of a human eye deciding "that one
is a 7" cannot. That is why this does not live under the gitignored `outputs/`.

Rules the code enforces:

- **Nothing is ever rewritten.** Relabeling appends a new record; the reader
  takes the last record per `rel`. The file is the history.
- **Each record is self-contained.** It snapshots the experiment id, backend,
  checkpoint slug, distance, seed and the sampler knobs *at labeling time*, so it
  stays a complete data point after the PNG and its sidecars are wiped.
- **A malformed line is skipped, not fatal.** A truncated write can never cost
  you the labels either side of it.

Written by the dashboard's 🏷 Label tab; read by `scripts/experiment_report.py`
and (later) the score regressor. Schema and summary maths live in
`semantic_anarchy/labels.py`, the tag dataset in `semantic_anarchy/tags.py`. Full workflow: [docs/reference/labeling.md](../docs/reference/labeling.md).
