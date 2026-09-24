# Insulator annotation: cCRE x cell-type counts and proportions

Reuses the ChromHMM coverage matrices built by
`revision/silencer/code/annotate_chromstates.py` (390 cCREs x 151 Allen
subclasses, one fraction matrix per 8-state label) and applies a different
label:

```
insulator  <=>  Chr-O fraction > 0.9  AND  Chr-A fraction < 0.1
```

i.e. the cCRE's bp are almost entirely in the "open, other" state and carry
essentially no active chromatin. Cutoffs are `--open-cutoff` / `--active-cutoff`.
chrX is unsegmented, so its 14 cCREs are NaN in both matrices and count as
neither insulator nor non-insulator.

## Denominator

The proportion is **insulators / cCREs tested in that cell type** — the tested
set is the pairs eligible at T7 >= 50 in
`revision/silencer/results/silencer_left_tail_tests.csv.gz`, the same denominator
as `plot_silencer_recall_by_celltype.py`, so the insulator and silencer
per-cell-type figures are read side by side. That is `--scope tested`, the
default: 2290 pairs, 28 cell types, 67 insulators, 17 cell types with >=1.

`--scope annotation` instead scores the whole panel (376 cCREs per cell type,
56776 pairs over 151 subclasses, 596 insulators). Every cell type then shares the
identical denominator, so its proportion bar is the count bar rescaled and says
nothing per cell type; it is kept only for the panel-wide totals and writes no
committed figure.

## Run

```bash
conda activate scvi
cd /gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
PYTHONPATH=revision/insulator/code python \
    revision/insulator/code/plot_insulator_by_celltype.py
```

`--top-n` caps how many cell types are drawn (default 30, by insulator count;
`0` draws all). Cell types with no annotated insulator stay in the CSV and are
left out of the figure.

## Scripts

| Script | Role |
|---|---|
| `code/insulator_truth.py` | loads the `Chr_O` / `Chr_A` / `ND` matrices as cell type x cCRE and derives the insulator mask; shares the filename -> subclass mapping with `revision/silencer/code/silencer_truth.py` so pairs join across the two annotations |
| `code/plot_insulator_by_celltype.py` | per-cell-type table and the two-bar figure (count on the left axis, proportion on the right) |
| `code/aggregate_labels_across_celltypes.py` | both labels collapsed over cell types: pair, cCRE and cell-type totals, their overlap, and per-cCRE breadth |

## Outputs

| File | Contents |
|---|---|
| `results/insulator_by_celltype_tested.csv` | per cell type: `n_cre_tested`, `n_insulator`, `fraction_insulator` |
| `figures/insulator_by_celltype_tested.pdf` | the 17 cell types with >=1 insulator among tested pairs |
| `results/label_totals_across_celltypes.csv` | silencer and insulator totals per scope: pairs, cCREs with >=1 cell type, cell types with >=1 cCRE |
| `results/label_totals_across_celltypes_overlap.csv` | pairs carrying both labels, and cCREs silencer-only / insulator-only / both / either |
| `results/label_totals_across_celltypes_per_cre_{panel,tested}.csv` | per cCRE, how many cell types give it each label |

## Aggregated over cell types

```bash
PYTHONPATH=revision/insulator/code:revision/silencer/code python \
    revision/insulator/code/aggregate_labels_across_celltypes.py
```

| scope | label | pairs | cCREs with >=1 cell type | cell types with >=1 cCRE |
|---|---|---|---|---|
| panel (56776 pairs) | silencer > 0.1 | 3163 (5.6%) | 240 / 376 | 133 / 151 |
| panel | insulator | 596 (1.0%) | 245 / 376 | 112 / 151 |
| tested (2290 pairs) | silencer > 0.1 | 268 (11.7%) | 71 / 185 | 22 / 28 |
| tested | insulator | 67 (2.9%) | 39 / 185 | 17 / 28 |

The labels are mutually exclusive pair by pair -- 0 pairs carry both, since
`Chr-O > 0.9` leaves under 0.1 for `Chr-R` + `Hc-P` -- but they overlap heavily
on cCREs across cell types: of the 342 panel cCREs carrying either label, 143
carry both in different cell types (15 of 95 on tested pairs).

Insulators reach nearly as many cCREs as silencers (245 vs 240) from a fifth as
many pairs, because insulator calls are cell-type-specific: the median labelled
cCRE is an insulator in 2 cell types (max 22) and a silencer in 7 (max 75).

The tested subset is enriched for both labels relative to the panel (silencer
11.7% vs 5.6%, insulator 2.9% vs 1.0%), so per-cell-type rates computed on
tested pairs sit on a denominator that is not representative of the panel.

## Result

Insulators are rare on this panel: 1.0% of annotated pairs (596/56776) and 2.9%
of tested pairs (67/2290), against 11.7% for the silencer label at `> 0.1`. The
per-cell-type maximum is 23 (L5 ET CTX Glut, whole panel) and 11 (same cell type,
tested pairs), so no cell type carries enough insulators to support a
precision/recall benchmark of the kind run in `revision/silencer/`.
