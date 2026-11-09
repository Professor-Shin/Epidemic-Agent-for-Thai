# Report analysis scripts

Every number and figure in `report/final_report_TH.md` is produced here, reading `outputs/runs/` directly
rather than `outputs/metrics.csv`, so a run that finished after the last analysis job is still picked up.
`outputs/` is gitignored, so these scripts are the only committed record of how a reported number was
derived; the run ids themselves are listed in the appendix of `report/working_notes_TH.md`.

All of them run from any working directory and take no arguments.

| script | report sections |
|---|---|
| `headline_tables.py` | 5.1 baselines · 5.3 the 2×2 and the summary table · 5.5 leader attack rates |
| `reason_analysis.py` | 5.4 reason diversity, top reasons and keyword frames |
| `order_check.py` | 5.7 action-order robustness |
| `robustness_tables.py` | 5.8 sensitivity · 5.9 `baseline_mix` · 5.10 long horizon |
| `fig_2x2_and_reasons.py` | figures 1 and 2 |
| `fig_leader_paths.py` | figure 3 |
| `fig_sensitivity.py` | figure 4 |

Two counting conventions are easy to get wrong and are fixed here deliberately:

- **Reason uniqueness** is counted on the stripped, lower-cased string. Counting the raw string instead
  moves `typhoon8b` by more than a point.
- **The health states are three, not two.** `no symptoms`, `fever and cough` and `recovered` are distinct
  strings in the prompt. Section 5.6 contrasts symptomatic days against everything else, so recovered
  agents sit in the comparison group; splitting them out changes the sign for three models.
