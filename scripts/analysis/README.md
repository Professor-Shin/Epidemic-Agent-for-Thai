# Report analysis scripts

Every number and figure in `report/final_report_TH.md` is produced here, reading `outputs/runs/` directly
rather than `outputs/metrics.csv`, so a run that finished after the last analysis job is still picked up.
`outputs/` is gitignored, so these scripts are the only committed record of how a reported number was
derived; the run ids themselves are listed in the appendix of `report/working_notes_TH.md`.

All of them run from any working directory. Only `policy_decomposition.py` (`--seed`, `--days`) and
`permutation_test.py` (`--condition`, `--days`) take arguments; the rest take none.

| script | report sections |
|---|---|
| `headline_tables.py` | 5.1 baselines · 5.3 the 2×2 and the summary table · 5.5 leader attack rates |
| `seir_validation.py` | 5.1 condition A run to convergence against the analytic final size |
| `policy_decomposition.py` | 5.5.1 the mechanical / behavioural split of the C−D gap |
| `permutation_test.py` | 5.3 exact permutation p for the localisation effect in each family |
| `reason_analysis.py` | 5.4 reason diversity, top reasons and keyword frames |
| `order_check.py` | 5.7 action-order robustness |
| `robustness_tables.py` | 5.8 sensitivity · 5.9 `baseline_mix` · 5.10 long horizon |
| `fig_2x2_and_reasons.py` | figures 1 and 2 |
| `fig_leader_paths.py` | figure 3 |
| `fig_sensitivity.py` | figure 4 |

Four conventions are easy to get wrong and are fixed here deliberately:

- **Reason uniqueness** is counted on the stripped, lower-cased string. Counting the raw string instead
  moves `typhoon8b` by more than a point.
- **The analytic final size must match the seeded initial condition.** The model starts with
  E = I = 0.002, so S0/N = 0.996 and the relation to solve is `s = s0 * exp(-R0 (1 - s))`, giving
  0.798176. The textbook `z = 1 - exp(-R0 z)` assumes a fully susceptible population and gives 0.796812.
  Comparing against the textbook value understates the agreement and leaves an unexplained 0.0014.
- **A permutation p can sit on a floor set by the seed count.** Two-sided, the floor is
  `2 / C(n_a + n_b, n_a)`: 0.10 at 3 seeds per model, 0.0022 at 6. With an effect 10-65x the pooled sd,
  every relabelling is less extreme than the observed one, so p lands exactly on that floor and is
  reporting the design, not the evidence. `permutation_test.py` prints the floor beside every p and
  flags the ones sitting on it.
- **The health states are three, not two.** `no symptoms`, `fever and cough` and `recovered` are distinct
  strings in the prompt. Section 5.6 contrasts symptomatic days against everything else, so recovered
  agents sit in the comparison group; splitting them out changes the sign for three models.
