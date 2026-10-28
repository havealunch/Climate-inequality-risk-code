# Climate-inequality-risk-code

Single-file replication package for **“Conditional climate inequality risk from global patterns to household spending.”**

## What this package does

`replication_analysis.py` is the complete public workflow. It contains, in one readable script:

1. Global country two-way fixed-effects estimates and, where lag columns are supplied, distributed-lag estimates and joint tests;
2. China city-level flood-exposure models, COVID-19 adjustment, city-specific linear trends, province-by-year fixed effects and continuous-exposure event-study/pretrend tests;
3. Nighttime-light sensitivity models;
4. CFPS household fixed-effects models, dynamic event studies, pretrend tests, alternative exposure measures, COVID-19 adjustment, province-by-year fixed effects and alternative expenditure outcomes;
5. The fixed-budget universal, equal-targeted and shortfall-proportional allocation scenarios over 5–25% budget levels; and
6. Machine-readable result tables and a compact diagnostic coefficient figure.

## Install and run

```bash
python -m pip install -r requirements.txt
python replication_analysis.py --analysis all --input data --output outputs
```

Run one component with `--analysis global`, `china`, `nighttime_lights`, `household`, `allocation` or `figures`.

## Local input files

Place inputs in a local `data/` folder. Do not upload that folder to GitHub.

| File                       | Required fields                                                                                                                                                                                                             |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `global_panel.csv`         | `country`, `year`, `gini_change_annualized`, `loss_gdp_winsor`; optional controls: `gdp_pc_growth`, `urbanization`, `trade_openness`, `control_corruption`, `education_expenditure`; optional lags: `loss_lag1`–`loss_lag5` |
| `china_city_panel.csv`     | `city`, `year`, `ln_urban_rural_ratio`, `exposure_z`, `post2020`; optional `covid_cases_per_100k`, `province`                                                                                                               |
| `china_ntl_panel.csv`      | `city`, `year`, `log_ntl`, `exposure_z`, `post2020`                                                                                                                                                                         |
| `cfps_household_panel.csv` | `household_id`, `city`, `year`, `ln_education_pc`, `ln_medical_pc`, `exposure_z`, `post2020`; optional `covid_cases_per_100k`, `province`, `precip_z`, `runoff_z`, expenditure-share outcomes                               |
| `cfps_policy_panel.csv`    | `household_id`, `year`, `income_pc`, `education_pc`, `medical_pc`                                                                                                                                                           |

## Data restrictions

No CFPS microdata, household-derived panel, household–city crosswalk, restricted geographical identifier, EM-DAT event-level record or restricted raster is supplied here. CFPS users must obtain authorised access at https://cfpsdata.pku.edu.cn/ or https://opendata.pku.edu.cn/dataverse/CFPS and create the local inputs themselves. Non-restricted numerical source data and full external-source documentation are archived at https://doi.org/10.6084/m9.figshare.32789007.

## Outputs

The script writes CSV coefficient tables, joint-test tables, allocation sensitivity tables and `06_replication_diagnostic.png` to `outputs/`. Compare numerical outputs to the Figshare source-data files. The code does not redistribute data and cannot replace source-specific access permissions.

## Software

The tested dependency versions are pinned in `requirements.txt`. The script uses a transparent dummy-variable fixed-effects estimator with CR1 clustered covariance estimates. It does not introduce new software or a new computational method.
