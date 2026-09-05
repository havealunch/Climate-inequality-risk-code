# Climate-inequality-risk-code
# Replication code package

This package contains transparent baseline scripts for the three empirical components and the budget-neutral allocation exercise. It does not contain restricted CFPS microdata or licensed spatial source files.

## Required directory structure

```text
data/
  global_panel.csv
  china_city_panel.csv
  cfps_household_panel.csv
  cfps_policy_panel.csv
outputs/
```

## Required columns

`global_panel.csv`: `country`, `year`, `gini_change_annualized`, `loss_gdp_winsor`, `gdp_pc_growth`, `urbanization`, `trade_openness`, `control_corruption`, `education_expenditure`.

`china_city_panel.csv`: `city`, `year`, `ln_urban_rural_ratio`, `exposure_z`, `post2020`, `covid_cases_per_100k`.

`cfps_household_panel.csv`: `household_id`, `year`, `ln_education_pc`, `ln_medical_pc`, `exposure_z`, `post2020`.

`cfps_policy_panel.csv`: `household_id`, `year`, `income_pc`, `education_pc`, `medical_pc`.

For CFPS, use only the authorised raw files to construct the derived inputs. Do not redistribute raw records. The submitted repository should include the de-identified derived data, this documentation, the variable construction log, and the scripts.

## Run

```bash
python -m pip install -r requirements.txt
python run_replication.py global --input data/global_panel.csv --output outputs
python run_replication.py china --input data/china_city_panel.csv --output outputs
python run_replication.py household --input data/cfps_household_panel.csv --output outputs
python run_replication.py policy --input data/cfps_policy_panel.csv --output outputs
```

The scripts estimate the baseline specifications only. Exact map production, flood-raster aggregation and CFPS harmonisation should be uploaded as additional scripts when the final processing workflow is consolidated.
