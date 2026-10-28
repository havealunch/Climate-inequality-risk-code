#!/usr/bin/env python3
"""Replication workflow for climate-inequality-risk study.

This single script estimates the global, city, nighttime-light, household and
fixed-budget allocation analyses described in the manuscript. It expects local
authorised input files in --input and never writes data to the code repository.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ---------- Generic utilities -------------------------------------------------
def read_data(input_dir, filename, required):
    path = Path(input_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required local input: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")
    return df


def add_interaction(df, left, right, name):
    df = df.copy()
    df[name] = pd.to_numeric(df[left], errors="coerce") * pd.to_numeric(df[right], errors="coerce")
    return df


def dummy_matrix(df, columns, trend_by=None):
    pieces = []
    for col in columns:
        pieces.append(pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=float))
    if trend_by is not None:
        centered_year = pd.to_numeric(df["year"], errors="coerce") - pd.to_numeric(df["year"], errors="coerce").mean()
        trend_dummies = pd.get_dummies(df[trend_by].astype(str), prefix=f"{trend_by}_trend", drop_first=True, dtype=float)
        pieces.append(trend_dummies.mul(centered_year, axis=0))
    return pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=df.index)


def fit_fe_cluster(df, outcome, focal_terms, fixed_effects, cluster, controls=(), trend_by=None):
    """OLS with dummy-absorbed fixed effects and CR1 clustered covariance."""
    focal_terms = list(focal_terms)
    needed = [outcome, cluster, *focal_terms, *fixed_effects, *controls]
    if trend_by:
        needed.append(trend_by)
    data = df.dropna(subset=needed).copy()
    if data.empty:
        raise ValueError(f"No complete observations for {outcome}.")
    X = pd.DataFrame({"const": 1.0}, index=data.index)
    for term in focal_terms:
        X[term] = pd.to_numeric(data[term], errors="coerce")
    for control in controls:
        X[control] = pd.to_numeric(data[control], errors="coerce")
    X = pd.concat([X, dummy_matrix(data, fixed_effects, trend_by)], axis=1)
    y = pd.to_numeric(data[outcome], errors="coerce").to_numpy(float)
    Xv = X.to_numpy(float)
    keep = np.isfinite(y) & np.isfinite(Xv).all(axis=1)
    y, Xv, data, X = y[keep], Xv[keep], data.iloc[np.flatnonzero(keep)].copy(), X.iloc[np.flatnonzero(keep)].copy()
    n, k = Xv.shape
    if n <= k:
        raise ValueError(f"Insufficient observations: n={n}, parameters={k}.")
    bread = np.linalg.pinv(Xv.T @ Xv)
    beta = bread @ Xv.T @ y
    resid = y - Xv @ beta
    groups = data[cluster].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    g = len(unique_groups)
    if g < 2:
        raise ValueError("At least two clusters are required.")
    meat = np.zeros((k, k))
    for group in unique_groups:
        score = Xv[groups == group].T @ resid[groups == group]
        meat += np.outer(score, score)
    cov = (g / (g - 1)) * ((n - 1) / (n - k)) * bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=g - 1)
    crit = stats.t.ppf(0.975, df=g - 1)
    table = pd.DataFrame({
        "term": X.columns, "beta": beta, "clustered_se": se, "t_statistic": t,
        "p_value": p, "ci_low": beta - crit * se, "ci_high": beta + crit * se,
        "observations": n, "clusters": g,
    })
    return table, cov, list(X.columns), data


def joint_wald(table, covariance, columns, terms):
    indices = [columns.index(t) for t in terms if t in columns]
    if not indices:
        return {"terms": ";".join(terms), "wald_chi2": np.nan, "df": 0, "p_value": np.nan}
    b = table.set_index("term").loc[[columns[i] for i in indices], "beta"].to_numpy()
    v = covariance[np.ix_(indices, indices)]
    stat = float(b.T @ np.linalg.pinv(v) @ b)
    return {"terms": ";".join(columns[i] for i in indices), "wald_chi2": stat, "df": len(indices), "p_value": stats.chi2.sf(stat, len(indices))}


def save(table, output_dir, filename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    table.to_csv(path, index=False)
    print(f"Wrote {path}")
    return path


def focal(table, names):
    return table[table["term"].isin(names)].copy()


def event_terms(df, exposure, time, reference, prefix="exposure_y"):
    out = df.copy()
    years = sorted(pd.Series(out[time].dropna().unique()).astype(int).tolist())
    terms = []
    for year in years:
        if year == reference:
            continue
        term = f"{prefix}{year}"
        out[term] = pd.to_numeric(out[exposure], errors="coerce") * (pd.to_numeric(out[time], errors="coerce") == year)
        terms.append(term)
    return out, terms


# ---------- 1. Global country panel ------------------------------------------
def global_analysis(input_dir, output_dir):
    df = read_data(input_dir, "global_panel.csv", ["country", "year", "gini_change_annualized", "loss_gdp_winsor"])
    controls = [c for c in ["gdp_pc_growth", "urbanization", "trade_openness", "control_corruption", "education_expenditure"] if c in df]
    table, cov, cols, _ = fit_fe_cluster(df, "gini_change_annualized", ["loss_gdp_winsor"], ["country", "year"], "country", controls)
    out = focal(table, ["loss_gdp_winsor"])
    out.insert(0, "model", "Preferred global two-way fixed effects")
    out.insert(1, "outcome", "annualized_gini_change")
    save(out, output_dir, "01_global_preferred.csv")

    lag_terms = [c for c in ["loss_gdp_winsor", "loss_lag1", "loss_lag2", "loss_lag3", "loss_lag4", "loss_lag5"] if c in df]
    if len(lag_terms) > 1:
        lag_table, lag_cov, lag_cols, _ = fit_fe_cluster(df, "gini_change_annualized", lag_terms, ["country", "year"], "country", controls)
        lag_out = focal(lag_table, lag_terms)
        lag_out.insert(0, "model", "Distributed-lag global two-way fixed effects")
        save(lag_out, output_dir, "01_global_distributed_lags.csv")
        joint = pd.DataFrame([joint_wald(lag_table, lag_cov, lag_cols, lag_terms)])
        joint["test"] = "All contemporaneous and lagged loss terms jointly zero"
        save(joint, output_dir, "01_global_distributed_lag_joint_test.csv")
    else:
        warnings.warn("Lag columns unavailable: distributed-lag model skipped.")


# ---------- 2. China city income analysis ------------------------------------
def china_income_analysis(input_dir, output_dir):
    df = read_data(input_dir, "china_city_panel.csv", ["city", "year", "ln_urban_rural_ratio", "exposure_z", "post2020"])
    df = add_interaction(df, "exposure_z", "post2020", "exposure_x_post")
    specifications = [("baseline", [], None)]
    if "covid_cases_per_100k" in df:
        specifications.append(("covid_adjusted", ["covid_cases_per_100k"], None))
    specifications.append(("city_linear_trends", [], "city"))
    if "province" in df:
        df["province_year"] = df["province"].astype(str) + "_" + df["year"].astype(str)
        specifications.append(("province_by_year_fixed_effects", [], None))
    results = []
    for label, controls, trend in specifications:
        fe = ["city", "year"] if label != "province_by_year_fixed_effects" else ["city", "province_year"]
        table, _, _, _ = fit_fe_cluster(df, "ln_urban_rural_ratio", ["exposure_x_post"], fe, "city", controls, trend)
        row = focal(table, ["exposure_x_post"])
        row.insert(0, "model", label)
        results.append(row)
    save(pd.concat(results, ignore_index=True), output_dir, "02_china_income_specifications.csv")

    event_df, terms = event_terms(df, "exposure_z", "year", 2019)
    event_table, event_cov, event_cols, _ = fit_fe_cluster(event_df, "ln_urban_rural_ratio", terms, ["city", "year"], "city")
    event_out = focal(event_table, terms)
    event_out["event_year"] = event_out["term"].str.extract(r"(\d{4})").astype(int)
    save(event_out, output_dir, "02_china_income_event_study.csv")
    pre = [t for t in terms if int(t[-4:]) < 2019]
    test = pd.DataFrame([joint_wald(event_table, event_cov, event_cols, pre)])
    test["test"] = "All 2013–2018 continuous-exposure event coefficients jointly zero"
    save(test, output_dir, "02_china_income_pretrend_test.csv")


# ---------- 3. Nighttime-light sensitivity -----------------------------------
def nighttime_light_analysis(input_dir, output_dir):
    df = read_data(input_dir, "china_ntl_panel.csv", ["city", "year", "log_ntl", "exposure_z", "post2020"])
    df = add_interaction(df, "exposure_z", "post2020", "exposure_x_post")
    rows = []
    for label, trend in [("baseline", None), ("city_linear_trends", "city")]:
        table, _, _, _ = fit_fe_cluster(df, "log_ntl", ["exposure_x_post"], ["city", "year"], "city", trend_by=trend)
        result = focal(table, ["exposure_x_post"])
        result.insert(0, "model", label)
        rows.append(result)
    save(pd.concat(rows, ignore_index=True), output_dir, "03_nighttime_light_specifications.csv")


# ---------- 4. Household expenditure reallocation ---------------------------
def household_analysis(input_dir, output_dir):
    required = ["household_id", "city", "year", "ln_education_pc", "ln_medical_pc", "exposure_z", "post2020"]
    df = read_data(input_dir, "cfps_household_panel.csv", required)
    df["education_medical_balance"] = pd.to_numeric(df["ln_education_pc"], errors="coerce") - pd.to_numeric(df["ln_medical_pc"], errors="coerce")
    df = add_interaction(df, "exposure_z", "post2020", "exposure_x_post")
    specs = [("baseline_household_and_year_FE", "exposure_x_post", [], ["household_id", "year"])]
    if "covid_cases_per_100k" in df:
        specs.append(("covid_adjusted", "exposure_x_post", ["covid_cases_per_100k"], ["household_id", "year"]))
    if "province" in df:
        df["province_year"] = df["province"].astype(str) + "_" + df["year"].astype(str)
        specs.append(("province_by_year_fixed_effects", "exposure_x_post", [], ["household_id", "province_year"]))
    for component, name in [("precip_z", "precipitation_anomaly"), ("runoff_z", "runoff_anomaly")]:
        if component in df:
            term = f"{component}_x_post"
            df = add_interaction(df, component, "post2020", term)
            specs.append((name, term, [], ["household_id", "year"]))
    rows = []
    for label, term, controls, fixed_effects in specs:
        table, _, _, _ = fit_fe_cluster(df, "education_medical_balance", [term], fixed_effects, "city", controls)
        result = focal(table, [term])
        result.insert(0, "model", label)
        rows.append(result)
    save(pd.concat(rows, ignore_index=True), output_dir, "04_household_balance_specifications.csv")

    event_df, terms = event_terms(df, "exposure_z", "year", 2018)
    event_table, event_cov, event_cols, _ = fit_fe_cluster(event_df, "education_medical_balance", terms, ["household_id", "year"], "city")
    event_out = focal(event_table, terms)
    event_out["event_year"] = event_out["term"].str.extract(r"(\d{4})").astype(int)
    save(event_out, output_dir, "04_household_balance_event_study.csv")
    pre = [t for t in terms if int(t[-4:]) < 2018]
    test = pd.DataFrame([joint_wald(event_table, event_cov, event_cols, pre)])
    test["test"] = "All pre-2018 household event coefficients jointly zero"
    save(test, output_dir, "04_household_balance_pretrend_test.csv")

    outcome_rows = []
    for outcome in ["ln_education_pc", "ln_medical_pc", "education_share", "medical_share", "essential_share"]:
        if outcome in df:
            table, _, _, _ = fit_fe_cluster(df, outcome, ["exposure_x_post"], ["household_id", "year"], "city")
            result = focal(table, ["exposure_x_post"])
            result.insert(0, "outcome", outcome)
            outcome_rows.append(result)
    if outcome_rows:
        save(pd.concat(outcome_rows, ignore_index=True), output_dir, "04_household_alternative_outcomes.csv")


# ---------- 5. Fixed-budget allocation ---------------------------------------
def allocation_analysis(input_dir, output_dir):
    df = read_data(input_dir, "cfps_policy_panel.csv", ["household_id", "year", "income_pc", "education_pc", "medical_pc"])
    panel = df[df["year"].isin([2020, 2022])].copy()
    wide = panel.pivot_table(index="household_id", columns="year", values=["income_pc", "education_pc", "medical_pc"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    need = ["income_pc_2020", "education_pc_2020", "education_pc_2022", "medical_pc_2022"]
    wide = wide.dropna(subset=need).copy()
    wide["income_quintile"] = pd.qcut(wide["income_pc_2020"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    wide["shortfall"] = (wide["education_pc_2020"] - wide["education_pc_2022"]).clip(lower=0)
    rows = []
    for budget_share in [0.05, 0.10, 0.15, 0.20, 0.25]:
        budget = budget_share * wide["shortfall"].sum()
        low = wide["income_quintile"].eq(1)
        transfer_rules = {
            "Universal allowance": pd.Series(budget / len(wide), index=wide.index),
            "Targeted equal allowance": pd.Series(0.0, index=wide.index),
            "Targeted shortfall protection": pd.Series(0.0, index=wide.index),
        }
        transfer_rules["Targeted equal allowance"].loc[low] = budget / low.sum()
        low_shortfall = wide.loc[low, "shortfall"].sum()
        if low_shortfall > 0:
            transfer_rules["Targeted shortfall protection"].loc[low] = budget * wide.loc[low, "shortfall"] / low_shortfall
        for scenario, transfer in transfer_rules.items():
            temp = wide.assign(transfer=transfer)
            for quintile, group in temp.groupby("income_quintile"):
                shortfall = group["shortfall"].sum()
                rows.append({"budget_share_pct": budget_share * 100, "scenario": scenario, "income_quintile": quintile,
                             "households": len(group), "budget_rmb": group["transfer"].sum(), "shortfall_rmb": shortfall,
                             "shortfall_coverage_pct": 100 * group["transfer"].sum() / shortfall if shortfall else 0})
    save(pd.DataFrame(rows), output_dir, "05_budget_neutral_allocation_sensitivity.csv")


# ---------- 6. Compact diagnostic figure -------------------------------------
def generate_diagnostic_figure(output_dir):
    output_dir = Path(output_dir)
    files = [p for p in output_dir.glob("*.csv") if "specifications" in p.name or "preferred" in p.name]
    if not files:
        warnings.warn("No result tables available for diagnostic figure.")
        return
    frames = []
    for path in files:
        d = pd.read_csv(path)
        if "beta" in d:
            d["source"] = path.stem
            frames.append(d)
    results = pd.concat(frames, ignore_index=True)
    results = results[np.isfinite(results["beta"])].head(12)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(results))))
    y = np.arange(len(results))
    ax.errorbar(results["beta"], y, xerr=[results["beta"]-results["ci_low"], results["ci_high"]-results["beta"]], fmt="o", color="#b44d57", ecolor="#597a9f", capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    labels = [f"{r.get('model', r['source'])}: {r['term']}" for _, r in results.iterrows()]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Estimated coefficient (95% confidence interval)")
    ax.set_title("Replication diagnostic: focal estimates")
    fig.tight_layout()
    target = output_dir / "06_replication_diagnostic.png"
    fig.savefig(target, dpi=300)
    plt.close(fig)
    print(f"Wrote {target}")


def main():
    parser = argparse.ArgumentParser(description="Single-file replication workflow.")
    parser.add_argument("--analysis", choices=["all", "global", "china", "nighttime_lights", "household", "allocation", "figures"], default="all")
    parser.add_argument("--input", default="data")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    functions = {"global": global_analysis, "china": china_income_analysis, "nighttime_lights": nighttime_light_analysis,
                 "household": household_analysis, "allocation": allocation_analysis}
    if args.analysis == "all":
        for fn in functions.values():
            fn(args.input, args.output)
        generate_diagnostic_figure(args.output)
    elif args.analysis == "figures":
        generate_diagnostic_figure(args.output)
    else:
        functions[args.analysis](args.input, args.output)


if __name__ == "__main__":
    main()
