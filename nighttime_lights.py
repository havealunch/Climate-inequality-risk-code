from code.common import read_csv_required, interaction, fe_cluster_ols, write_result


def run(input_dir, output_dir):
    df = read_csv_required(
        f"{input_dir}/china_ntl_panel.csv", ["city", "year", "log_ntl", "exposure_z", "post2020"]
    )
    df = interaction(df, "exposure_z", "post2020")
    result, _ = fe_cluster_ols(df, "log_ntl", "exposure_x_post", ["city", "year"], "city")
    result.insert(0, "model", "Nighttime-light city and year fixed effects")
    result.insert(1, "outcome", "log_mean_radiance")
    return write_result(result, output_dir, "nighttime_light_results.csv")
