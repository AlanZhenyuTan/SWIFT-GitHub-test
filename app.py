from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import tco_model as model


st.set_page_config(
    page_title="Truck TCO and LCE Analysis",
    page_icon="🚛",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        padding: 12px 14px;
        border-radius: 14px;
    }
    .small-note {color:#64748b; font-size:0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Formatting and display helpers
# -----------------------------------------------------------------------------
def fmt_gbp(value: float, decimals: int = 0) -> str:
    return f"£{value:,.{decimals}f}"


def fmt_num(value: Any, decimals: int = 4) -> Any:
    if isinstance(value, (float, np.floating)):
        return round(float(value), decimals)
    return value


def show_figure(fig, *, use_container_width: bool = True) -> None:
    if fig is None:
        st.warning("The plotting function did not return a Matplotlib figure.")
        return
    st.pyplot(fig, use_container_width=use_container_width)
    plt.close(fig)


def dataclass_input_table(shared, diesel, betc, bets) -> pd.DataFrame:
    rows = []
    for group_name, obj in (
        ("SharedInputs", shared),
        ("DieselInputs", diesel),
        ("BETCInputs", betc),
        ("BETSInputs", bets),
    ):
        for key, value in asdict(obj).items():
            rows.append(
                {
                    "Parameter": key,
                    "Label": model.get_pretty_label(key),
                    "Value": fmt_num(value),
                }
            )
    return pd.DataFrame(rows)


def uncertainty_specs_table(specs: list[dict]) -> pd.DataFrame:
    """Convert triangular uncertainty specifications to an editable table."""
    rows = []
    for spec in specs:
        rows.append(
            {
                "Variable": spec["variable"],
                "Label": model.get_pretty_label(spec["variable"]),
                "Min": float(spec["left"]),
                "Mode": float(spec["mode"]),
                "Max": float(spec["right"]),
            }
        )
    return pd.DataFrame(rows)


def table_to_uncertainty_specs(
    edited_df: pd.DataFrame,
    default_specs: list[dict],
) -> tuple[list[dict], list[str]]:
    """Return validated specs; invalid rows fall back to their defaults."""
    defaults = {spec["variable"]: deepcopy(spec) for spec in default_specs}
    specs = []
    warnings = []

    for _, row in edited_df.iterrows():
        variable = str(row["Variable"])
        if variable not in defaults:
            continue
        spec = deepcopy(defaults[variable])
        try:
            left = float(row["Min"])
            mode = float(row["Mode"])
            right = float(row["Max"])
            if not (np.isfinite(left) and np.isfinite(mode) and np.isfinite(right)):
                raise ValueError
            if left > mode or mode > right:
                raise ValueError
            spec.update({"left": left, "mode": mode, "right": right})
        except (TypeError, ValueError):
            warnings.append(
                f"{model.get_pretty_label(variable)} must satisfy Min ≤ Mode ≤ Max; defaults are being used."
            )
        specs.append(spec)

    # Preserve any default rows accidentally removed by the editor.
    existing = {spec["variable"] for spec in specs}
    specs.extend(deepcopy(spec) for name, spec in defaults.items() if name not in existing)
    return specs, warnings


def lce_mc_default_range_table(shared) -> pd.DataFrame:
    """Editable LCE Monte Carlo ranges.

    Relative rows are expressed as decimal changes: -0.20 means -20%.
    """
    return pd.DataFrame(
        [
            {
                "Analysis": "curb/glider weight",
                "Range type": "Relative change",
                "Min": -0.20,
                "Mode": 0.00,
                "Max": 0.20,
            },
            {
                "Analysis": "Full-loaded VKT per day",
                "Range type": "Absolute value",
                "Min": float(shared.full_loaded_km_per_day) * 0.50,
                "Mode": float(shared.full_loaded_km_per_day),
                "Max": float(shared.full_loaded_km_per_day) * 1.80,
            },
            {
                "Analysis": "curb/glider production emission factor",
                "Range type": "Absolute value",
                "Min": 6.59,
                "Mode": 7.00,
                "Max": 8.00,
            },
            {
                "Analysis": "battery production emission factor",
                "Range type": "Absolute value",
                "Min": 40.00,
                "Mode": 57.00,
                "Max": 69.00,
            },
        ]
    )


def validate_lce_mc_range_table(edited_df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    rows = []
    warnings = []
    for _, row in edited_df.iterrows():
        name = str(row["Analysis"])
        try:
            left = float(row["Min"])
            mode = float(row["Mode"])
            right = float(row["Max"])
            if not (np.isfinite(left) and np.isfinite(mode) and np.isfinite(right)):
                raise ValueError
            if left > mode or mode > right:
                raise ValueError
        except (TypeError, ValueError):
            warnings.append(f"{name} must satisfy Min ≤ Mode ≤ Max.")
            continue
        rows.append(
            {
                "name": name,
                "range_type": str(row["Range type"]),
                "left": left,
                "mode": mode,
                "right": right,
            }
        )
    return rows, warnings

def tco_component_table(results: dict) -> pd.DataFrame:
    keys = [
        "truck_acquisition_cost_npv",
        "truck_residual_value_npv",
        "fixed_operating_cost_npv",
        "depot_infrastructure_cost_npv",
        "energy_cost_npv",
        "energy_service_total_cost_npv",
        "tco_discounted",
        "tco_per_km_discounted",
    ]
    rows = []
    for vehicle_key, vehicle_label in (
        ("diesel", "Diesel"),
        ("bet_c", "BET-C"),
        ("bet_s", "BET-S"),
    ):
        result = results[vehicle_key]
        row = {"Vehicle": vehicle_label}
        for key in keys:
            if key in result:
                row[model.get_pretty_label(key)] = result[key]
        rows.append(row)
    return pd.DataFrame(rows)


def lce_summary_table(lce_results: dict) -> pd.DataFrame:
    rows = []
    for key, label in (("diesel", "Diesel"), ("bet_c", "BET-C"), ("bet_s", "BET-S")):
        result = lce_results[key]
        rows.append(
            {
                "Vehicle": label,
                "Payload (tonnes)": result["payload_tonnes"],
                "Life-cycle emissions (kg CO2e)": result["life_cycle_emissions"],
                "kg CO2e/km": result["life_cycle_emissions_per_km"],
                "kg CO2e/tonne-km": result["life_cycle_emissions_per_tonne_km"],
                "Second-life kg CO2e/km": result.get(
                    "life_cycle_emission_per_km_including_second_life_credit", np.nan
                ),
                "Second-life kg CO2e/tonne-km": result.get(
                    "life_cycle_emission_per_tonne_km_including_second_life_credit", np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def adjusted_sensitivity_specs(shared, diesel, betc, bets):
    """Use the current Streamlit baseline for percentage-change sensitivities."""
    specs = deepcopy(model.sensitivity_specs)
    targets = {
        "shared": shared,
        "diesel": diesel,
        "betc": betc,
        "bets": bets,
    }
    for spec in specs:
        if "direct_values" in spec:
            continue
        variable = spec["variable_name"]
        target_class = spec["target_class"]
        target_names = [target_class] if isinstance(target_class, str) else target_class
        if variable == "diesel_price_multiplier":
            spec["base_value"] = 1.0
            continue
        for target_name in target_names:
            obj = targets[target_name]
            if hasattr(obj, variable):
                spec["base_value"] = getattr(obj, variable)
                break
    return specs




def _append_tco_outputs(row: dict, results: dict) -> dict:
    diesel_result = results["diesel"]
    betc_result = results["bet_c"]
    bets_result = results["bet_s"]

    diesel_tco = diesel_result["tco_discounted"]
    betc_tco = betc_result["tco_discounted"]
    bets_tco = bets_result["tco_discounted"]
    diesel_per_km = diesel_result["tco_per_km_discounted"]
    betc_per_km = betc_result["tco_per_km_discounted"]
    bets_per_km = bets_result["tco_per_km_discounted"]

    row.update(
        {
            "diesel_tco": diesel_tco,
            "bet_c_tco": betc_tco,
            "bet_s_tco": bets_tco,
            "gap_bet_c_diesel": betc_tco - diesel_tco,
            "gap_bet_s_diesel": bets_tco - diesel_tco,
            "gap_bet_s_bet_c": bets_tco - betc_tco,
            "diesel_tco_per_km": diesel_per_km,
            "bet_c_tco_per_km": betc_per_km,
            "bet_s_tco_per_km": bets_per_km,
            "gap_bet_c_diesel_per_km": betc_per_km - diesel_per_km,
            "gap_bet_s_diesel_per_km": bets_per_km - diesel_per_km,
            "gap_bet_s_bet_c_per_km": bets_per_km - betc_per_km,
        }
    )
    return row


def run_tco_monte_carlo_custom(
    shared,
    diesel,
    betc,
    bets,
    specs: list[dict],
    n_runs: int,
    random_seed: int,
) -> pd.DataFrame:
    """Run full TCO Monte Carlo using the sidebar triangular ranges."""
    frames = []
    for scenario_index, include_subsidy in enumerate((True, False)):
        rng = np.random.default_rng(random_seed + scenario_index * 100_000)
        scenario = model.subsidy_scenario_label(include_subsidy)
        active_specs = [
            spec
            for spec in specs
            if include_subsidy or spec["variable"] != "bet_subsidy"
        ]
        rows = []

        for iteration in range(n_runs):
            shared_i = shared
            diesel_i = diesel
            betc_i = betc
            bets_i = bets
            sampled_inputs = {}

            if not include_subsidy:
                shared_i = replace(shared_i, bet_subsidy=0.0)

            for spec in active_specs:
                sampled = float(
                    rng.triangular(spec["left"], spec["mode"], spec["right"])
                )
                sampled_inputs[spec["variable"]] = sampled
                shared_i, diesel_i, betc_i, bets_i = model.apply_single_variable_change(
                    shared_i,
                    diesel_i,
                    betc_i,
                    bets_i,
                    spec,
                    sampled,
                )

            if not include_subsidy:
                sampled_inputs["bet_subsidy"] = 0.0

            results_i = model.run_model(
                shared=shared_i,
                diesel_inp=diesel_i,
                betc_inp=betc_i,
                bets_inp=bets_i,
            )
            row = {
                "subsidy_scenario": scenario,
                "iteration": iteration + 1,
                **sampled_inputs,
                "diesel_depot_price_per_l": shared_i.diesel_depot_price_per_l,
                "diesel_public_price_per_l": shared_i.diesel_public_price_per_l,
            }
            rows.append(_append_tco_outputs(row, results_i))
        frames.append(pd.DataFrame(rows))

    return pd.concat(frames, ignore_index=True)


def run_independent_tco_monte_carlo_custom(
    shared,
    diesel,
    betc,
    bets,
    specs: list[dict],
    n_runs: int,
    random_seed: int,
) -> pd.DataFrame:
    """Vary one sidebar uncertainty at a time for both subsidy scenarios."""
    rows = []
    for scenario_index, include_subsidy in enumerate((True, False)):
        scenario = model.subsidy_scenario_label(include_subsidy)
        active_specs = [
            spec
            for spec in specs
            if include_subsidy or spec["variable"] != "bet_subsidy"
        ]
        for spec_index, spec in enumerate(active_specs):
            rng = np.random.default_rng(
                random_seed + scenario_index * 100_000 + spec_index * 1_000
            )
            for iteration in range(n_runs):
                sampled = float(
                    rng.triangular(spec["left"], spec["mode"], spec["right"])
                )
                shared_i = replace(shared, bet_subsidy=0.0) if not include_subsidy else shared
                diesel_i = diesel
                betc_i = betc
                bets_i = bets
                shared_i, diesel_i, betc_i, bets_i = model.apply_single_variable_change(
                    shared_i,
                    diesel_i,
                    betc_i,
                    bets_i,
                    spec,
                    sampled,
                )
                results_i = model.run_model(
                    shared=shared_i,
                    diesel_inp=diesel_i,
                    betc_inp=betc_i,
                    bets_inp=bets_i,
                )
                row = {
                    "subsidy_scenario": scenario,
                    "variable": spec["variable"],
                    "sampled_value": sampled,
                    "iteration": iteration + 1,
                }
                rows.append(_append_tco_outputs(row, results_i))
    return pd.DataFrame(rows)


def _projected_spec(spec: dict, default_spec: dict, shared, diesel, betc, bets) -> dict:
    """Scale an edited absolute range with the projected-year baseline."""
    variable = spec["variable"]
    if variable in {"diesel_price_multiplier", "bet_subsidy"}:
        return deepcopy(spec)

    target_names = spec["target_class"]
    if isinstance(target_names, str):
        target_names = [target_names]
    targets = {"shared": shared, "diesel": diesel, "betc": betc, "bets": bets}

    projected_mode = None
    for target_name in target_names:
        obj = targets[target_name]
        if hasattr(obj, variable):
            projected_mode = float(getattr(obj, variable))
            break
    if projected_mode is None:
        return deepcopy(spec)

    default_mode = float(default_spec["mode"])
    scale = projected_mode / default_mode if default_mode != 0 else 1.0
    projected = deepcopy(spec)
    projected["left"] = float(spec["left"]) * scale
    projected["mode"] = float(spec["mode"]) * scale
    projected["right"] = float(spec["right"]) * scale
    return projected


def run_projection_monte_carlo_custom(
    start_year: int,
    end_year: int,
    n_runs: int,
    random_seed: int,
    specs: list[dict],
    shared,
    diesel,
    betc,
    bets,
) -> pd.DataFrame:
    """Projection Monte Carlo using the sidebar ranges and current baselines."""
    default_map = {
        spec["variable"]: spec
        for spec in model.get_uncertainty_specs(include_subsidy_uncertainty=True)
    }
    rows = []
    for scenario_index, include_subsidy in enumerate((True, False)):
        scenario = model.subsidy_scenario_label(include_subsidy)
        rng = np.random.default_rng(random_seed + scenario_index * 100_000)

        for year in range(start_year, end_year + 1):
            shared_base, diesel_base, betc_base, bets_base = model.build_projected_inputs_for_year(
                target_year=year,
                base_year=start_year,
                shared=shared,
                diesel_inp=diesel,
                betc_inp=betc,
                bets_inp=bets,
            )
            active_specs = [
                _projected_spec(
                    spec,
                    default_map.get(spec["variable"], spec),
                    shared_base,
                    diesel_base,
                    betc_base,
                    bets_base,
                )
                for spec in specs
                if include_subsidy or spec["variable"] != "bet_subsidy"
            ]

            for iteration in range(n_runs):
                shared_i = shared_base
                diesel_i = diesel_base
                betc_i = betc_base
                bets_i = bets_base
                if not include_subsidy:
                    shared_i = replace(shared_i, bet_subsidy=0.0)

                for spec in active_specs:
                    sampled = float(
                        rng.triangular(spec["left"], spec["mode"], spec["right"])
                    )
                    shared_i, diesel_i, betc_i, bets_i = model.apply_single_variable_change(
                        shared_i,
                        diesel_i,
                        betc_i,
                        bets_i,
                        spec,
                        sampled,
                    )

                results_i = model.run_model(
                    shared=shared_i,
                    diesel_inp=diesel_i,
                    betc_inp=betc_i,
                    bets_inp=bets_i,
                )
                rows.append(
                    {
                        "subsidy_scenario": scenario,
                        "year": year,
                        "iteration": iteration + 1,
                        "diesel_tco_discounted": results_i["diesel"]["tco_discounted"],
                        "betc_tco_discounted": results_i["bet_c"]["tco_discounted"],
                        "bets_tco_discounted": results_i["bet_s"]["tco_discounted"],
                        "diesel_tco_per_km": results_i["diesel"]["tco_per_km_discounted"],
                        "betc_tco_per_km": results_i["bet_c"]["tco_per_km_discounted"],
                        "bets_tco_per_km": results_i["bet_s"]["tco_per_km_discounted"],
                    }
                )
    return pd.DataFrame(rows)


def plot_monte_carlo_histogram_grid(df: pd.DataFrame, per_km: bool = False):
    """Return all six Monte Carlo distributions in one reliable Figure."""
    if per_km:
        specs = [
            ("diesel_tco_per_km", "Diesel TCO per km", "£/km"),
            ("bet_c_tco_per_km", "BET-C TCO per km", "£/km"),
            ("bet_s_tco_per_km", "BET-S TCO per km", "£/km"),
            ("gap_bet_c_diesel_per_km", "BET-C − Diesel per km", "£/km"),
            ("gap_bet_s_diesel_per_km", "BET-S − Diesel per km", "£/km"),
            ("gap_bet_s_bet_c_per_km", "BET-S − BET-C per km", "£/km"),
        ]
    else:
        specs = [
            ("diesel_tco", "Diesel discounted TCO", "£"),
            ("bet_c_tco", "BET-C discounted TCO", "£"),
            ("bet_s_tco", "BET-S discounted TCO", "£"),
            ("gap_bet_c_diesel", "BET-C − Diesel", "£"),
            ("gap_bet_s_diesel", "BET-S − Diesel", "£"),
            ("gap_bet_s_bet_c", "BET-S − BET-C", "£"),
        ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (column, title, xlabel) in zip(axes.flat, specs):
        for scenario, sub_df in df.groupby("subsidy_scenario"):
            values = sub_df[column].dropna()
            ax.hist(values, bins=20, alpha=0.45, label=str(scenario))
            mean_value = values.mean()
            ax.axvline(mean_value, linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_projection_summary(summary_df: pd.DataFrame, per_km: bool = False):
    """Overlay both subsidy scenarios in one projection Figure."""
    if per_km:
        vehicle_specs = [
            ("diesel_tco_per_km", "Diesel"),
            ("betc_tco_per_km", "BET-C"),
            ("bets_tco_per_km", "BET-S"),
        ]
        ylabel = "Discounted TCO (£/km)"
        title = "Projected discounted TCO per km"
    else:
        vehicle_specs = [
            ("diesel_tco_discounted", "Diesel"),
            ("betc_tco_discounted", "BET-C"),
            ("bets_tco_discounted", "BET-S"),
        ]
        ylabel = "Discounted TCO (£)"
        title = "Projected discounted TCO"

    fig, ax = plt.subplots(figsize=(10, 6))
    linestyles = ["-", "--", ":", "-."]
    for scenario_index, (scenario, scenario_df) in enumerate(
        summary_df.groupby("subsidy_scenario")
    ):
        linestyle = linestyles[scenario_index % len(linestyles)]
        ordered = scenario_df.sort_values("year")
        for metric, vehicle_label in vehicle_specs:
            line = ax.plot(
                ordered["year"],
                ordered[f"{metric}_p50"],
                marker="o",
                linestyle=linestyle,
                label=f"{vehicle_label} — {scenario}",
            )[0]
            ax.fill_between(
                ordered["year"],
                ordered[f"{metric}_p5"],
                ordered[f"{metric}_p95"],
                alpha=0.10,
                color=line.get_color(),
            )
    ax.set_title(title)
    ax.set_xlabel("Purchase year")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def build_lce_mc_specs(shared, lce_settings: dict, range_rows: list[dict]) -> list[dict]:
    ranges = {row["name"]: row for row in range_rows}

    def values(name):
        row = ranges[name]
        return row["left"], row["mode"], row["right"]

    weight_low, weight_mode, weight_high = values("curb/glider weight")
    vkt_low, vkt_mode, vkt_high = values("Full-loaded VKT per day")
    prod_low, prod_mode, prod_high = values(
        "curb/glider production emission factor"
    )
    battery_low, battery_mode, battery_high = values(
        "battery production emission factor"
    )

    return [
        {
            "name": "curb/glider weight",
            "distribution": "triangular",
            "vehicle_values": {
                "diesel": {
                    "parameter": "truck_curb_weight_kg",
                    "base_value": lce_settings["diesel"]["truck_curb_weight_kg"],
                },
                "bet_c": {
                    "parameter": "glider_weight_kg",
                    "base_value": lce_settings["bet_c"]["glider_weight_kg"],
                },
                "bet_s": {
                    "parameter": "glider_weight_kg",
                    "base_value": lce_settings["bet_s"]["glider_weight_kg"],
                },
            },
            "shared_sample_groups": [["bet_c", "bet_s"]],
            "relative_low": weight_low,
            "relative_mode": weight_mode,
            "relative_high": weight_high,
        },
        {
            "name": "Full-loaded VKT per day",
            "distribution": "triangular",
            "input_map": {"shared": "full_loaded_km_per_day"},
            "low": vkt_low,
            "mode": vkt_mode,
            "high": vkt_high,
        },
        {
            "name": "curb/glider production emission factor",
            "distribution": "triangular",
            "variable_map": {
                "diesel": "truck_production_emission_factor_kg_per_kg",
                "bet_c": "glider_production_emission_factor_kg_per_kg",
                "bet_s": "glider_production_emission_factor_kg_per_kg",
            },
            "low": prod_low,
            "mode": prod_mode,
            "high": prod_high,
        },
        {
            "name": "battery production emission factor",
            "distribution": "triangular",
            "variable_map": {
                "bet_c": "battery_production_emission_factor_kg_per_kwh",
                "bet_s": "battery_production_emission_factor_kg_per_kwh",
            },
            "low": battery_low,
            "mode": battery_mode,
            "high": battery_high,
        },
    ]


def _run_lce_mc_analysis_with_settings(
    specs,
    n_runs,
    random_seed,
    shared,
    diesel,
    betc,
    bets,
    lce_settings,
    analysis_name,
):
    rng = np.random.default_rng(random_seed)
    sampled_values = {
        spec["name"]: model._draw_lce_mc_samples(spec, rng, n_runs)
        for spec in specs
    }
    rows = []

    for run_id in range(n_runs):
        shared_i, diesel_i, betc_i, bets_i = shared, diesel, betc, bets
        diesel_kwargs = deepcopy(lce_settings["diesel"])
        betc_kwargs = deepcopy(lce_settings["bet_c"])
        bets_kwargs = deepcopy(lce_settings["bet_s"])

        for spec in specs:
            sampled = sampled_values[spec["name"]][run_id]
            shared_i, diesel_i, betc_i, bets_i = model._apply_lce_uncertainty_spec(
                spec,
                sampled,
                shared_i,
                diesel_i,
                betc_i,
                bets_i,
                diesel_kwargs,
                betc_kwargs,
                bets_kwargs,
            )

        lce_result = model.run_lce_model(
            shared=shared_i,
            diesel_inp=diesel_i,
            betc_inp=betc_i,
            bets_inp=bets_i,
            diesel_lce_kwargs=diesel_kwargs,
            betc_lce_kwargs=betc_kwargs,
            bets_lce_kwargs=bets_kwargs,
        )
        rows.append(model._flatten_lce_results_for_mc(analysis_name, run_id, lce_result))
    return pd.DataFrame(rows)


def run_lce_monte_carlo_custom(
    specs,
    n_runs,
    random_seed,
    shared,
    diesel,
    betc,
    bets,
    lce_settings,
):
    single_frames = []
    for index, spec in enumerate(specs):
        single_frames.append(
            _run_lce_mc_analysis_with_settings(
                [spec],
                n_runs,
                random_seed + index,
                shared,
                diesel,
                betc,
                bets,
                lce_settings,
                spec["name"],
            )
        )
    single_df = pd.concat(single_frames, ignore_index=True)
    total_df = _run_lce_mc_analysis_with_settings(
        specs,
        n_runs,
        random_seed + 10_000,
        shared,
        diesel,
        betc,
        bets,
        lce_settings,
        "Total LCE uncertainty",
    )
    return single_df, total_df


def run_lce_sensitivity_custom(
    shared,
    diesel,
    betc,
    bets,
    lce_settings,
):
    """
    Run deterministic LCE sensitivity analyses using the same settings as
    tco_model.default_lce_sensitivity_specs().

    Interpretation:
    1. Curb/glider weight:
       relative changes around vehicle-specific base values.
    2. Full-loaded VKT per day:
       relative changes around a fixed base value of 240 km/day.
    3. Curb/glider production emission factor:
       absolute parameter values.
    4. Battery production emission factor:
       absolute parameter values.
    """

    specs = [
        {
            "name": "curb/glider weight",
            "vehicle_values": {
                "diesel": {
                    "parameter": "truck_curb_weight_kg",
                    "base_value": 11022.0,
                },
                "bet_c": {
                    "parameter": "glider_weight_kg",
                    "base_value": 9986.0,
                },
                "bet_s": {
                    "parameter": "glider_weight_kg",
                    "base_value": 9986.0,
                },
            },
            "changes": [-0.20, 0.00, 0.20],
        },
        {
            "name": "Full-loaded VKT per day",
            "input_map": {
                "shared": "full_loaded_km_per_day",
            },
            "base_value": 240.0,
            "changes": [-0.50, -0.20, 0.00, 0.40, 0.80],
        },
        {
            "name": "curb/glider production emission factor",
            "variable_map": {
                "diesel": "truck_production_emission_factor_kg_per_kg",
                "bet_c": "glider_production_emission_factor_kg_per_kg",
                "bet_s": "glider_production_emission_factor_kg_per_kg",
            },
            "values": [6.59, 7.00, 8.00],
        },
        {
            "name": "battery production emission factor",
            "variable_map": {
                "bet_c": "battery_production_emission_factor_kg_per_kwh",
                "bet_s": "battery_production_emission_factor_kg_per_kwh",
            },
            "values": [40.0, 57.0, 69.0],
        },
    ]

    def format_change_label(change: float) -> str:
        """Format relative changes as -20%, 0%, +20%, etc."""
        if np.isclose(change, 0.0):
            return "0%"
        return f"{change:+.0%}"

    def format_absolute_label(value: float) -> str:
        """Format absolute sensitivity values without unnecessary decimals."""
        return f"{value:g}"

    all_results = {}

    for spec in specs:
        # Relative-change sensitivity
        if "changes" in spec:
            scenarios = [float(value) for value in spec["changes"]]
            x_labels = [
                format_change_label(value)
                for value in scenarios
            ]

        # Absolute-value sensitivity
        elif "values" in spec:
            scenarios = [float(value) for value in spec["values"]]
            x_labels = [
                format_absolute_label(value)
                for value in scenarios
            ]

        else:
            raise ValueError(
                f"Sensitivity specification '{spec['name']}' must contain "
                "'changes' or 'values'."
            )

        output = {
            "x_labels": x_labels,
            "diesel": [],
            "bet_c": [],
            "bet_s": [],
            "bet_c_vs_diesel": [],
            "bet_s_vs_diesel": [],
            "bet_s_vs_bet_c": [],
            "diesel_per_km": [],
            "bet_c_per_km": [],
            "bet_s_per_km": [],
            "bet_c_vs_diesel_per_km": [],
            "bet_s_vs_diesel_per_km": [],
            "bet_s_vs_bet_c_per_km": [],
        }

        for scenario in scenarios:
            shared_i = shared
            diesel_i = diesel
            betc_i = betc
            bets_i = bets

            diesel_kwargs = deepcopy(lce_settings["diesel"])
            betc_kwargs = deepcopy(lce_settings["bet_c"])
            bets_kwargs = deepcopy(lce_settings["bet_s"])

            # -------------------------------------------------------------
            # Relative changes to vehicle-specific LCE parameters
            # Example:
            # actual weight = base weight × (1 + relative change)
            # -------------------------------------------------------------
            if "vehicle_values" in spec:
                for vehicle, vehicle_spec in spec["vehicle_values"].items():
                    parameter = vehicle_spec["parameter"]
                    base_value = float(vehicle_spec["base_value"])
                    actual_value = base_value * (1.0 + scenario)

                    if vehicle == "diesel":
                        diesel_kwargs[parameter] = actual_value
                    elif vehicle == "bet_c":
                        betc_kwargs[parameter] = actual_value
                    elif vehicle == "bet_s":
                        bets_kwargs[parameter] = actual_value
                    else:
                        raise ValueError(
                            f"Unknown vehicle '{vehicle}' in "
                            f"sensitivity '{spec['name']}'."
                        )

            # -------------------------------------------------------------
            # Relative changes to dataclass inputs
            # Full-loaded VKT:
            # actual VKT = 240 × (1 + relative change)
            # -------------------------------------------------------------
            elif "input_map" in spec:
                base_value = float(spec["base_value"])
                actual_value = base_value * (1.0 + scenario)

                input_map = spec["input_map"]

                if "shared" in input_map:
                    shared_i = replace(
                        shared_i,
                        **{
                            input_map["shared"]: actual_value,
                        },
                    )

                if "diesel" in input_map:
                    diesel_i = replace(
                        diesel_i,
                        **{
                            input_map["diesel"]: actual_value,
                        },
                    )

                if "bet_c" in input_map:
                    betc_i = replace(
                        betc_i,
                        **{
                            input_map["bet_c"]: actual_value,
                        },
                    )

                if "bet_s" in input_map:
                    bets_i = replace(
                        bets_i,
                        **{
                            input_map["bet_s"]: actual_value,
                        },
                    )

            # -------------------------------------------------------------
            # Absolute values passed directly to compute_lce_* parameters
            # -------------------------------------------------------------
            elif "variable_map" in spec:
                variable_map = spec["variable_map"]
                actual_value = float(scenario)

                if "diesel" in variable_map:
                    diesel_kwargs[
                        variable_map["diesel"]
                    ] = actual_value

                if "bet_c" in variable_map:
                    betc_kwargs[
                        variable_map["bet_c"]
                    ] = actual_value

                if "bet_s" in variable_map:
                    bets_kwargs[
                        variable_map["bet_s"]
                    ] = actual_value

            else:
                raise ValueError(
                    f"Sensitivity specification '{spec['name']}' has no "
                    "vehicle_values, input_map or variable_map."
                )

            lce_result = model.run_lce_model(
                shared=shared_i,
                diesel_inp=diesel_i,
                betc_inp=betc_i,
                bets_inp=bets_i,
                diesel_lce_kwargs=diesel_kwargs,
                betc_lce_kwargs=betc_kwargs,
                bets_lce_kwargs=bets_kwargs,
            )

            metrics = model.extract_lce_metrics(lce_result)

            # Per tonne-km results
            output["diesel"].append(
                metrics["diesel_tonne_km"]
            )
            output["bet_c"].append(
                metrics["bet_c_tonne_km"]
            )
            output["bet_s"].append(
                metrics["bet_s_tonne_km"]
            )
            output["bet_c_vs_diesel"].append(
                metrics["bet_c_vs_diesel_tonne_km"]
            )
            output["bet_s_vs_diesel"].append(
                metrics["bet_s_vs_diesel_tonne_km"]
            )
            output["bet_s_vs_bet_c"].append(
                metrics["bet_s_vs_bet_c_tonne_km"]
            )

            # Per-km results
            output["diesel_per_km"].append(
                metrics["diesel_km"]
            )
            output["bet_c_per_km"].append(
                metrics["bet_c_km"]
            )
            output["bet_s_per_km"].append(
                metrics["bet_s_km"]
            )
            output["bet_c_vs_diesel_per_km"].append(
                metrics["bet_c_vs_diesel_km"]
            )
            output["bet_s_vs_diesel_per_km"].append(
                metrics["bet_s_vs_diesel_km"]
            )
            output["bet_s_vs_bet_c_per_km"].append(
                metrics["bet_s_vs_bet_c_km"]
            )

        all_results[spec["name"]] = output

    return all_results
# -----------------------------------------------------------------------------
# Cached model runs
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_baseline(shared_dict, diesel_dict, betc_dict, bets_dict):
    return model.run_model(
        shared=model.SharedInputs(**shared_dict),
        diesel_inp=model.DieselInputs(**diesel_dict),
        betc_inp=model.BETCInputs(**betc_dict),
        bets_inp=model.BETSInputs(**bets_dict),
    )


@st.cache_data(show_spinner=False)
def cached_tco_sensitivity(shared_dict, diesel_dict, betc_dict, bets_dict):
    shared = model.SharedInputs(**shared_dict)
    diesel = model.DieselInputs(**diesel_dict)
    betc = model.BETCInputs(**betc_dict)
    bets = model.BETSInputs(**bets_dict)
    specs = adjusted_sensitivity_specs(shared, diesel, betc, bets)
    return model.run_multiple_sensitivity_analyses(
        specs,
        shared=shared,
        diesel_inp=diesel,
        betc_inp=betc,
        bets_inp=bets,
    )


@st.cache_data(show_spinner=False)
def cached_heatmaps(shared_dict, diesel_dict, bets_dict):
    shared = model.SharedInputs(**shared_dict)
    diesel = model.DieselInputs(**diesel_dict)
    bets = model.BETSInputs(**bets_dict)
    baas_grid = model.run_baas_viability_grid(shared=shared, bets_inp=bets)
    utilisation_grid = model.run_baas_utilisation_viability_grid(
        shared=shared,
        bets_inp=bets,
        expected_station_utilisations=np.arange(0.20, 0.50, 0.10),
        fixed_swapping_fee=bets.swapping_fee_flat,
    )
    gap_grid = model.run_baas_utilisation_tco_gap_grid(
        shared=shared,
        diesel_inp=diesel,
        bets_inp=bets,
        fixed_swapping_fee=bets.swapping_fee_flat,
    )
    return baas_grid, utilisation_grid, gap_grid


@st.cache_data(show_spinner=False)
def cached_tco_mc(
    shared_dict,
    diesel_dict,
    betc_dict,
    bets_dict,
    n_runs: int,
    random_seed: int,
    uncertainty_specs_json: str,
):
    shared = model.SharedInputs(**shared_dict)
    diesel = model.DieselInputs(**diesel_dict)
    betc = model.BETCInputs(**betc_dict)
    bets = model.BETSInputs(**bets_dict)
    specs = json.loads(uncertainty_specs_json)
    mc_df = run_tco_monte_carlo_custom(
        shared, diesel, betc, bets, specs, n_runs, random_seed
    )
    summary_df, probability_df = model.summarize_monte_carlo_results(mc_df)
    independent_df = run_independent_tco_monte_carlo_custom(
        shared, diesel, betc, bets, specs, n_runs, random_seed
    )
    return mc_df, summary_df, probability_df, independent_df


@st.cache_data(show_spinner=False)
def cached_margin(n_runs: int, random_seed: int):
    raw = model.run_margin_sweep_with_and_without_subsidy_uncertainty(
        margins=np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
        n_runs=n_runs,
        random_seed=random_seed,
    )
    return raw, model.summarize_margin_uncertainty(raw)


@st.cache_data(show_spinner=False)
def cached_projection(
    start_year: int,
    end_year: int,
    n_runs: int,
    random_seed: int,
    uncertainty_specs_json: str,
    shared_dict,
    diesel_dict,
    betc_dict,
    bets_dict,
):
    raw = run_projection_monte_carlo_custom(
        start_year=start_year,
        end_year=end_year,
        n_runs=n_runs,
        random_seed=random_seed,
        specs=json.loads(uncertainty_specs_json),
        shared=model.SharedInputs(**shared_dict),
        diesel=model.DieselInputs(**diesel_dict),
        betc=model.BETCInputs(**betc_dict),
        bets=model.BETSInputs(**bets_dict),
    )
    total_summary = model.summarize_projection_uncertainty(
        raw,
        metric_cols=[
            "diesel_tco_discounted",
            "betc_tco_discounted",
            "bets_tco_discounted",
        ],
    )
    per_km_summary = model.summarize_projection_uncertainty(
        raw,
        metric_cols=["diesel_tco_per_km", "betc_tco_per_km", "bets_tco_per_km"],
    )
    return raw, total_summary, per_km_summary


@st.cache_data(show_spinner=False)
def cached_lce_baseline(
    shared_dict,
    diesel_dict,
    betc_dict,
    bets_dict,
    lce_settings_json: str,
):
    shared = model.SharedInputs(**shared_dict)
    diesel = model.DieselInputs(**diesel_dict)
    betc = model.BETCInputs(**betc_dict)
    bets = model.BETSInputs(**bets_dict)
    settings = json.loads(lce_settings_json)
    return model.run_lce_model(
        shared=shared,
        diesel_inp=diesel,
        betc_inp=betc,
        bets_inp=bets,
        diesel_lce_kwargs=settings["diesel"],
        betc_lce_kwargs=settings["bet_c"],
        bets_lce_kwargs=settings["bet_s"],
    )


@st.cache_data(show_spinner=False)
def cached_lce_sensitivity(
    shared_dict,
    diesel_dict,
    betc_dict,
    bets_dict,
    lce_settings_json: str,
):
    return run_lce_sensitivity_custom(
        model.SharedInputs(**shared_dict),
        model.DieselInputs(**diesel_dict),
        model.BETCInputs(**betc_dict),
        model.BETSInputs(**bets_dict),
        json.loads(lce_settings_json),
    )


@st.cache_data(show_spinner=False)
def cached_lce_mc(
    shared_dict,
    diesel_dict,
    betc_dict,
    bets_dict,
    lce_settings_json: str,
    lce_mc_ranges_json: str,
    n_runs: int,
    random_seed: int,
):
    shared = model.SharedInputs(**shared_dict)
    diesel = model.DieselInputs(**diesel_dict)
    betc = model.BETCInputs(**betc_dict)
    bets = model.BETSInputs(**bets_dict)
    lce_settings = json.loads(lce_settings_json)
    specs = build_lce_mc_specs(
        shared,
        lce_settings,
        json.loads(lce_mc_ranges_json),
    )
    return run_lce_monte_carlo_custom(
        specs,
        n_runs,
        random_seed,
        shared,
        diesel,
        betc,
        bets,
        lce_settings,
    )


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Model controls")
base_shared = model.SharedInputs()
base_diesel = model.DieselInputs()
base_betc = model.BETCInputs(
    battery_recycle_value_ratio=base_shared.battery_recycle_value_ratio
)
base_bets = model.BETSInputs(
    battery_recycle_value_ratio=base_shared.battery_recycle_value_ratio
)

with st.sidebar.expander("General and operation", expanded=True):
    years = st.number_input("TCO horizon (years)", 1, 20, base_shared.years, 1)
    discount_rate = st.number_input(
        "Discount rate", 0.0, 0.50, base_shared.discount_rate, 0.01, format="%.3f"
    )
    full_loaded_km_per_day = st.number_input(
        "Full-loaded km per day", 1.0, value=base_shared.full_loaded_km_per_day, step=10.0
    )
    unladen_ratio_to_full = st.number_input(
        "Unladen/full-loaded distance ratio(UK3/7)",
        0.0,
        2.0,
        base_shared.unladen_ratio_to_full,
        0.05,
        format="%.3f",
    )
    operational_days_per_year = st.number_input(
        "Operational days per year", 1, 365, base_shared.operational_days_per_year, 1
    )
    shift_per_day = st.number_input(
        "Shifts per day", 0.1, 5.0, base_shared.shift_per_day, 0.1
    )
    driver_hourly_pay = st.number_input(
        "Driver hourly pay (£)", 0.0, value=base_shared.driver_hourly_pay, step=0.5
    )
    worked_hours_per_week = st.number_input(
        "Worked hours per week", 1.0, 100.0, base_shared.worked_hours_per_week, 1.0
    )

with st.sidebar.expander("Financing, insurance and subsidy"):
    cost_of_capital = st.number_input(
        "Fleet cost of capital", 0.0, 0.50, base_shared.cost_of_capital, 0.01, format="%.3f"
    )
    upfront_payment_percentage = st.number_input(
        "Upfront payment share",
        0.0,
        1.0,
        base_shared.upfront_payment_percentage,
        0.05,
        format="%.2f",
    )
    loan_term_years = st.number_input(
        "Loan term (years)", 1, 20, base_shared.loan_term_years, 1
    )
    aeaas_cost_of_capital = st.number_input(
        "AEaaS cost of capital",
        0.0,
        0.50,
        base_shared.aeaas_cost_of_capital,
        0.01,
        format="%.3f",
    )
    diesel_insurance = st.number_input(
        "Diesel annual insurance (£)", 0.0, value=base_shared.diesel_insurance, step=500.0
    )
    bet_insurance_markup = st.number_input(
        "BET insurance markup",
        0.0,
        2.0,
        base_shared.bet_insurance_markup,
        0.05,
        format="%.2f",
    )
    bet_subsidy = st.number_input(
        "BET purchase subsidy (£)", 0.0, value=base_shared.bet_subsidy, step=1000.0
    )

with st.sidebar.expander("Energy and refuelling"):
    diesel_depot_share = st.number_input(
        "Diesel depot share", 0.0, 1.0, base_shared.diesel_depot_share, 0.05
    )
    diesel_depot_price_per_l = st.number_input(
        "Diesel depot price (£/L)", 0.0, value=base_shared.diesel_depot_price_per_l, step=0.01
    )
    diesel_public_price_per_l = st.number_input(
        "Diesel public price (£/L)", 0.0, value=base_shared.diesel_public_price_per_l, step=0.01
    )
    bet_depot_share = st.number_input(
        "BET-C depot charging share", 0.0, 1.0, base_shared.bet_depot_share, 0.05
    )
    bet_depot_energy_price_per_kwh = st.number_input(
        "BET depot electricity (£/kWh)",
        0.0,
        value=base_shared.bet_depot_energy_price_per_kwh,
        step=0.01,
    )
    bet_public_energy_price_per_kwh = st.number_input(
        "BET public electricity (£/kWh)",
        0.0,
        value=base_shared.bet_public_energy_price_per_kwh,
        step=0.01,
    )
    peak_price_per_kwh = st.number_input(
        "BaaS peak electricity (£/kWh)", 0.0, value=base_shared.peak_price_per_kwh, step=0.01
    )
    off_peak_price_per_kwh = st.number_input(
        "BaaS off-peak electricity (£/kWh)",
        0.0,
        value=base_shared.off_peak_price_per_kwh,
        step=0.01,
    )
    off_peak_share = st.number_input(
        "Off-peak swapping share", 0.0, 1.0, base_shared.off_peak_share, 0.05
    )
    electricity_margin = st.number_input(
        "BaaS electricity margin", 0.0, 3.0, base_shared.electricity_margin, 0.05
    )

with st.sidebar.expander("Vehicle and battery"):
    diesel_capex = st.number_input(
        "Diesel truck CAPEX (£)", 0.0, value=base_diesel.capex, step=1000.0
    )
    diesel_service_cost = st.number_input(
        "Diesel annual service cost (£)",
        0.0,
        value=base_diesel.annual_service_cost,
        step=500.0,
    )
    glider_capex = st.number_input(
        "Electric glider CAPEX (£)", 0.0, value=base_betc.glider_capex, step=1000.0
    )
    bet_service_cost = st.number_input(
        "BET annual service cost (£)", 0.0, value=base_betc.annual_service_cost, step=500.0
    )
    battery_price_per_kwh = st.number_input(
        "Battery price (£/kWh)", 0.0, value=base_betc.battery_price_per_kwh, step=5.0
    )
    betc_battery_capacity = st.number_input(
        "BET-C battery capacity (kWh)", 1.0, value=base_betc.battery_capacity_kwh, step=10.0
    )
    bets_battery_pack_capacity = st.number_input(
        "BET-S pack capacity (kWh)",
        1.0,
        value=base_bets.battery_pack_capacity_kwh,
        step=10.0,
    )
    battery_packs_per_truck = st.number_input(
        "BET-S packs per truck", 1.0, 10.0, base_bets.battery_packs_per_truck, 1.0
    )
    battery_lifetime_cycles = st.number_input(
        "Battery lifetime cycles", 1.0, value=base_betc.battery_lifetime_cycles, step=100.0
    )
    battery_recycle_value_ratio = st.number_input(
        "Battery residual/recycle ratio",
        0.0,
        1.0,
        base_shared.battery_recycle_value_ratio,
        0.05,
    )
    full_loaded_kwh_per_km_year1 = st.number_input(
        "BET full-loaded kWh/km in year 1",
        0.1,
        value=base_betc.full_loaded_kwh_per_km_year1,
        step=0.05,
    )

with st.sidebar.expander("BET-S station and BaaS"):
    expected_station_utilisation = st.number_input(
        "Expected station utilisation", 0.01, 1.0, base_bets.expected_station_utilisation, 0.05
    )
    station_capex = st.number_input(
        "Station CAPEX (£)", 0.0, value=base_bets.station_capex, step=50_000.0
    )
    site_capex = st.number_input(
        "Site CAPEX (£)", 0.0, value=base_bets.site_capex, step=50_000.0
    )
    station_annual_staff_costs = st.number_input(
        "Station annual staff costs (£)",
        0.0,
        value=base_bets.station_annual_staff_costs,
        step=5_000.0,
    )
    station_annual_other_service_costs = st.number_input(
        "Station other annual costs (£)",
        0.0,
        value=base_bets.station_annual_other_service_costs,
        step=1_000.0,
    )
    expected_annual_return_on_battery_renting = st.number_input(
        "Annual return on battery renting",
        0.0,
        1.0,
        base_bets.expected_annual_return_on_battery_renting,
        0.05,
    )
    swapping_fee_flat = st.number_input(
        "Fixed swapping fee (£/swap)", 0.0, value=base_bets.swapping_fee_flat, step=0.5
    )

with st.sidebar.expander("AEaaS granular cost factors"):
    aeaas_factor_values = {}
    for field, label in (
        ("aeaas_glider_cost_factor", "Glider cost factor"),
        ("aeaas_insurance_cost_factor", "Insurance cost factor"),
        ("aeaas_annual_service_cost_factor", "Annual service factor"),
        ("aeaas_station_capex_factor", "Station CAPEX factor"),
        ("aeaas_station_opex_factor", "Station OPEX factor"),
        ("aeaas_battery_depr_factor", "Battery depreciation factor"),
        ("aeaas_battery_service_factor", "Battery service factor"),
        ("aeaas_battery_rent_factor", "Battery rent factor"),
        ("aeaas_fixed_swapping_fee_factor", "Fixed swapping fee factor"),
        ("aeaas_energy_cost_factor", "Energy cost factor"),
    ):
        aeaas_factor_values[field] = st.number_input(
            label,
            0.0,
            2.0,
            float(getattr(base_shared, field)),
            0.05,
            format="%.2f",
        )

with st.sidebar.expander("LCE — Diesel assumptions"):
    diesel_lce_years = st.number_input("Diesel LCE horizon (years)", 1, 50, 10, 1)
    diesel_lce_curb_weight = st.number_input(
        "Diesel curb weight (kg)", 1.0, value=11_022.0, step=100.0
    )
    diesel_lce_production_factor = st.number_input(
        "Diesel production factor (kg CO2e/kg)", 0.0, value=7.0, step=0.1
    )
    diesel_lce_recycle_ratio = st.number_input(
        "Diesel recycling emission-saving ratio", 0.0, 1.0, 0.17, 0.01
    )
    diesel_lce_use_factor = st.number_input(
        "Diesel use factor (kg CO2e/L)", 0.0, value=3.17257, step=0.01, format="%.5f"
    )
    diesel_lce_payload = st.number_input(
        "Diesel payload benchmark (kg)", 1.0, value=29_000.0, step=1_000.0
    )

with st.sidebar.expander("LCE — BET-C assumptions"):
    betc_lce_electricity_factor = st.number_input(
        "BET-C electricity factor (kg CO2e/kWh)", 0.0, value=0.196, step=0.01, format="%.3f"
    )
    betc_lce_glider_weight = st.number_input(
        "BET-C glider weight (kg)", 1.0, value=9_986.0, step=100.0
    )
    betc_lce_glider_production_factor = st.number_input(
        "BET-C glider production factor (kg CO2e/kg)", 0.0, value=7.0, step=0.1
    )
    betc_lce_glider_recycle_ratio = st.number_input(
        "BET-C glider recycling saving ratio", 0.0, 1.0, 0.17, 0.01
    )
    betc_lce_battery_production_factor = st.number_input(
        "BET-C battery production factor (kg CO2e/kWh)", 0.0, value=57.0, step=1.0
    )
    betc_lce_battery_recycle_ratio = st.number_input(
        "BET-C battery recycling saving ratio", 0.0, 1.0, 0.05, 0.01
    )
    betc_lce_battery_sets = st.number_input(
        "BET-C battery sets over LCE life", 0.1, 20.0, 2.0, 0.1
    )
    betc_lce_payload_benchmark = st.number_input(
        "BET-C diesel payload benchmark (kg)", 1.0, value=29_000.0, step=1_000.0
    )
    betc_lce_payload_penalty = st.number_input(
        "BET-C payload penalty", 0.0, 1.0, 0.12, 0.01
    )

with st.sidebar.expander("LCE — BET-S assumptions"):
    bets_lce_electricity_factor = st.number_input(
        "BET-S electricity factor (kg CO2e/kWh)", 0.0, value=0.196, step=0.01, format="%.3f"
    )
    bets_lce_glider_weight = st.number_input(
        "BET-S glider weight (kg)", 1.0, value=9_986.0, step=100.0
    )
    bets_lce_glider_production_factor = st.number_input(
        "BET-S glider production factor (kg CO2e/kg)", 0.0, value=7.0, step=0.1
    )
    bets_lce_glider_recycle_ratio = st.number_input(
        "BET-S glider recycling saving ratio", 0.0, 1.0, 0.17, 0.01
    )
    bets_lce_battery_production_factor = st.number_input(
        "BET-S battery production factor (kg CO2e/kWh)", 0.0, value=57.0, step=1.0
    )
    bets_lce_battery_recycle_ratio = st.number_input(
        "BET-S battery recycling saving ratio", 0.0, 1.0, 0.19, 0.01
    )
    bets_lce_battery_sets = st.number_input(
        "BET-S battery sets over LCE life", 0.1, 20.0, 2.0, 0.1
    )
    bets_lce_second_life_ratio = st.number_input(
        "BET-S second-life capacity ratio", 0.0, 1.0, 0.80, 0.01
    )
    bets_lce_payload_benchmark = st.number_input(
        "BET-S diesel payload benchmark (kg)", 1.0, value=29_000.0, step=1_000.0
    )
    bets_lce_payload_penalty = st.number_input(
        "BET-S payload penalty", 0.0, 1.0, 0.12, 0.01
    )

with st.sidebar.expander("Simulation settings"):
    mc_runs = st.number_input("Monte Carlo runs", 20, 5000, 500, 20)
    random_seed = st.number_input("Random seed", 0, value=42, step=1)
    projection_start_year = st.number_input("Projection start year", 2020, 2050, 2026, 1)
    projection_end_year = st.number_input("Projection end year", 2021, 2060, 2040, 1)


default_tco_uncertainty_specs = model.get_uncertainty_specs(
    include_subsidy_uncertainty=True
)
with st.sidebar.expander("TCO Monte Carlo uncertainty ranges"):
    st.caption("Edit the triangular Min, Mode and Max values used by TCO Monte Carlo, one-at-a-time analysis and projection.")
    edited_tco_uncertainty = st.data_editor(
        uncertainty_specs_table(default_tco_uncertainty_specs),
        hide_index=True,
        disabled=["Variable", "Label"],
        column_config={
            "Min": st.column_config.NumberColumn(format="%.5g"),
            "Mode": st.column_config.NumberColumn(format="%.5g"),
            "Max": st.column_config.NumberColumn(format="%.5g"),
        },
        key="tco_uncertainty_editor",
        use_container_width=True,
    )
    tco_uncertainty_specs, tco_uncertainty_warnings = table_to_uncertainty_specs(
        edited_tco_uncertainty,
        default_tco_uncertainty_specs,
    )
    for warning in tco_uncertainty_warnings:
        st.warning(warning)

with st.sidebar.expander("LCE Monte Carlo uncertainty ranges"):
    st.caption("Relative-change rows use decimals: -0.20 means -20%.")
    edited_lce_mc_ranges = st.data_editor(
        lce_mc_default_range_table(base_shared),
        hide_index=True,
        disabled=["Analysis", "Range type"],
        column_config={
            "Min": st.column_config.NumberColumn(format="%.5g"),
            "Mode": st.column_config.NumberColumn(format="%.5g"),
            "Max": st.column_config.NumberColumn(format="%.5g"),
        },
        key="lce_uncertainty_editor",
        use_container_width=True,
    )
    lce_mc_range_rows, lce_mc_range_warnings = validate_lce_mc_range_table(
        edited_lce_mc_ranges
    )
    for warning in lce_mc_range_warnings:
        st.warning(warning)


shared = replace(
    base_shared,
    years=int(years),
    discount_rate=float(discount_rate),
    full_loaded_km_per_day=float(full_loaded_km_per_day),
    unladen_ratio_to_full=float(unladen_ratio_to_full),
    operational_days_per_year=int(operational_days_per_year),
    shift_per_day=float(shift_per_day),
    driver_hourly_pay=float(driver_hourly_pay),
    worked_hours_per_week=float(worked_hours_per_week),
    cost_of_capital=float(cost_of_capital),
    upfront_payment_percentage=float(upfront_payment_percentage),
    loan_term_years=int(loan_term_years),
    aeaas_cost_of_capital=float(aeaas_cost_of_capital),
    diesel_insurance=float(diesel_insurance),
    bet_insurance_markup=float(bet_insurance_markup),
    bet_subsidy=float(bet_subsidy),
    diesel_depot_share=float(diesel_depot_share),
    diesel_depot_price_per_l=float(diesel_depot_price_per_l),
    diesel_public_price_per_l=float(diesel_public_price_per_l),
    bet_depot_share=float(bet_depot_share),
    bet_depot_energy_price_per_kwh=float(bet_depot_energy_price_per_kwh),
    bet_public_energy_price_per_kwh=float(bet_public_energy_price_per_kwh),
    peak_price_per_kwh=float(peak_price_per_kwh),
    off_peak_price_per_kwh=float(off_peak_price_per_kwh),
    off_peak_share=float(off_peak_share),
    electricity_margin=float(electricity_margin),
    battery_recycle_value_ratio=float(battery_recycle_value_ratio),
    **{key: float(value) for key, value in aeaas_factor_values.items()},
)

diesel = replace(
    base_diesel,
    capex=float(diesel_capex),
    annual_service_cost=float(diesel_service_cost),
)

betc = replace(
    base_betc,
    glider_capex=float(glider_capex),
    annual_service_cost=float(bet_service_cost),
    battery_capacity_kwh=float(betc_battery_capacity),
    battery_price_per_kwh=float(battery_price_per_kwh),
    battery_lifetime_cycles=float(battery_lifetime_cycles),
    battery_recycle_value_ratio=float(battery_recycle_value_ratio),
    full_loaded_kwh_per_km_year1=float(full_loaded_kwh_per_km_year1),
)

bets = replace(
    base_bets,
    glider_capex=float(glider_capex),
    annual_service_cost=float(bet_service_cost),
    battery_pack_capacity_kwh=float(bets_battery_pack_capacity),
    battery_packs_per_truck=float(battery_packs_per_truck),
    battery_price_per_kwh=float(battery_price_per_kwh),
    battery_lifetime_cycles=float(battery_lifetime_cycles),
    battery_recycle_value_ratio=float(battery_recycle_value_ratio),
    full_loaded_kwh_per_km_year1=float(full_loaded_kwh_per_km_year1),
    expected_station_utilisation=float(expected_station_utilisation),
    station_capex=float(station_capex),
    site_capex=float(site_capex),
    station_annual_staff_costs=float(station_annual_staff_costs),
    station_annual_other_service_costs=float(station_annual_other_service_costs),
    expected_annual_return_on_battery_renting=float(
        expected_annual_return_on_battery_renting
    ),
    swapping_fee_flat=float(swapping_fee_flat),
)

lce_settings = {
    "diesel": {
        "lce_years": int(diesel_lce_years),
        "truck_curb_weight_kg": float(diesel_lce_curb_weight),
        "truck_production_emission_factor_kg_per_kg": float(
            diesel_lce_production_factor
        ),
        "truck_recycle_emission_saving_ratio": float(diesel_lce_recycle_ratio),
        "diesel_use_emission_factor_kg_co2e_per_litre": float(
            diesel_lce_use_factor
        ),
        "diesel_truck_payload_kg": float(diesel_lce_payload),
    },
    "bet_c": {
        "electricity_emission_factor_kg_per_kwh": float(
            betc_lce_electricity_factor
        ),
        "glider_weight_kg": float(betc_lce_glider_weight),
        "glider_production_emission_factor_kg_per_kg": float(
            betc_lce_glider_production_factor
        ),
        "glider_recycle_emission_saving_ratio": float(
            betc_lce_glider_recycle_ratio
        ),
        "battery_production_emission_factor_kg_per_kwh": float(
            betc_lce_battery_production_factor
        ),
        "battery_recycle_emission_saving_ratio": float(
            betc_lce_battery_recycle_ratio
        ),
        "battery_sets_needed": float(betc_lce_battery_sets),
        "diesel_truck_payload_kg": float(betc_lce_payload_benchmark),
        "payload_penalty": float(betc_lce_payload_penalty),
    },
    "bet_s": {
        "electricity_emission_factor_kg_per_kwh": float(
            bets_lce_electricity_factor
        ),
        "glider_weight_kg": float(bets_lce_glider_weight),
        "glider_production_emission_factor_kg_per_kg": float(
            bets_lce_glider_production_factor
        ),
        "glider_recycle_emission_saving_ratio": float(
            bets_lce_glider_recycle_ratio
        ),
        "battery_production_emission_factor_kg_per_kwh": float(
            bets_lce_battery_production_factor
        ),
        "battery_recycle_emission_saving_ratio": float(
            bets_lce_battery_recycle_ratio
        ),
        "battery_sets_needed": float(bets_lce_battery_sets),
        "second_life_capacity_ratio": float(bets_lce_second_life_ratio),
        "diesel_truck_payload_kg": float(bets_lce_payload_benchmark),
        "payload_penalty": float(bets_lce_payload_penalty),
    },
}


tco_uncertainty_specs_json = json.dumps(tco_uncertainty_specs, sort_keys=True)
lce_mc_ranges_json = json.dumps(lce_mc_range_rows, sort_keys=True)
lce_settings_json = json.dumps(lce_settings, sort_keys=True)

shared_dict = asdict(shared)
diesel_dict = asdict(diesel)
betc_dict = asdict(betc)
bets_dict = asdict(bets)

results = cached_baseline(shared_dict, diesel_dict, betc_dict, bets_dict)
gaps = model.extract_tco_gaps(results)
lce_results = cached_lce_baseline(
    shared_dict,
    diesel_dict,
    betc_dict,
    bets_dict,
    lce_settings_json,
)


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------
st.title("Truck TCO and Life-Cycle Emissions Analysis")
st.caption(
    "Computationally intensive Monte Carlo sections run only when requested."
)

with st.expander("Current model inputs", expanded=False):
    st.dataframe(dataclass_input_table(shared, diesel, betc, bets), use_container_width=True)

with st.expander("Current Monte Carlo uncertainty ranges", expanded=False):
    st.markdown("**TCO ranges**")
    st.dataframe(
        uncertainty_specs_table(tco_uncertainty_specs),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("**LCE ranges**")
    st.dataframe(pd.DataFrame(lce_mc_range_rows), use_container_width=True, hide_index=True)

(
    tab_baseline,
    tab_sensitivity,
    tab_baas,
    tab_mc,
    tab_aeaas,
    tab_projection,
    tab_lce,
) = st.tabs(
    [
        "TCO baseline",
        "TCO sensitivity",
        "BaaS heatmaps",
        "TCO Monte Carlo",
        "AEaaS",
        "TCO Projection",
        "LCE",
    ]
)

with tab_baseline:
    st.header("Deterministic TCO results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Diesel discounted TCO", fmt_gbp(results["diesel"]["tco_discounted"]))
    c2.metric("BET-C discounted TCO", fmt_gbp(results["bet_c"]["tco_discounted"]))
    c3.metric("BET-S discounted TCO", fmt_gbp(results["bet_s"]["tco_discounted"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("BET-C − Diesel", fmt_gbp(gaps["bet_c_vs_diesel"]))
    c2.metric("BET-S − Diesel", fmt_gbp(gaps["bet_s_vs_diesel"]))
    c3.metric("BET-S − BET-C", fmt_gbp(gaps["bet_s_vs_bet_c"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Diesel TCO/km", fmt_gbp(results["diesel"]["tco_per_km_discounted"], 3))
    c2.metric("BET-C TCO/km", fmt_gbp(results["bet_c"]["tco_per_km_discounted"], 3))
    c3.metric("BET-S TCO/km", fmt_gbp(results["bet_s"]["tco_per_km_discounted"], 3))

    row1 = st.columns(4)
    with row1[0]:
        show_figure(model.plot_tco_comparison(results))
    with row1[1]:
        show_figure(model.plot_tco_gap(results))
    with row1[2]:
        show_figure(model.plot_tco_per_km_comparison(results))
    with row1[3]:
        show_figure(model.plot_tco_per_km_gap(results))



with tab_sensitivity:
    st.header("Deterministic sensitivity analysis")
    sensitivity_results = cached_tco_sensitivity(
        shared_dict, diesel_dict, betc_dict, bets_dict
    )
    for sensitivity_result in sensitivity_results:
        variable_label = model.get_pretty_label(sensitivity_result["variable_name"])
        target_label = "+".join(sensitivity_result["target_class"])
        st.subheader(f"{variable_label}")
        cols = st.columns(2)
        with cols[0]:
            show_figure(model.plot_sensitivity_bar(sensitivity_result))
        with cols[1]:
            show_figure(model.plot_sensitivity_bar_per_km(sensitivity_result))

with tab_baas:
    st.header("BaaS provider and utilisation heatmaps")
    with st.spinner("Calculating BaaS grids..."):
        baas_grid_df, utilisation_grid_df, tco_gap_df = cached_heatmaps(
            shared_dict, diesel_dict, bets_dict
        )
    st.subheader("IRR and payback across fee, energy-margin and battery-return assumptions")
    show_figure(model.plot_baas_irr_payback_heatmaps(baas_grid_df))
    st.subheader("IRR and payback across station-utilisation assumptions")
    show_figure(model.plot_baas_utilisation_irr_payback_heatmaps(utilisation_grid_df))
    st.subheader("BET-S minus Diesel discounted TCO")
    show_figure(
        model.plot_baas_utilisation_tco_gap_heatmaps(
            tco_gap_df,
            gap_column="gap_bets_diesel",
            title="BET-S minus Diesel Discounted TCO by Station Utilisation",
        )
    )
    st.subheader("BET-S minus Diesel discounted TCO per km")
    show_figure(
        model.plot_baas_utilisation_tco_gap_heatmaps(
            tco_gap_df,
            gap_column="gap_bets_diesel_per_km",
            title="BET-S minus Diesel Discounted TCO per km by Station Utilisation",
        )
    )

with tab_mc:
    st.header("TCO uncertainty using Monte Carlo simulation")
    st.info(
        "The triangular uncertainty ranges are editable in the sidebar and are used by the full simulation and every one-at-a-time run."
    )
    run_mc = st.button("Run / refresh TCO Monte Carlo", key="run_tco_mc")
    mc_signature = json.dumps(
        {
            "runs": int(mc_runs),
            "seed": int(random_seed),
            "specs": tco_uncertainty_specs,
            "shared": shared_dict,
            "diesel": diesel_dict,
            "betc": betc_dict,
            "bets": bets_dict,
        },
        sort_keys=True,
    )
    if run_mc:
        st.session_state["tco_mc_signature"] = mc_signature
    if st.session_state.get("tco_mc_signature") == mc_signature:
        with st.spinner("Running full and one-at-a-time TCO Monte Carlo analyses..."):
            mc_df, mc_summary_df, mc_probability_df, independent_mc_df = cached_tco_mc(
                shared_dict,
                diesel_dict,
                betc_dict,
                bets_dict,
                int(mc_runs),
                int(random_seed),
                tco_uncertainty_specs_json,
            )

        with st.expander("Monte Carlo summary tables", expanded=False):
            st.dataframe(mc_summary_df, use_container_width=True, hide_index=True)
            st.dataframe(mc_probability_df, use_container_width=True, hide_index=True)

        st.subheader("Discounted TCO per km distributions")
        show_figure(plot_monte_carlo_histogram_grid(mc_df, per_km=True))

        with st.expander("Show discounted total TCO distributions", expanded=False):
            show_figure(plot_monte_carlo_histogram_grid(mc_df, per_km=False))

        driver_inputs = [
            spec["variable"] for spec in tco_uncertainty_specs
            if spec["variable"] in mc_df.columns
        ]
        drivers_df = model.get_drivers_of_gap(
            mc_df,
            gap_column="gap_bet_s_diesel",
            input_columns=driver_inputs,
        )
        st.subheader("Drivers of the BET-S − Diesel TCO gap")
        show_figure(model.plot_drivers(drivers_df, gap_name="BET-S - Diesel"))

        st.subheader("Independent one-at-a-time uncertainty")
        for scenario, scenario_df in independent_mc_df.groupby("subsidy_scenario"):
            st.markdown(f"**{scenario}**")
            show_figure(model.plot_independent_bets_vs_diesel_boxplot(scenario_df))
            show_figure(model.plot_independent_tco_boxplots(scenario_df))
            show_figure(model.plot_independent_gap_boxplots(scenario_df))
    else:
        st.caption("Click the run button to calculate this section with the current simulation settings.")

with tab_aeaas:
    st.header("Asset-and-Energy-as-a-Service")
    st.markdown(
        "The asset manager purchases trucks and energy assets, applies the granular cost factors "
        "in the sidebar, and charges fleet operators according to usage and the target margin."
    )
    run_margin = st.button("Run / refresh AEaaS uncertainty", key="run_margin")
    margin_signature = (int(mc_runs), int(random_seed))
    if run_margin:
        st.session_state["margin_signature"] = margin_signature
    if st.session_state.get("margin_signature") == margin_signature:
        with st.spinner("Running AEaaS margin uncertainty..."):
            margin_raw_df, margin_summary_df = cached_margin(int(mc_runs), int(random_seed))
        for scenario, sub_df in margin_summary_df.groupby("subsidy_scenario"):
            st.subheader(str(scenario))
            cols = st.columns(2)
            ordered = sub_df.sort_values("asset_manager_margin")
            with cols[0]:
                show_figure(
                    model.plot_margin_vs_freight_all_in_per_km_with_uncertainty(
                        ordered, title_suffix=f"- {scenario}"
                    )
                )
            with cols[1]:
                show_figure(
                    model.plot_margin_vs_gap_with_uncertainty(
                        ordered, title_suffix=f"- {scenario}"
                    )
                )

    else:
        st.caption("Click the run button to calculate this section.")

with tab_projection:
    st.header("TCO projection under uncertainty")
    if int(projection_end_year) <= int(projection_start_year):
        st.warning("Projection end year must be later than projection start year.")
    else:
        run_projection = st.button("Run / refresh projection", key="run_projection")
        projection_signature = json.dumps(
            {
                "start": int(projection_start_year),
                "end": int(projection_end_year),
                "runs": int(mc_runs),
                "seed": int(random_seed),
                "specs": tco_uncertainty_specs,
                "shared": shared_dict,
                "diesel": diesel_dict,
                "betc": betc_dict,
                "bets": bets_dict,
            },
            sort_keys=True,
        )
        if run_projection:
            st.session_state["projection_signature"] = projection_signature
        if st.session_state.get("projection_signature") == projection_signature:
            with st.spinner("Running projected TCO uncertainty..."):
                projection_raw, projection_total, projection_per_km = cached_projection(
                    int(projection_start_year),
                    int(projection_end_year),
                    int(mc_runs),
                    int(random_seed),
                    tco_uncertainty_specs_json,
                    shared_dict,
                    diesel_dict,
                    betc_dict,
                    bets_dict,
                )
            projection_groups = list(projection_total.groupby("subsidy_scenario"))
            projection_cols = st.columns(2)
            for col, (scenario, sub_df) in zip(projection_cols, projection_groups):
                with col:
                    show_figure(
                        model.plot_projection_with_uncertainty(
                            sub_df.sort_values("year"),
                            title_suffix=f"- {scenario}",
                        )
                    )
        else:
            st.caption("Click the run button to calculate this section.")

with tab_lce:
    st.header("Life-cycle emissions")

    st.markdown("**Life-cycle emissions per kilometre**")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Diesel kg CO2e/km",
        f"{lce_results['diesel']['life_cycle_emissions_per_km']:.4f}",
    )
    c2.metric(
        "BET-C kg CO2e/km",
        f"{lce_results['bet_c']['life_cycle_emissions_per_km']:.4f}",
    )
    c3.metric(
        "BET-S kg CO2e/km",
        f"{lce_results['bet_s']['life_cycle_emissions_per_km']:.4f}",
    )

    st.markdown("**Life-cycle emissions per tonne-kilometre (Representative Payload)**")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Diesel kg CO2e/tonne-km",
        f"{lce_results['diesel']['life_cycle_emissions_per_tonne_km']:.4f}",
    )
    c2.metric(
        "BET-C kg CO2e/tonne-km",
        f"{lce_results['bet_c']['life_cycle_emissions_per_tonne_km']:.4f}",
    )
    c3.metric(
        "BET-S kg CO2e/tonne-km",
        f"{lce_results['bet_s']['life_cycle_emissions_per_tonne_km']:.4f}",
    )


    cols = st.columns(2)
    with cols[0]:
        show_figure(model.plot_lce_per_tonne_km_comparison(lce_results))
    with cols[1]:
        show_figure(model.plot_lce_per_km_comparison(lce_results))

    st.subheader("LCE deterministic sensitivity")
    lce_sensitivity_results = cached_lce_sensitivity(
        shared_dict,
        diesel_dict,
        betc_dict,
        bets_dict,
        lce_settings_json,
    )
    for sensitivity_name, lce_sensitivity in lce_sensitivity_results.items():
        st.markdown(f"**{sensitivity_name}**")
        cols = st.columns(2)
        with cols[0]:
            show_figure(
                model.plot_lce_sensitivity_bar(
                    lce_sensitivity,
                    title=f"LCE sensitivity per tonne-km: {sensitivity_name}",
                    metric="tonne_km",
                )
            )
        with cols[1]:
            show_figure(
                model.plot_lce_sensitivity_bar(
                    lce_sensitivity,
                    title=f"LCE sensitivity per km: {sensitivity_name}",
                    metric="km",
                )
            )

    st.subheader("LCE Monte Carlo uncertainty")
    run_lce_mc = st.button("Run / refresh LCE Monte Carlo", key="run_lce_mc")
    lce_mc_signature_json = json.dumps(
        {
            "n_runs": int(mc_runs),
            "seed": int(random_seed),
            "shared": shared_dict,
            "diesel": diesel_dict,
            "betc": betc_dict,
            "bets": bets_dict,
            "settings": lce_settings,
            "ranges": lce_mc_range_rows,
        },
        sort_keys=True,
    )
    if run_lce_mc:
        st.session_state["lce_mc_signature"] = lce_mc_signature_json
    if st.session_state.get("lce_mc_signature") == lce_mc_signature_json:
        with st.spinner("Running single-variable and total LCE uncertainty..."):
            lce_single_df, lce_total_df = cached_lce_mc(
                shared_dict,
                diesel_dict,
                betc_dict,
                bets_dict,
                lce_settings_json,
                lce_mc_ranges_json,
                int(mc_runs),
                int(random_seed),
            )

        st.markdown("**Single-variable uncertainty**")
        cols = st.columns(2)
        with cols[0]:
            show_figure(
                model.plot_single_variable_lce_monte_carlo_boxplots_combined(
                    lce_single_df, metric="per_tonne_km", include_second_life=True
                )
            )
        with cols[1]:
            show_figure(
                model.plot_single_variable_lce_monte_carlo_boxplots_combined(
                    lce_single_df, metric="per_km", include_second_life=True
                )
            )

        st.markdown("**Total LCE uncertainty**")
        cols = st.columns(2)
        with cols[0]:
            show_figure(
                model.plot_total_lce_uncertainty_bar_with_percentile_range(
                    lce_total_df,
                    metric="per_tonne_km",
                    include_second_life=True,
                    use_median=True,
                )
            )
        with cols[1]:
            show_figure(
                model.plot_total_lce_uncertainty_bar_with_percentile_range(
                    lce_total_df,
                    metric="per_km",
                    include_second_life=True,
                    use_median=True,
                )
            )
    else:
        st.caption("Click the run button to calculate LCE Monte Carlo uncertainty.")

