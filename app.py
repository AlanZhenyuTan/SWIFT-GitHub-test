from __future__ import annotations

from dataclasses import asdict, replace

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

from tco_model import (
    BETCInputs,
    BETSInputs,
    DieselInputs,
    SharedInputs,
    extract_tco_gaps,
    get_drivers_of_gap,
    get_pretty_label,
    get_uncertainty_specs,
    run_independent_variable_monte_carlo,
    run_margin_sweep_with_uncertainty,
    run_model,
    run_monte_carlo_simulation,
    run_projection_monte_carlo,
    run_tco_projection,
    summarize_independent_effect_spread,
    summarize_monte_carlo_results,
    summarize_projection_uncertainty,
)

st.set_page_config(page_title="Truck TCO Analysis", layout="wide")


# -------------------------------
# Styling
# -------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        padding: 14px 16px;
        border-radius: 16px;
    }
    .section-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        margin: 0.8rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------
# Helpers
# -------------------------------
def fmt_gbp(value: float, decimals: int = 0) -> str:
    return f"£{value:,.{decimals}f}"


def uncertainty_table(shared, diesel, betc, bets, uncertainty_overrides=None) -> pd.DataFrame:
    rows = []
    for spec in get_uncertainty_specs(shared, diesel, betc, bets, uncertainty_overrides):
        mode = spec["mode"]

        if spec["variable"] == "discount_rate":
            mode = shared.discount_rate
        elif spec["variable"] == "full_loaded_km_per_day":
            mode = shared.full_loaded_km_per_day
        elif spec["variable"] == "peak_price_per_kwh":
            mode = shared.peak_price_per_kwh
        elif spec["variable"] == "off_peak_share":
            mode = shared.off_peak_share
        elif spec["variable"] == "bet_depot_energy_price_per_kwh":
            mode = shared.bet_depot_energy_price_per_kwh
        elif spec["variable"] == "bet_public_energy_price_per_kwh":
            mode = shared.bet_public_energy_price_per_kwh
        elif spec["variable"] == "bet_subsidy":
            mode = shared.bet_subsidy
        elif spec["variable"] == "full_loaded_kwh_per_km_year1":
            mode = betc.full_loaded_kwh_per_km_year1
        elif spec["variable"] == "glider_capex":
            mode = betc.glider_capex
        elif spec["variable"] == "battery_price_per_kwh":
            mode = betc.battery_price_per_kwh
        elif spec["variable"] == "battery_lifetime_cycles":
            mode = betc.battery_lifetime_cycles
        elif spec["variable"] == "battery_capacity_kwh":
            mode = betc.battery_capacity_kwh
        elif spec["variable"] == "battery_recycle_ratio":
            mode = betc.battery_recycle_ratio

        rows.append(
            {
                "Variable": get_pretty_label(spec["variable"]),
                "Min": spec["left"],
                "Mode": mode,
                "Max": spec["right"],
            }
        )

    return pd.DataFrame(rows)


def fig_tco_comparison(results):
    labels = ["Diesel", "BET-C", "BET-S"]
    values = [
        results["diesel"]["tco_discounted"],
        results["bet_c"]["tco_discounted_recycle"],
        results["bet_s"]["tco_discounted_recycle"],
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])
    ax.set_title("Discounted TCO comparison")
    ax.set_ylabel("TCO (£)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def fig_tco_gap(gaps):
    labels = ["BET-C - Diesel", "BET-S - Diesel", "BET-S - BET-C"]
    values = [gaps["bet_c_vs_diesel"], gaps["bet_s_vs_diesel"], gaps["bet_s_vs_bet_c"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(labels, values, color=["tab:purple", "tab:red", "tab:brown"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Discounted TCO gaps")
    ax.set_ylabel("TCO gap (£)")
    ax.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.0f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.tight_layout()
    return fig

def fig_tco_per_km_comparison(results):
    labels = ["Diesel", "BET-C", "BET-S"]
    values = [
        results["diesel"]["tco_per_km_discounted"],
        results["bet_c"]["tco_per_km_discounted_recycle"],
        results["bet_s"]["tco_per_km_discounted_recycle"],
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])


    ax.set_title("Discounted TCO per km Comparison")
    ax.set_ylabel("TCO (£/km)")

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    return fig

def fig_monte_carlo_histograms(df: pd.DataFrame):
    specs = [
        ("diesel_tco", "Diesel discounted TCO", "TCO (£)", "tab:blue"),
        ("bet_c_tco", "BET-C discounted TCO", "TCO (£)", "tab:orange"),
        ("bet_s_tco", "BET-S discounted TCO", "TCO (£)", "tab:green"),
        ("gap_bet_c_diesel", "BET-C - Diesel Gap", "TCO gap (£)", "tab:purple"),
        ("gap_bet_s_diesel", "BET-S - Diesel Gap", "TCO gap (£)", "tab:red"),
        ("gap_bet_s_bet_c", "BET-S - BET-C Gap", "TCO gap (£)", "tab:brown"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))
    for ax, (col, title, xlabel, color) in zip(axes.flatten(), specs):
        ax.hist(df[col].dropna(), bins=20, color=color, alpha=0.82)
        mean_value = df[col].mean()
        ax.axvline(df[col].mean(), color="black", linestyle="--", linewidth=1.2, label="Mean")
        ax.text(mean_value,ax.get_ylim()[1] * 0.9,f"Mean = {mean_value:,.0f}",va="top",ha="right")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
    fig.suptitle("Monte Carlo distributions", fontsize=16, y=1.02)
    fig.tight_layout()
    return fig


def fig_driver_bar(driver_df: pd.DataFrame, gap_name="BET-S - Diesel"):
    labels = [get_pretty_label(v) for v in driver_df["variable"]]
    values = driver_df["correlation_with_gap"]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    colors = ["tab:red" if v >= 0 else "tab:blue" for v in values]
    bars = ax.bar(labels, values, color=colors, alpha=0.82)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"Drivers of {gap_name} gap")
    ax.set_ylabel("Correlation with the gap")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.tight_layout()
    return fig


def _independent_variable_order(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = exclude or []
    return [v for v in df["variable"].drop_duplicates() if v not in exclude]


def fig_independent_tco_boxplots(df: pd.DataFrame):
    variable_order = _independent_variable_order(df)
    positions, data, centers, boundaries = [], [], [], []
    gap_between_groups = 2.0
    for g, var in enumerate(variable_order):
        base = 1.0 + g * (3 + gap_between_groups)
        data.extend([
            df.loc[df["variable"] == var, "diesel_tco"].dropna(),
            df.loc[df["variable"] == var, "bet_c_tco"].dropna(),
            df.loc[df["variable"] == var, "bet_s_tco"].dropna(),
        ])
        positions.extend([base, base + 1, base + 2])
        centers.append(base + 1)
        if g < len(variable_order) - 1:
            next_base = 1.0 + (g + 1) * (3 + gap_between_groups)
            boundaries.append((base + 2 + next_base) / 2)

    fig, ax = plt.subplots(figsize=(20, 6.2))
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
    colors = (["tab:blue", "tab:orange", "tab:green"] * len(variable_order))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.7)
    for x in boundaries:
        ax.axvline(x=x, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)
    ax.set_xticks(centers)
    ax.set_xticklabels([get_pretty_label(v) for v in variable_order], rotation=35, ha="right")
    ax.set_ylabel("Discounted TCO (£)")
    ax.set_title("Independent one-at-a-time effect on TCO")
    ax.legend(handles=[
        mpatches.Patch(color="tab:blue", label="Diesel"),
        mpatches.Patch(color="tab:orange", label="BET-C"),
        mpatches.Patch(color="tab:green", label="BET-S"),
    ], loc="upper right")
    fig.tight_layout()
    return fig


def fig_independent_gap_boxplots(df: pd.DataFrame):
    variable_order = _independent_variable_order(df)
    positions, data, centers, boundaries = [], [], [], []
    gap_between_groups = 2.0
    for g, var in enumerate(variable_order):
        base = 1.0 + g * (3 + gap_between_groups)
        data.extend([
            df.loc[df["variable"] == var, "gap_bet_c_diesel"].dropna(),
            df.loc[df["variable"] == var, "gap_bet_s_diesel"].dropna(),
            df.loc[df["variable"] == var, "gap_bet_s_bet_c"].dropna(),
        ])
        positions.extend([base, base + 1, base + 2])
        centers.append(base + 1)
        if g < len(variable_order) - 1:
            next_base = 1.0 + (g + 1) * (3 + gap_between_groups)
            boundaries.append((base + 2 + next_base) / 2)

    fig, ax = plt.subplots(figsize=(20, 6.2))
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
    colors = (["tab:purple", "tab:red", "tab:brown"] * len(variable_order))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.7)
    for x in boundaries:
        ax.axvline(x=x, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(centers)
    ax.set_xticklabels([get_pretty_label(v) for v in variable_order], rotation=35, ha="right")
    ax.set_ylabel("TCO gap (£)")
    ax.set_title("Independent one-at-a-time effect on TCO gaps")
    ax.legend(handles=[
        mpatches.Patch(color="tab:purple", label="BET-C - Diesel"),
        mpatches.Patch(color="tab:red", label="BET-S - Diesel"),
        mpatches.Patch(color="tab:brown", label="BET-S - BET-C"),
    ], loc="upper right")
    fig.tight_layout()
    return fig


def fig_independent_bets_vs_diesel_boxplot(df: pd.DataFrame):
    variable_order = _independent_variable_order(
        df,
        exclude=["bet_depot_energy_price_per_kwh", "bet_public_energy_price_per_kwh", "battery_capacity_kwh"],
    )
    data = [df.loc[df["variable"] == var, "gap_bet_s_diesel"].dropna() for var in variable_order]
    positions = list(range(1, len(variable_order) + 1))
    fig, ax = plt.subplots(figsize=(18, 5.8))
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("tab:red")
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.7)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([get_pretty_label(v) for v in variable_order], rotation=35, ha="right")
    ax.set_ylabel("BET-S - Diesel TCO gap (£)")
    ax.set_title("Independent one-at-a-time effect on BET-S - Diesel gap")
    fig.tight_layout()
    return fig


def fig_projection(df: pd.DataFrame, ycols: list[str], labels: list[str], title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for col, label in zip(ycols, labels):
        ax.plot(df["year"], df[col], marker="o", linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Purchase year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(df["year"])
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_projection_uncertainty(summary_df: pd.DataFrame, metric_base: str, title: str, ylabel: str):
    specs = [("diesel", "Diesel"), ("betc", "BET-C"), ("bets", "BET-S")]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for prefix, label in specs:
        metric = f"{prefix}_{metric_base}"
        ax.plot(summary_df["year"], summary_df[f"{metric}_p50"], marker="o", linewidth=2, label=f"{label} median")
        ax.fill_between(summary_df["year"], summary_df[f"{metric}_p5"], summary_df[f"{metric}_p95"], alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel("Purchase year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(summary_df["year"])
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    return fig


def summarize_margin_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for margin, group in df.groupby("asset_manager_margin"):
        rows.append(
            {
                "asset_manager_margin": margin,
                "diesel_p5": group["diesel_tco_per_km"].quantile(0.05),
                "diesel_p50": group["diesel_tco_per_km"].quantile(0.50),
                "diesel_p95": group["diesel_tco_per_km"].quantile(0.95),
                "bets_p5": group["bets_freight_all_in_per_km"].quantile(0.05),
                "bets_p50": group["bets_freight_all_in_per_km"].quantile(0.50),
                "bets_p95": group["bets_freight_all_in_per_km"].quantile(0.95),
                "gap_p5": group["bets_minus_diesel_per_km"].quantile(0.05),
                "gap_p50": group["bets_minus_diesel_per_km"].quantile(0.50),
                "gap_p95": group["bets_minus_diesel_per_km"].quantile(0.95),
            }
        )
    return pd.DataFrame(rows).sort_values("asset_manager_margin").reset_index(drop=True)


def fig_margin_cost(summary_df: pd.DataFrame):
    x = summary_df["asset_manager_margin"] * 100
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(x, summary_df["diesel_p50"], marker="o", linewidth=2, label="Diesel Truck TCO per km")
    ax.fill_between(x, summary_df["diesel_p5"], summary_df["diesel_p95"], alpha=0.18)
    ax.plot(x, summary_df["bets_p50"], marker="o", color="tab:green", linewidth=2, label="BET-S AEaaS Cost per km")
    ax.fill_between(x, summary_df["bets_p5"], summary_df["bets_p95"],color="tab:green", alpha=0.18)
    ax.set_title("AEaaS margin and freight cost per km")
    ax.set_xlabel("Asset-manager margin (%)")
    ax.set_ylabel("Cost (£/km)")
    ax.legend()
    fig.tight_layout()
    return fig


# -------------------------------
# Cached runners
# -------------------------------
@st.cache_data(show_spinner=False)
def run_baseline_cached(shared_dict, diesel_dict, betc_dict, bets_dict, asset_manager_margin):
    return run_model(
        shared=SharedInputs(**shared_dict),
        diesel_inp=DieselInputs(**diesel_dict),
        betc_inp=BETCInputs(**betc_dict),
        bets_inp=BETSInputs(**bets_dict),
        asset_manager_margin=asset_manager_margin,
    )


@st.cache_data(show_spinner=False)
def run_mc_cached(n_runs: int, random_seed: int, shared_dict, diesel_dict, betc_dict, bets_dict, uncertainty_overrides):
    shared = SharedInputs(**shared_dict)
    diesel = DieselInputs(**diesel_dict)
    betc = BETCInputs(**betc_dict)
    bets = BETSInputs(**bets_dict)
    mc_df = run_monte_carlo_simulation(
        n_runs=n_runs,
        random_seed=random_seed,
        shared=shared,
        diesel_inp=diesel,
        betc_inp=betc,
        bets_inp=bets,
        uncertainty_overrides=uncertainty_overrides,
    )
    summary_df, probability_df = summarize_monte_carlo_results(mc_df)
    driver_df = get_drivers_of_gap(
        mc_df,
        gap_column="gap_bet_s_diesel",
        input_columns=[
            "bet_subsidy",
            "full_loaded_km_per_day",
            "peak_price_per_kwh",
            "off_peak_share",
            "bet_depot_energy_price_per_kwh",
            "bet_public_energy_price_per_kwh",
            "full_loaded_kwh_per_km_year1",
            "glider_capex",
            "battery_price_per_kwh",
            "battery_recycle_ratio",
            "battery_lifetime_cycles",
            "unladen_energy_saving",
            "battery_capacity_kwh",
            "discount_rate",
        ],
    )
    return mc_df, summary_df, probability_df, driver_df


@st.cache_data(show_spinner=False)
def run_independent_mc_cached(n_runs: int, random_seed: int, shared_dict, diesel_dict, betc_dict, bets_dict, uncertainty_overrides):
    shared = SharedInputs(**shared_dict)
    diesel = DieselInputs(**diesel_dict)
    betc = BETCInputs(**betc_dict)
    bets = BETSInputs(**bets_dict)
    df = run_independent_variable_monte_carlo(
        n_runs=n_runs,
        random_seed=random_seed,
        shared=shared,
        diesel_inp=diesel,
        betc_inp=betc,
        bets_inp=bets,
        uncertainty_overrides=uncertainty_overrides,
    )
    summary = summarize_independent_effect_spread(df)
    return df, summary


@st.cache_data(show_spinner=False)
def run_projection_cached(shared_dict, diesel_dict, betc_dict, bets_dict, start_year: int, end_year: int):
    return run_tco_projection(
        start_year=start_year,
        end_year=end_year,
        shared=SharedInputs(**shared_dict),
        diesel_inp=DieselInputs(**diesel_dict),
        betc_inp=BETCInputs(**betc_dict),
        bets_inp=BETSInputs(**bets_dict),
    )


@st.cache_data(show_spinner=False)
def run_projection_mc_cached(start_year: int, end_year: int, n_runs: int, random_seed: int, shared_dict, diesel_dict, betc_dict, bets_dict, uncertainty_overrides):
    shared = SharedInputs(**shared_dict)
    diesel = DieselInputs(**diesel_dict)
    betc = BETCInputs(**betc_dict)
    bets = BETSInputs(**bets_dict)
    df = run_projection_monte_carlo(
        start_year=start_year,
        end_year=end_year,
        n_runs=n_runs,
        random_seed=random_seed,
        shared=shared,
        diesel_inp=diesel,
        betc_inp=betc,
        bets_inp=bets,
        uncertainty_overrides=uncertainty_overrides,
    )
    summary = summarize_projection_uncertainty(
        df,
        metric_cols=[
            "diesel_tco_discounted",
            "betc_tco_discounted",
            "bets_tco_discounted",
            "diesel_tco_per_km",
            "betc_tco_per_km",
            "bets_tco_per_km",
            "diesel_tco_per_kwh",
            "betc_tco_per_kwh",
            "bets_tco_per_kwh",
        ],
    )
    return df, summary


@st.cache_data(show_spinner=False)
def run_margin_mc_cached(margins_tuple, n_runs: int, random_seed: int, shared_dict, diesel_dict, betc_dict, bets_dict, uncertainty_overrides):
    shared = SharedInputs(**shared_dict)
    diesel = DieselInputs(**diesel_dict)
    betc = BETCInputs(**betc_dict)
    bets = BETSInputs(**bets_dict)
    df = run_margin_sweep_with_uncertainty(
        margins=np.array(margins_tuple),
        n_runs=n_runs,
        random_seed=random_seed,
        shared=shared,
        diesel_inp=diesel,
        betc_inp=betc,
        bets_inp=bets,
        uncertainty_overrides=uncertainty_overrides,
    )
    return df, summarize_margin_uncertainty(df)


# -------------------------------
# Sidebar inputs
# -------------------------------
def build_inputs():
    base_shared = SharedInputs()
    base_diesel = DieselInputs()
    base_betc = BETCInputs(battery_recycle_ratio=base_shared.battery_recycle_value_ratio)
    base_bets = BETSInputs(battery_recycle_ratio=base_shared.battery_recycle_value_ratio)

    st.sidebar.title("Controls")
    st.sidebar.caption("Please change core assumptions here; Monte Carlo ranges, which will change if assumptions here are changed, are shown in the main page expander.")

    years = st.sidebar.number_input("Analysis years", min_value=1, max_value=20, value=base_shared.years, step=1)
    discount_rate = st.sidebar.number_input("Discount rate", min_value=0.0, value=base_shared.discount_rate, step=0.01, format="%.3f")
    full_loaded_km_per_day = st.sidebar.number_input("Full-loaded km/day", min_value=0.0, value=base_shared.full_loaded_km_per_day, step=10.0)
    bet_subsidy = st.sidebar.number_input("BET purchase subsidy (£)", min_value=0.0, value=base_shared.bet_subsidy, step=1000.0)
    asset_manager_margin = st.sidebar.number_input("AEaaS asset-manager margin", min_value=0.0, value=0.10, step=0.01, format="%.2f")

    with st.sidebar.expander("Energy prices", expanded=False):
        diesel_public_price_per_l = st.number_input("Diesel public price (£/L)", value=base_shared.diesel_public_price_per_l, step=0.01)
        diesel_depot_price_per_l = st.number_input("Diesel depot price (£/L)", value=base_shared.diesel_depot_price_per_l, step=0.01)
        bet_depot_energy_price_per_kwh = st.number_input("BET depot price (£/kWh)", value=base_shared.bet_depot_energy_price_per_kwh, step=0.01)
        bet_public_energy_price_per_kwh = st.number_input("BET public price (£/kWh)", value=base_shared.bet_public_energy_price_per_kwh, step=0.01)
        peak_price_per_kwh = st.number_input("Peak swapping price (£/kWh)", value=base_shared.peak_price_per_kwh, step=0.01)
        off_peak_price_per_kwh = st.number_input("Off-peak swapping price (£/kWh)", value=base_shared.off_peak_price_per_kwh, step=0.01)
        off_peak_share = st.slider("Off-peak swapping percentage", 0.0, 1.0, float(base_shared.off_peak_share), 0.01)
        electricity_margin = st.number_input("Electricity margin of energy providers", min_value=0.0, value=base_shared.electricity_margin, step=0.01, format="%.2f")

    with st.sidebar.expander("Vehicle inputs", expanded=False):
        diesel_capex = st.number_input("Diesel truck acquisiton cost (£)", value=base_diesel.capex, step=1000.0)
        glider_capex = st.number_input("Electric glider acquisition cost (£)", value=base_betc.glider_capex, step=1000.0)
        bet_battery_price = st.number_input("BET battery price (£/kWh)", value=base_betc.battery_price_per_kwh, step=5.0)
        battery_recycle_ratio = st.number_input ("Battery residual percentage", value=base_betc.battery_recycle_ratio, step=0.01)
        diesel_l_per_km = st.number_input("Diesel year-1 fuel economy (L/km)", value=base_diesel.fuel_economy_full_loaded_year1_l_per_km, step=0.01, format="%.3f")
        betc_battery_capacity = st.number_input("BET-C battery capacity (kWh)", value=base_betc.battery_capacity_kwh, step=10.0)
        bet_kwh_per_km = st.number_input("BET year-1 full-loaded kWh/km", value=base_betc.full_loaded_kwh_per_km_year1, step=0.01, format="%.3f")
        battery_lifetime_cycles = st.number_input("Battery lifetime cycles", value=float(base_betc.battery_lifetime_cycles), step=100.0)

    with st.sidebar.expander("AEaaS granular savings", expanded=False):
        st.caption("These savings represent the percentage of cost an AEaaS provider has compared with an individual fleet manager. 0.9 = 90% cost saving.")
        aeaas_glider_cost_factor = st.number_input("Glider cost saving", value=base_shared.aeaas_glider_cost_factor, step=0.01, format="%.2f")
        aeaas_insurance_cost_factor = st.number_input("Insurance cost saving", value=base_shared.aeaas_insurance_cost_factor, step=0.01, format="%.2f")
        aeaas_station_capex_factor = st.number_input("Station CAPEX saving", value=base_shared.aeaas_station_capex_factor, step=0.01, format="%.2f")
        aeaas_site_capex_factor = st.number_input("Site CAPEX saving", value=base_shared.aeaas_site_capex_factor, step=0.01, format="%.2f")
        aeaas_battery_depr_factor = st.number_input("Battery depreciation saving", value=base_shared.aeaas_battery_depr_factor, step=0.01, format="%.2f")
        aeaas_battery_service_factor = st.number_input("Battery service saving", value=base_shared.aeaas_battery_service_factor, step=0.01, format="%.2f")
        aeaas_battery_rent_factor = st.number_input("Battery rent saving", value=base_shared.aeaas_battery_rent_factor, step=0.01, format="%.2f")
        aeaas_fixed_swapping_fee_factor = st.number_input("Fixed swapping fee saving", value=base_shared.aeaas_fixed_swapping_fee_factor, step=0.01, format="%.2f")
        aeaas_energy_cost_factor = st.number_input("Energy cost saving", value=base_shared.aeaas_energy_cost_factor, step=0.01, format="%.2f")

    shared = replace(
        base_shared,
        years=years,
        discount_rate=discount_rate,
        full_loaded_km_per_day=full_loaded_km_per_day,
        bet_subsidy=bet_subsidy,
        diesel_public_price_per_l=diesel_public_price_per_l,
        diesel_depot_price_per_l=diesel_depot_price_per_l,
        bet_depot_energy_price_per_kwh=bet_depot_energy_price_per_kwh,
        bet_public_energy_price_per_kwh=bet_public_energy_price_per_kwh,
        peak_price_per_kwh=peak_price_per_kwh,
        off_peak_price_per_kwh=off_peak_price_per_kwh,
        off_peak_share=off_peak_share,
        electricity_margin=electricity_margin,
        aeaas_glider_cost_factor=aeaas_glider_cost_factor,
        aeaas_insurance_cost_factor=aeaas_insurance_cost_factor,
        aeaas_station_capex_factor=aeaas_station_capex_factor,
        aeaas_site_capex_factor=aeaas_site_capex_factor,
        aeaas_battery_depr_factor=aeaas_battery_depr_factor,
        aeaas_battery_service_factor=aeaas_battery_service_factor,
        aeaas_battery_rent_factor=aeaas_battery_rent_factor,
        aeaas_fixed_swapping_fee_factor=aeaas_fixed_swapping_fee_factor,
        aeaas_energy_cost_factor=aeaas_energy_cost_factor,
    )
    diesel = replace(base_diesel, capex=diesel_capex, fuel_economy_full_loaded_year1_l_per_km=diesel_l_per_km)
    betc = replace(
        base_betc,
        glider_capex=glider_capex,
        battery_capacity_kwh=betc_battery_capacity,
        battery_price_per_kwh=bet_battery_price,
        full_loaded_kwh_per_km_year1=bet_kwh_per_km,
        battery_lifetime_cycles=battery_lifetime_cycles,
        battery_recycle_ratio=battery_recycle_ratio
    )
    bets = replace(
        base_bets,
        glider_capex=glider_capex,
        battery_price_per_kwh=bet_battery_price,
        full_loaded_kwh_per_km_year1=bet_kwh_per_km,
        battery_lifetime_cycles=battery_lifetime_cycles,
        battery_recycle_ratio=battery_recycle_ratio
    )
    return shared, diesel, betc, bets, asset_manager_margin


def build_uncertainty_overrides(shared, diesel, betc, bets):
    """Allow users to override Monte Carlo Min/Max bounds in the sidebar."""
    overrides = {}

    with st.sidebar.expander("Monte Carlo uncertainty ranges", expanded=False):
        st.caption(
            "Change Min/Max values used for triangular Monte Carlo sampling. "
            "Mode still follows the current control input values."
        )

        default_specs = get_uncertainty_specs(shared, diesel, betc, bets)
        for spec in default_specs:
            var = spec["variable"]
            label = get_pretty_label(var)

            st.markdown(f"**{label}**")
            col_min, col_max = st.columns(2)
            left_value = col_min.number_input(
                "Min",
                value=float(spec["left"]),
                key=f"uncertainty_min_{var}",
                format="%.6f",
            )
            right_value = col_max.number_input(
                "Max",
                value=float(spec["right"]),
                key=f"uncertainty_max_{var}",
                format="%.6f",
            )

            overrides[var] = {"left": left_value, "right": right_value}

    return overrides

# -------------------------------
# App layout
# -------------------------------
st.title("Truck TCO Analysis")


shared, diesel, betc, bets, asset_manager_margin = build_inputs()
uncertainty_overrides = build_uncertainty_overrides(shared, diesel, betc, bets)
with st.expander("All current model input values", expanded=False):
    input_values = {
        "SharedInputs": asdict(shared),
        "DieselInputs": asdict(diesel),
        "BETCInputs": asdict(betc),
        "BETSInputs": asdict(bets),
        "asset_manager_margin": asset_manager_margin,
    }

    rows = []
    for group, values in input_values.items():
        if isinstance(values, dict):
            for name, value in values.items():
                rows.append({
                    "Parameter": name,
                    "Value": value,
                })
        else:
            rows.append({
                "Parameter": group,
                "Value": values,
            })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )
results = run_baseline_cached(asdict(shared), asdict(diesel), asdict(betc), asdict(bets), asset_manager_margin)
gaps = extract_tco_gaps(results)

st.divider()

st.markdown("### Deterministic TCO results")

col1, col2 , col3 = st.columns(3)

with col1:
    st.pyplot(
        fig_tco_comparison(results),
        use_container_width=True
    )

with col2:
    st.pyplot(
        fig_tco_gap(gaps),
        use_container_width=True
    )

with col3:
    st.pyplot(
        fig_tco_per_km_comparison(results),
        use_container_width=True
    )

st.divider()

with st.expander("Monte Carlo simulation parameter ranges", expanded=False):
    st.dataframe(uncertainty_table(shared, diesel, betc, bets, uncertainty_overrides), use_container_width=True, hide_index=True)

st.markdown("### TCO Results with Uncertainty")
mc_runs = 500
mc_seed = 42

with st.spinner("Running Monte Carlo simulation..."):
    mc_df, mc_summary_df, mc_probability_df, driver_df = run_mc_cached(int(mc_runs), int(mc_seed), asdict(shared), asdict(diesel), asdict(betc), asdict(bets), uncertainty_overrides)
st.pyplot(fig_monte_carlo_histograms(mc_df), use_container_width=True)
st.pyplot(fig_driver_bar(driver_df), use_container_width=True)
with st.expander("Summary and probability tables", expanded=False):
    left, right = st.columns(2)
    left.dataframe(mc_summary_df, use_container_width=True)
    right.dataframe(mc_probability_df, use_container_width=True)
    st.dataframe(driver_df, use_container_width=True)

st.divider()

st.markdown("### Independent one-at-a-time Monte Carlo")
ind_runs = 500
ind_seed = 42

with st.spinner("Running independent-variable Monte Carlo..."):
    indep_df, indep_summary_df = run_independent_mc_cached(int(ind_runs), int(ind_seed), asdict(shared), asdict(diesel), asdict(betc), asdict(bets), uncertainty_overrides)
    # Always show this one
st.pyplot(
    fig_independent_bets_vs_diesel_boxplot(indep_df),
    use_container_width=True
)
# Button to show the other two plots
if st.button("Show All Plots of TCOs and Gaps"):
    st.pyplot(
        fig_independent_tco_boxplots(indep_df),
        use_container_width=True
    )

    st.pyplot(
        fig_independent_gap_boxplots(indep_df),
        use_container_width=True
    )

with st.expander("Independent-variable spread ranking", expanded=False):
    st.dataframe(indep_summary_df, use_container_width=True)

st.divider()
st.markdown("### TCO Projection with Uncertainty")


start_year = 2026
end_year = 2040
proj_runs = 500

projection_df = run_projection_cached(asdict(shared), asdict(diesel), asdict(betc), asdict(bets), int(start_year), int(end_year))
with st.expander("Projected input/output table", expanded=False):
    st.dataframe(projection_df, use_container_width=True)



with st.spinner("Running projection Monte Carlo..."):
    projection_mc_df, projection_summary_df = run_projection_mc_cached(int(start_year), int(end_year), int(proj_runs), 42, asdict(shared), asdict(diesel), asdict(betc), asdict(bets), uncertainty_overrides)
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig_projection_uncertainty(projection_summary_df, "tco_discounted", "Projected TCO with Uncertainty", "TCO (£)"), use_container_width=True)
with col2:
    st.pyplot(fig_projection_uncertainty(projection_summary_df, "tco_per_km", "Projected TCO/km with Uncertainty", "TCO (£/km)"), use_container_width=True)
with st.expander("Projection MC summary table", expanded=False):
    st.dataframe(projection_summary_df, use_container_width=True)

st.markdown("#### AEaaS Cost with Uncertainty")
aeaas_col1, aeaas_col2 = st.columns([1, 2])
margin_runs = aeaas_col1.number_input("AEaaS MC runs", min_value=50, max_value=3000, value=500, step=50)
margin_text = aeaas_col2.text_input("Margins (comma-separated decimals)", value="0.00,0.05,0.10,0.15,0.20,0.25,0.30")
try:
    margin_tuple = tuple(float(x.strip()) for x in margin_text.split(",") if x.strip())
except ValueError:
    st.error("Please enter margins as comma-separated numbers, e.g. 0.00,0.05,0.10")
    margin_tuple = tuple()

if margin_tuple:
    with st.spinner("Running AEaaS margin uncertainty..."):
        margin_uncertainty_df, margin_summary_df = run_margin_mc_cached(margin_tuple, int(margin_runs), 42, asdict(shared), asdict(diesel), asdict(betc), asdict(bets), uncertainty_overrides)
    st.pyplot(fig_margin_cost(margin_summary_df), use_container_width=True)
    with st.expander("AEaaS margin summary table", expanded=False):
        st.dataframe(margin_summary_df, use_container_width=True)
