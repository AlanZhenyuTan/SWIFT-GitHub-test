try
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataclasses import replace

from tco_model import (
    SharedInputs,
    DieselInputs,
    BETCInputs,
    BETSInputs,
    run_model,
    extract_tco_gaps,
    run_tco_projection,
    run_projection_monte_carlo,
    summarize_projection_uncertainty,
    run_margin_sweep_with_uncertainty,
    run_monte_carlo_simulation,
    summarize_monte_carlo_results,
    get_drivers_of_gap,
    get_pretty_label,
    run_independent_variable_monte_carlo,
    summarize_independent_effect_spread,
)


st.set_page_config(page_title="Truck TCO App", layout="wide")


# -------------------------------
# Plot helpers for Streamlit
# -------------------------------
def fig_tco_comparison(results):
    labels = ["Diesel", "BET-C", "BET-S"]
    values = [
        results["diesel"]["tco_5y_discounted"],
        results["bet_c"]["tco_5y_discounted_recycle"],
        results["bet_s"]["tco_5y_discounted_recycle"],
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values)
    ax.set_title("5-Year Discounted TCO Comparison")
    ax.set_ylabel("TCO (£)")
    ax.set_xlabel("Truck Type")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    return fig


def fig_tco_gap(gaps):
    labels = ["BET-C - Diesel", "BET-S - Diesel", "BET-S - BET-C"]
    values = [
        gaps["bet_c_vs_diesel"],
        gaps["bet_s_vs_diesel"],
        gaps["bet_s_vs_bet_c"],
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values)
    ax.set_title("TCO Gaps")
    ax.set_ylabel("Difference (£)")
    ax.set_xlabel("Comparison")
    ax.axhline(0, linewidth=1)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )

    fig.tight_layout()
    return fig


def fig_projection(df, ycols, labels, title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for col, label in zip(ycols, labels):
        ax.plot(df["year"], df[col], marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Purchase Year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(df["year"])
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_projection_uncertainty(summary_df):
    specs = [
        ("diesel_tco_5y_discounted", "Diesel"),
        ("betc_tco_5y_discounted", "BET-C"),
        ("bets_tco_5y_discounted", "BET-S"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for metric, label in specs:
        years = summary_df["year"]
        p5 = summary_df[f"{metric}_p5"]
        p50 = summary_df[f"{metric}_p50"]
        p95 = summary_df[f"{metric}_p95"]
        ax.plot(years, p50, marker="o", label=f"{label} median")
        ax.fill_between(years, p5, p95, alpha=0.2)

    ax.set_title("Projected 5-Year Discounted TCO with Uncertainty Bands")
    ax.set_xlabel("Purchase Year")
    ax.set_ylabel("5-Year Discounted TCO (£)")
    ax.set_xticks(summary_df["year"])
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    return fig


def summarize_margin_uncertainty(df):
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


def fig_margin_cost(summary_df):
    x = summary_df["asset_manager_margin"] * 100
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, summary_df["diesel_p50"], marker="o", label="Diesel own TCO per km (median)")
    ax.fill_between(x, summary_df["diesel_p5"], summary_df["diesel_p95"], alpha=0.2)
    ax.plot(x, summary_df["bets_p50"], marker="o", label="BET-S AEaaS per km (median)")
    ax.fill_between(x, summary_df["bets_p5"], summary_df["bets_p95"], alpha=0.2)
    ax.set_xlabel("Asset-manager margin (%)")
    ax.set_ylabel("Cost (£/km)")
    ax.set_title("Effect of Asset-manager Margin on Freight Cost per km")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_margin_gap(summary_df):
    x = summary_df["asset_manager_margin"] * 100
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, summary_df["gap_p50"], marker="o", label="BET-S AEaaS - Diesel (median)")
    ax.fill_between(x, summary_df["gap_p5"], summary_df["gap_p95"], alpha=0.2)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Asset-manager margin (%)")
    ax.set_ylabel("Cost Gap (£/km)")
    ax.set_title("Effect of Asset-manager Margin on BET-S AEaaS - Diesel Gap")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_driver_bar(driver_df, gap_name="BET-S - Diesel"):
    labels = [get_pretty_label(v) for v in driver_df["variable"]]
    values = driver_df["correlation_with_gap"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, values)
    ax.axhline(0, linewidth=1)
    ax.set_title(f"Drivers of {gap_name}")
    ax.set_xlabel("Input variable")
    ax.set_ylabel("Correlation with gap")
    ax.tick_params(axis="x", rotation=30)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    fig.tight_layout()
    return fig


def fig_independent_bets_vs_diesel_boxplot(df):
    variable_order = list(df["variable"].drop_duplicates())
    data = [df.loc[df["variable"] == var, "gap_bet_s_diesel"].dropna() for var in variable_order]
    positions = list(range(1, len(variable_order) + 1))

    fig, ax = plt.subplots(figsize=(14, 5.5))
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgray")
    ax.set_xticks(positions)
    ax.set_xticklabels([get_pretty_label(v) for v in variable_order], rotation=35, ha="right")
    ax.set_ylabel("BET-S - Diesel TCO Gap (£)")
    ax.set_title("Independent effect of each uncertain variable on BET-S - Diesel gap")
    ax.axhline(0, linewidth=1)
    fig.tight_layout()
    return fig


# -------------------------------
# Cached runners
# -------------------------------
@st.cache_data(show_spinner=False)
def run_baseline_cached(shared_dict, diesel_dict, betc_dict, bets_dict, asset_manager_margin):
    shared = SharedInputs(**shared_dict)
    diesel = DieselInputs(**diesel_dict)
    betc = BETCInputs(**betc_dict)
    bets = BETSInputs(**bets_dict)
    results = run_model(shared=shared, diesel_inp=diesel, betc_inp=betc, bets_inp=bets, asset_manager_margin=asset_manager_margin)
    gaps = extract_tco_gaps(results)
    return results, gaps


@st.cache_data(show_spinner=False)
def run_projection_cached(shared_dict, diesel_dict, betc_dict, bets_dict, start_year, end_year):
    shared = SharedInputs(**shared_dict)
    diesel = DieselInputs(**diesel_dict)
    betc = BETCInputs(**betc_dict)
    bets = BETSInputs(**bets_dict)
    return run_tco_projection(
        start_year=start_year,
        end_year=end_year,
        shared=shared,
        diesel_inp=diesel,
        betc_inp=betc,
        bets_inp=bets,
    )


@st.cache_data(show_spinner=False)
def run_projection_mc_cached(start_year, end_year, n_runs, random_seed):
    df = run_projection_monte_carlo(
        start_year=start_year,
        end_year=end_year,
        n_runs=n_runs,
        random_seed=random_seed,
    )
    summary = summarize_projection_uncertainty(
        df,
        metric_cols=[
            "diesel_tco_5y_discounted",
            "betc_tco_5y_discounted",
            "bets_tco_5y_discounted",
        ],
    )
    return df, summary


@st.cache_data(show_spinner=False)
def run_margin_mc_cached(margins_tuple, n_runs, random_seed):
    df = run_margin_sweep_with_uncertainty(
        margins=np.array(margins_tuple),
        n_runs=n_runs,
        random_seed=random_seed,
    )
    return df, summarize_margin_uncertainty(df)


@st.cache_data(show_spinner=False)
def run_mc_cached(n_runs, random_seed):
    mc_df = run_monte_carlo_simulation(n_runs=n_runs, random_seed=random_seed)
    summary_df, probability_df = summarize_monte_carlo_results(mc_df)
    driver_df = get_drivers_of_gap(
        mc_df,
        gap_column="gap_bet_s_diesel",
        input_columns=[
            "discount_rate",
            "full_loaded_km_per_day",
            "peak_price_per_kwh",
            "off_peak_share",
            "bet_depot_energy_price_per_kwh",
            "bet_public_energy_price_per_kwh",
            "full_loaded_kwh_per_km_year1",
            "battery_recycle_ratio",
            "battery_lifetime_cycles",
            "unladen_energy_saving",
            "battery_capacity_kwh",
            "bet_subsidy",
        ],
    )
    return mc_df, summary_df, probability_df, driver_df


@st.cache_data(show_spinner=False)
def run_independent_mc_cached(n_runs, random_seed):
    df = run_independent_variable_monte_carlo(n_runs=n_runs, random_seed=random_seed)
    summary = summarize_independent_effect_spread(df)
    return df, summary


# -------------------------------
# Build input objects
# -------------------------------
def build_input_panels():
    base_shared = SharedInputs()
    base_diesel = DieselInputs()
    base_betc = BETCInputs(battery_recycle_ratio=base_shared.battery_recycle_value_ratio)
    base_bets = BETSInputs(battery_recycle_ratio=base_shared.battery_recycle_value_ratio)

    st.sidebar.header("Main controls")
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
        off_peak_share = st.slider("Off-peak share", min_value=0.0, max_value=1.0, value=float(base_shared.off_peak_share), step=0.01)
        electricity_margin = st.number_input("Electricity margin", min_value=0.0, value=base_shared.electricity_margin, step=0.01, format="%.2f")

    with st.sidebar.expander("Diesel inputs", expanded=False):
        diesel_capex = st.number_input("Diesel CAPEX (£)", value=base_diesel.capex, step=1000.0)
        diesel_service = st.number_input("Diesel annual service cost (£)", value=base_diesel.annual_service_cost, step=100.0)
        diesel_l_per_km = st.number_input("Diesel year-1 fuel economy (L/km)", value=base_diesel.fuel_economy_full_loaded_year1_l_per_km, step=0.01, format="%.3f")

    with st.sidebar.expander("BET-C inputs", expanded=False):
        betc_glider_capex = st.number_input("BET-C glider CAPEX (£)", value=base_betc.glider_capex, step=1000.0)
        betc_battery_capacity = st.number_input("BET-C battery capacity (kWh)", value=base_betc.battery_capacity_kwh, step=10.0)
        betc_battery_price = st.number_input("BET-C battery price (£/kWh)", value=base_betc.battery_price_per_kwh, step=1.0)
        betc_battery_lifetime_cycles = st.number_input("BET-C battery lifetime cycles", value=float(base_betc.battery_lifetime_cycles), step=100.0)
        betc_kwh_per_km = st.number_input("BET-C year-1 full-loaded kWh/km", value=base_betc.full_loaded_kwh_per_km_year1, step=0.01, format="%.3f")
        betc_service = st.number_input("BET-C annual service cost (£)", value=base_betc.annual_service_cost, step=100.0)

    with st.sidebar.expander("BET-S inputs", expanded=False):
        bets_glider_capex = st.number_input("BET-S glider CAPEX (£)", value=base_bets.glider_capex, step=1000.0)
        bets_battery_price = st.number_input("BET-S battery price (£/kWh)", value=base_bets.battery_price_per_kwh, step=1.0)
        bets_battery_lifetime_cycles = st.number_input("BET-S battery lifetime cycles", value=float(base_bets.battery_lifetime_cycles), step=100.0)
        bets_kwh_per_km = st.number_input("BET-S year-1 full-loaded kWh/km", value=base_bets.full_loaded_kwh_per_km_year1, step=0.01, format="%.3f")
        bets_service = st.number_input("BET-S annual battery service cost (£)", value=base_bets.annual_battery_service_cost, step=100.0)
        bets_rent = st.number_input("BET-S battery rent per month (£)", value=base_bets.battery_rent_per_month_ex_depreciation, step=10.0)
        bets_swap_fee = st.number_input("BET-S swapping fee flat (£)", value=base_bets.swapping_fee_flat, step=0.5)

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
    )

    diesel = replace(
        base_diesel,
        capex=diesel_capex,
        annual_service_cost=diesel_service,
        fuel_economy_full_loaded_year1_l_per_km=diesel_l_per_km,
    )

    betc = replace(
        base_betc,
        glider_capex=betc_glider_capex,
        battery_capacity_kwh=betc_battery_capacity,
        battery_price_per_kwh=betc_battery_price,
        battery_lifetime_cycles=betc_battery_lifetime_cycles,
        full_loaded_kwh_per_km_year1=betc_kwh_per_km,
        annual_service_cost=betc_service,
    )

    bets = replace(
        base_bets,
        glider_capex=bets_glider_capex,
        battery_price_per_kwh=bets_battery_price,
        battery_lifetime_cycles=bets_battery_lifetime_cycles,
        full_loaded_kwh_per_km_year1=bets_kwh_per_km,
        annual_battery_service_cost=bets_service,
        battery_rent_per_month_ex_depreciation=bets_rent,
        swapping_fee_flat=bets_swap_fee,
    )

    return shared, diesel, betc, bets, asset_manager_margin


# -------------------------------
# App layout
# -------------------------------
st.title("Truck TCO Streamlit App")
st.caption("Baseline TCO, future projection, AEaaS, Monte Carlo, and driver analysis based on your current model code.")

shared, diesel, betc, bets, asset_manager_margin = build_input_panels()

results, gaps = run_baseline_cached(shared.__dict__, diesel.__dict__, betc.__dict__, bets.__dict__, asset_manager_margin)

st.subheader("Baseline results")
col1, col2, col3 = st.columns(3)
col1.metric("Diesel 5y discounted TCO", f"£{results['diesel']['tco_5y_discounted']:,.0f}")
col2.metric("BET-C 5y discounted TCO", f"£{results['bet_c']['tco_5y_discounted_recycle']:,.0f}")
col3.metric("BET-S 5y discounted TCO", f"£{results['bet_s']['tco_5y_discounted_recycle']:,.0f}")

col4, col5, col6 = st.columns(3)
col4.metric("Diesel discounted TCO/km", f"£{results['diesel']['tco_per_km_discounted']:.4f}")
col5.metric("BET-C discounted TCO/km", f"£{results['bet_c']['tco_per_km_discounted_recycle']:.4f}")
col6.metric("BET-S discounted TCO/km", f"£{results['bet_s']['tco_per_km_discounted_recycle']:.4f}")

with st.expander("Show detailed baseline outputs"):
    baseline_df = pd.DataFrame(
        {
            "Metric": [
                "5y discounted TCO",
                "Discounted TCO per km",
                "Discounted TCO per kWh",
            ],
            "Diesel": [
                results["diesel"]["tco_5y_discounted"],
                results["diesel"]["tco_per_km_discounted"],
                results["diesel"]["tco_per_kwh_discounted"],
            ],
            "BET-C": [
                results["bet_c"]["tco_5y_discounted_recycle"],
                results["bet_c"]["tco_per_km_discounted_recycle"],
                results["bet_c"]["tco_per_kwh_discounted_recycle"],
            ],
            "BET-S": [
                results["bet_s"]["tco_5y_discounted_recycle"],
                results["bet_s"]["tco_per_km_discounted_recycle"],
                results["bet_s"]["tco_per_kwh_discounted_recycle"],
            ],
        }
    )
    st.dataframe(baseline_df, use_container_width=True)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.pyplot(fig_tco_comparison(results))
with chart_col2:
    st.pyplot(fig_tco_gap(gaps))

st.subheader("Key gaps")
gap_col1, gap_col2, gap_col3 = st.columns(3)
gap_col1.metric("BET-C - Diesel", f"£{gaps['bet_c_vs_diesel']:,.0f}")
gap_col2.metric("BET-S - Diesel", f"£{gaps['bet_s_vs_diesel']:,.0f}")
gap_col3.metric("BET-S - BET-C", f"£{gaps['bet_s_vs_bet_c']:,.0f}")


# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Projection",
    "AEaaS Margin",
    "Monte Carlo",
    "Independent-variable MC",
])

with tab1:
    st.markdown("### Future TCO projection")
    proj_col1, proj_col2, proj_col3 = st.columns(3)
    start_year = proj_col1.number_input("Start year", min_value=2020, max_value=2040, value=2026, step=1)
    end_year = proj_col2.number_input("End year", min_value=2020, max_value=2045, value=2040, step=1)
    proj_runs = proj_col3.number_input("Projection MC runs", min_value=50, max_value=3000, value=300, step=50)

    projection_df = run_projection_cached(shared.__dict__, diesel.__dict__, betc.__dict__, bets.__dict__, start_year, end_year)
    st.dataframe(projection_df, use_container_width=True)

    st.pyplot(
        fig_projection(
            projection_df,
            ["diesel_tco_5y_discounted", "betc_tco_5y_discounted", "bets_tco_5y_discounted"],
            ["Diesel", "BET-C", "BET-S"],
            "Projected 5-Year Discounted TCO",
            "TCO (£)",
        )
    )
    st.pyplot(
        fig_projection(
            projection_df,
            ["diesel_tco_per_km", "betc_tco_per_km", "bets_tco_per_km"],
            ["Diesel", "BET-C", "BET-S"],
            "Projected Discounted TCO per km",
            "TCO (£/km)",
        )
    )
    st.pyplot(
        fig_projection(
            projection_df,
            ["diesel_tco_per_kwh", "betc_tco_per_kwh", "bets_tco_per_kwh"],
            ["Diesel", "BET-C", "BET-S"],
            "Projected Discounted TCO per kWh",
            "TCO (£/kWh)",
        )
    )

    if st.button("Run projection Monte Carlo"):
        with st.spinner("Running projection Monte Carlo..."):
            projection_mc_df, summary_tco = run_projection_mc_cached(start_year, end_year, int(proj_runs), 42)
        st.pyplot(fig_projection_uncertainty(summary_tco))
        with st.expander("Show projection MC summary table"):
            st.dataframe(summary_tco, use_container_width=True)

with tab2:
    st.markdown("### AEaaS margin analysis")
    margin_runs = st.number_input("AEaaS Monte Carlo runs", min_value=50, max_value=3000, value=500, step=50)
    margins = st.text_input(
        "Margins (comma-separated decimals)",
        value="0.00,0.05,0.10,0.15,0.20,0.25,0.30",
    )
    margin_tuple = tuple(float(x.strip()) for x in margins.split(",") if x.strip())

    if st.button("Run AEaaS margin uncertainty"):
        with st.spinner("Running AEaaS margin uncertainty..."):
            margin_uncertainty_df, margin_summary_df = run_margin_mc_cached(margin_tuple, int(margin_runs), 42)
        st.pyplot(fig_margin_cost(margin_summary_df))
        st.pyplot(fig_margin_gap(margin_summary_df))
        with st.expander("Show margin summary table"):
            st.dataframe(margin_summary_df, use_container_width=True)

with tab3:
    st.markdown("### Monte Carlo and driver analysis")
    mc_runs = st.number_input("Monte Carlo runs", min_value=50, max_value=5000, value=500, step=50)
    if st.button("Run Monte Carlo"):
        with st.spinner("Running Monte Carlo simulation..."):
            mc_df, mc_summary_df, mc_probability_df, driver_df = run_mc_cached(int(mc_runs), 42)

        st.markdown("#### Summary")
        st.dataframe(mc_summary_df, use_container_width=True)
        st.markdown("#### Probabilities")
        st.dataframe(mc_probability_df, use_container_width=True)
        st.markdown("#### Driver ranking")
        st.dataframe(driver_df, use_container_width=True)
        st.pyplot(fig_driver_bar(driver_df))

with tab4:
    st.markdown("### Independent-variable Monte Carlo")
    indep_runs = st.number_input("Independent-variable MC runs", min_value=50, max_value=5000, value=500, step=50)
    if st.button("Run independent-variable MC"):
        with st.spinner("Running independent-variable Monte Carlo..."):
            indep_mc_df, indep_summary_df = run_independent_mc_cached(int(indep_runs), 42)
        st.pyplot(fig_independent_bets_vs_diesel_boxplot(indep_mc_df))
        with st.expander("Show spread ranking table"):
            st.dataframe(indep_summary_df, use_container_width=True)
