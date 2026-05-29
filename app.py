
from __future__ import annotations

from dataclasses import asdict, replace
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import tco_model as model

st.set_page_config(page_title="Truck TCO Streamlit App", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        padding: 14px 16px;
        border-radius: 16px;
    }
    .small-note {color:#64748b; font-size:0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_gbp(value: float, decimals: int = 0) -> str:
    return f"£{value:,.{decimals}f}"


def fmt_num(value, decimals: int = 4):
    if isinstance(value, float):
        return round(value, decimals)
    return value


def build_input_table(shared, diesel, betc, bets) -> pd.DataFrame:
    rows = []
    for group_name, obj in [
        ("SharedInputs", shared),
        ("DieselInputs", diesel),
        ("BETCInputs", betc),
        ("BETSInputs", bets),
    ]:
        for key, value in asdict(obj).items():
            rows.append(
                {
                    "Group": group_name,
                    "Parameter": key,
                    "Label": model.get_pretty_label(key),
                    "Value": fmt_num(value),
                }
            )
    return pd.DataFrame(rows)


def build_uncertainty_table(uncertainty_overrides=None) -> pd.DataFrame:
    rows = []
    for spec in model.get_uncertainty_specs(include_subsidy_uncertainty=True, uncertainty_overrides=uncertainty_overrides):
        rows.append(
            {
                "Variable": spec["variable"],
                "Label": model.get_pretty_label(spec["variable"]),
                "Target": spec["target_class"],
                "Min": spec["left"],
                "Mode": spec["mode"],
                "Max": spec["right"],
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_baseline(shared_dict, diesel_dict, betc_dict, bets_dict):
    shared = model.SharedInputs(**shared_dict)
    diesel = model.DieselInputs(**diesel_dict)
    betc = model.BETCInputs(**betc_dict)
    bets = model.BETSInputs(**bets_dict)
    return model.run_model(shared, diesel, betc, bets)


@st.cache_data(show_spinner=False)
def cached_sensitivity():
    return model.run_multiple_sensitivity_analyses(model.sensitivity_specs)


@st.cache_data(show_spinner=False)
def cached_heatmap_data():
    shared = model.SharedInputs()
    bets = model.BETSInputs(battery_recycle_value_ratio=shared.battery_recycle_value_ratio)
    baas_grid_df = model.run_baas_viability_grid(shared=shared, bets_inp=bets)
    tco_gap_df = model.run_baas_utilisation_tco_gap_grid(shared=shared, bets_inp=bets)
    utilisation_grid_df = model.run_baas_utilisation_viability_grid(
        shared=shared,
        bets_inp=bets,
        expected_station_utilisations=np.arange(0.20, 0.50, 0.10),
        fixed_swapping_fee=3.0,
    )
    return baas_grid_df, tco_gap_df, utilisation_grid_df


@st.cache_data(show_spinner=False)
def cached_mc(n_runs: int, random_seed: int, uncertainty_overrides_json: str):
    model.set_uncertainty_overrides(json.loads(uncertainty_overrides_json) if uncertainty_overrides_json else {})
    mc_df = model.run_monte_carlo_simulation_with_and_without_subsidy(
        n_runs=n_runs,
        random_seed=random_seed,
    )
    summary_df, probability_df = model.summarize_monte_carlo_results(mc_df)
    indep_df = model.run_independent_variable_monte_carlo_with_and_without_subsidy(
        n_runs=n_runs,
        random_seed=random_seed,
    )
    return mc_df, summary_df, probability_df, indep_df


@st.cache_data(show_spinner=False)
def cached_margin(n_runs: int, random_seed: int, uncertainty_overrides_json: str):
    model.set_uncertainty_overrides(json.loads(uncertainty_overrides_json) if uncertainty_overrides_json else {})
    margin_uncertainty_df = model.run_margin_sweep_with_and_without_subsidy_uncertainty(
        margins=np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
        n_runs=n_runs,
        random_seed=random_seed,
    )
    return model.summarize_margin_uncertainty(margin_uncertainty_df)


@st.cache_data(show_spinner=False)
def cached_projection(start_year: int, end_year: int, n_runs: int, random_seed: int, uncertainty_overrides_json: str):
    model.set_uncertainty_overrides(json.loads(uncertainty_overrides_json) if uncertainty_overrides_json else {})
    projection_mc_df = model.run_projection_monte_carlo_with_and_without_subsidy(
        start_year=start_year,
        end_year=end_year,
        n_runs=n_runs,
        random_seed=random_seed,
    )
    summary_tco = model.summarize_projection_uncertainty(
        projection_mc_df,
        metric_cols=[
            "diesel_tco_discounted",
            "betc_tco_discounted",
            "bets_tco_discounted",
        ],
    )
    return summary_tco


# -------------------------------
# Sidebar inputs
# -------------------------------
st.sidebar.header("Controls")

base_shared = model.SharedInputs()
base_diesel = model.DieselInputs()
base_betc = model.BETCInputs(battery_recycle_value_ratio=base_shared.battery_recycle_value_ratio)
base_bets = model.BETSInputs(battery_recycle_value_ratio=base_shared.battery_recycle_value_ratio)

with st.sidebar.expander("General / operation", expanded=True):
    years = st.number_input("Analysis horizon (years)", min_value=1, max_value=20, value=base_shared.years, step=1)
    discount_rate = st.number_input("Discount rate", min_value=0.0, max_value=0.5, value=base_shared.discount_rate, step=0.01, format="%.3f")
    full_loaded_km_per_day = st.number_input("Full-loaded km per day", min_value=1.0, value=base_shared.full_loaded_km_per_day, step=10.0)
    operational_days_per_year = st.number_input("Operational days per year", min_value=1, max_value=365, value=base_shared.operational_days_per_year, step=1)
    shift_per_day = st.number_input("Shift per day", min_value=0.1, max_value=5.0, value=base_shared.shift_per_day, step=0.1)

with st.sidebar.expander("Financing / subsidy", expanded=False):
    cost_of_capital = st.number_input("Fleet cost of capital", min_value=0.0, max_value=0.5, value=base_shared.cost_of_capital, step=0.01, format="%.3f")
    upfront_payment_percentage = st.number_input("Upfront payment percentage", min_value=0.0, max_value=1.0, value=base_shared.upfront_payment_percentage, step=0.05, format="%.2f")
    loan_term_years = st.number_input("Loan term years", min_value=1, max_value=20, value=base_shared.loan_term_years, step=1)
    aeaas_cost_of_capital = st.number_input("AEaaS cost of capital", min_value=0.0, max_value=0.5, value=base_shared.aeaas_cost_of_capital, step=0.01, format="%.3f")
    bet_subsidy = st.number_input("BET purchase subsidy", min_value=0.0, value=base_shared.bet_subsidy, step=1000.0)

with st.sidebar.expander("Energy prices", expanded=False):
    diesel_depot_price_per_l = st.number_input("Diesel depot price (£/L)", min_value=0.0, value=base_shared.diesel_depot_price_per_l, step=0.01)
    diesel_public_price_per_l = st.number_input("Diesel public price (£/L)", min_value=0.0, value=base_shared.diesel_public_price_per_l, step=0.01)
    bet_depot_energy_price_per_kwh = st.number_input("BET depot electricity price (£/kWh)", min_value=0.0, value=base_shared.bet_depot_energy_price_per_kwh, step=0.01)
    bet_public_energy_price_per_kwh = st.number_input("BET public electricity price (£/kWh)", min_value=0.0, value=base_shared.bet_public_energy_price_per_kwh, step=0.01)
    peak_price_per_kwh = st.number_input("BaaS provider peak electricity price (£/kWh)", min_value=0.0, value=base_shared.peak_price_per_kwh, step=0.01)
    off_peak_price_per_kwh = st.number_input("BaaS provider off-peak electricity price (£/kWh)", min_value=0.0, value=base_shared.off_peak_price_per_kwh, step=0.01)
    off_peak_share = st.number_input("Off-peak share", min_value=0.0, max_value=1.0, value=base_shared.off_peak_share, step=0.05)
    electricity_margin = st.number_input("BaaS electricity margin", min_value=0.0, max_value=3.0, value=base_shared.electricity_margin, step=0.05)

with st.sidebar.expander("Vehicle / battery", expanded=False):
    diesel_capex = st.number_input("Diesel CAPEX", min_value=0.0, value=base_diesel.capex, step=1000.0)
    glider_capex = st.number_input("Electric glider CAPEX", min_value=0.0, value=base_betc.glider_capex, step=1000.0)
    battery_price_per_kwh = st.number_input("Battery price (£/kWh)", min_value=0.0, value=base_betc.battery_price_per_kwh, step=5.0)
    betc_battery_capacity = st.number_input("BET-C battery capacity (kWh)", min_value=1.0, value=base_betc.battery_capacity_kwh, step=10.0)
    bets_battery_pack_capacity = st.number_input("BET-S pack capacity (kWh)", min_value=1.0, value=base_bets.battery_pack_capacity_kwh, step=10.0)
    battery_lifetime_cycles = st.number_input("Battery lifetime cycles", min_value=1.0, value=base_betc.battery_lifetime_cycles, step=100.0)
    full_loaded_kwh_per_km_year1 = st.number_input("BET full-loaded kWh/km in year 1", min_value=0.1, value=base_betc.full_loaded_kwh_per_km_year1, step=0.05)

with st.sidebar.expander("BET-S station", expanded=False):
    expected_station_utilisation = st.number_input("Expected station utilisation", min_value=0.01, max_value=1.0, value=base_bets.expected_station_utilisation, step=0.05)
    station_capex = st.number_input("Station CAPEX", min_value=0.0, value=base_bets.station_capex, step=50000.0)
    site_capex = st.number_input("Site CAPEX", min_value=0.0, value=base_bets.site_capex, step=50000.0)
    station_annual_staff_costs = st.number_input("Station annual staff costs", min_value=0.0, value=base_bets.station_annual_staff_costs, step=5000.0)
    station_annual_other_service_costs = st.number_input("Station annual other service costs", min_value=0.0, value=base_bets.station_annual_other_service_costs, step=1000.0)
    swapping_fee_flat = st.number_input("Fixed swapping fee (£/swap)", min_value=0.0, value=base_bets.swapping_fee_flat, step=0.5)

with st.sidebar.expander("AEaaS granular savings", expanded=False):
    aeaas_glider_cost_factor = st.number_input("AEaaS glider cost factor", min_value=0.0, max_value=2.0, value=base_shared.aeaas_glider_cost_factor, step=0.05, format="%.2f")
    aeaas_insurance_cost_factor = st.number_input("AEaaS insurance cost factor", min_value=0.0, max_value=2.0, value=base_shared.aeaas_insurance_cost_factor, step=0.05, format="%.2f")
    aeaas_annual_service_cost_factor = st.number_input("AEaaS annual service cost factor", min_value=0.0, max_value=2.0, value=base_shared.aeaas_annual_service_cost_factor, step=0.05, format="%.2f")
    aeaas_station_capex_factor = st.number_input("AEaaS station CAPEX factor", min_value=0.0, max_value=2.0, value=base_shared.aeaas_station_capex_factor, step=0.05, format="%.2f")
    aeaas_station_opex_factor = st.number_input("AEaaS station OPEX factor", min_value=0.0, max_value=2.0, value=base_shared.aeaas_station_opex_factor, step=0.05, format="%.2f")
    aeaas_battery_depr_factor = st.number_input("AEaaS battery depreciation factor", min_value=0.0, max_value=2.0, value=base_shared.aeaas_battery_depr_factor, step=0.05, format="%.2f")
    aeaas_battery_service_factor = st.number_input("AEaaS battery service factor", min_value=0.0, max_value=2.0, value=base_shared.aeaas_battery_service_factor, step=0.05, format="%.2f")
    aeaas_battery_rent_factor = st.number_input("AEaaS battery rent factor", min_value=0.0, max_value=2.0, value=base_shared.aeaas_battery_rent_factor, step=0.05, format="%.2f")
    aeaas_fixed_swapping_fee_factor = st.number_input("AEaaS fixed swapping fee factor", min_value=0.0, max_value=2.0, value=base_shared.aeaas_fixed_swapping_fee_factor, step=0.05, format="%.2f")
    aeaas_energy_cost_factor = st.number_input("AEaaS energy cost factor", min_value=0.0, max_value=2.0, value=base_shared.aeaas_energy_cost_factor, step=0.05, format="%.2f")

with st.sidebar.expander("Monte Carlo / projection settings", expanded=False):
    mc_runs = st.number_input("Monte Carlo runs", min_value=50, max_value=5000, value=500, step=50)
    random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    projection_start_year = st.number_input("Projection start year", min_value=2020, max_value=2050, value=2026, step=1)
    projection_end_year = st.number_input("Projection end year", min_value=2021, max_value=2060, value=2040, step=1)

with st.sidebar.expander("Monte Carlo uncertainty ranges", expanded=False):
    st.caption("Edit Min / Mode / Max. These ranges are used by full MC, independent MC, AEaaS margin uncertainty, and projected TCO uncertainty.")
    uncertainty_editor_df = build_uncertainty_table()[["Variable", "Label", "Target", "Min", "Mode", "Max"]]
    edited_uncertainty_df = st.data_editor(
        uncertainty_editor_df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=["Variable", "Label", "Target"],
    )

uncertainty_overrides = {
    str(row["Variable"]): {"left": float(row["Min"]), "mode": float(row["Mode"]), "right": float(row["Max"])}
    for _, row in edited_uncertainty_df.iterrows()
}
uncertainty_overrides_json = json.dumps(uncertainty_overrides, sort_keys=True)
model.set_uncertainty_overrides(uncertainty_overrides)

shared = replace(
    base_shared,
    years=int(years),
    discount_rate=float(discount_rate),
    full_loaded_km_per_day=float(full_loaded_km_per_day),
    operational_days_per_year=int(operational_days_per_year),
    shift_per_day=float(shift_per_day),
    cost_of_capital=float(cost_of_capital),
    upfront_payment_percentage=float(upfront_payment_percentage),
    loan_term_years=int(loan_term_years),
    aeaas_cost_of_capital=float(aeaas_cost_of_capital),
    bet_subsidy=float(bet_subsidy),
    diesel_depot_price_per_l=float(diesel_depot_price_per_l),
    diesel_public_price_per_l=float(diesel_public_price_per_l),
    bet_depot_energy_price_per_kwh=float(bet_depot_energy_price_per_kwh),
    bet_public_energy_price_per_kwh=float(bet_public_energy_price_per_kwh),
    peak_price_per_kwh=float(peak_price_per_kwh),
    off_peak_price_per_kwh=float(off_peak_price_per_kwh),
    off_peak_share=float(off_peak_share),
    electricity_margin=float(electricity_margin),
    aeaas_glider_cost_factor=float(aeaas_glider_cost_factor),
    aeaas_insurance_cost_factor=float(aeaas_insurance_cost_factor),
    aeaas_annual_service_cost_factor=float(aeaas_annual_service_cost_factor),
    aeaas_station_capex_factor=float(aeaas_station_capex_factor),
    aeaas_station_opex_factor=float(aeaas_station_opex_factor),
    aeaas_battery_depr_factor=float(aeaas_battery_depr_factor),
    aeaas_battery_service_factor=float(aeaas_battery_service_factor),
    aeaas_battery_rent_factor=float(aeaas_battery_rent_factor),
    aeaas_fixed_swapping_fee_factor=float(aeaas_fixed_swapping_fee_factor),
    aeaas_energy_cost_factor=float(aeaas_energy_cost_factor),
)

diesel = replace(base_diesel, capex=float(diesel_capex))
betc = replace(
    base_betc,
    glider_capex=float(glider_capex),
    battery_capacity_kwh=float(betc_battery_capacity),
    battery_price_per_kwh=float(battery_price_per_kwh),
    battery_lifetime_cycles=float(battery_lifetime_cycles),
    full_loaded_kwh_per_km_year1=float(full_loaded_kwh_per_km_year1),
)
bets = replace(
    base_bets,
    glider_capex=float(glider_capex),
    battery_pack_capacity_kwh=float(bets_battery_pack_capacity),
    battery_price_per_kwh=float(battery_price_per_kwh),
    battery_lifetime_cycles=float(battery_lifetime_cycles),
    full_loaded_kwh_per_km_year1=float(full_loaded_kwh_per_km_year1),
    expected_station_utilisation=float(expected_station_utilisation),
    station_capex=float(station_capex),
    site_capex=float(site_capex),
    station_annual_staff_costs=float(station_annual_staff_costs),
    station_annual_other_service_costs=float(station_annual_other_service_costs),
    swapping_fee_flat=float(swapping_fee_flat),
)

# Keep BET-C and BET-S residual assumptions aligned with SharedInputs.
betc = replace(betc, battery_recycle_value_ratio=shared.battery_recycle_value_ratio)
bets = replace(bets, battery_recycle_value_ratio=shared.battery_recycle_value_ratio)

results = cached_baseline(asdict(shared), asdict(diesel), asdict(betc), asdict(bets))
gaps = model.extract_tco_gaps(results)

st.title("Truck TCO Analysis")

with st.expander("Current model inputs", expanded=False):
    st.dataframe(build_input_table(shared, diesel, betc, bets), use_container_width=True)

with st.expander("Monte Carlo uncertainty inputs", expanded=False):
    st.dataframe(build_uncertainty_table(uncertainty_overrides), use_container_width=True)

# -------------------------------
# 1. Baseline Results
# -------------------------------
st.header("1. Deterministic TCO Results")
col1, col2, col3 = st.columns(3)
col1.metric("Diesel discounted TCO", fmt_gbp(results["diesel"]["tco_discounted"]))
col2.metric("BET-C discounted TCO", fmt_gbp(results["bet_c"]["tco_discounted_recycle"]))
col3.metric("BET-S discounted TCO", fmt_gbp(results["bet_s"]["tco_discounted_recycle"]))

col1, col2, col3 = st.columns(3)
col1.metric("BET-C - Diesel", fmt_gbp(gaps["bet_c_vs_diesel"]))
col2.metric("BET-S - Diesel", fmt_gbp(gaps["bet_s_vs_diesel"]))
col3.metric("BET-S - BET-C", fmt_gbp(gaps["bet_s_vs_bet_c"]))

det_cols = st.columns(4)
with det_cols[0]:
    st.pyplot(model.plot_tco_comparison(results), use_container_width=True)
with det_cols[1]:
    st.pyplot(model.plot_tco_gap(results), use_container_width=True)
with det_cols[2]:
    st.pyplot(model.plot_tco_per_km_comparison(results), use_container_width=True)
with det_cols[3]:
    st.pyplot(model.plot_tco_per_km_gap(results), use_container_width=True)

# -------------------------------
# 2. Sensitivity
# -------------------------------
st.header("2. Sensitivity Analysis")

sensitivity_results = cached_sensitivity()
for start_idx in range(0, len(sensitivity_results), 3):
    cols = st.columns(3)
    for col, sensitivity_result in zip(cols, sensitivity_results[start_idx:start_idx + 3]):
        with col:
            st.pyplot(model.plot_sensitivity_bar(sensitivity_result))

# -------------------------------
# 3. Heatmap
# -------------------------------
st.header("3. Heatmaps from a BaaS Provider's Perspective")
baas_grid_df, tco_gap_df, utilisation_grid_df = cached_heatmap_data()
st.subheader("BaaS IRR / payback heatmaps 1")
st.pyplot(model.plot_baas_irr_payback_heatmaps(baas_grid_df))
st.subheader("BaaS IRR / payback heatmaps 2")
st.pyplot(model.plot_baas_utilisation_irr_payback_heatmaps(utilisation_grid_df))
st.subheader("TCO Gap between BETs and Diesel Trucks under Different BaaS Price Scenarios")
st.pyplot(model.plot_baas_utilisation_tco_gap_heatmaps(tco_gap_df))


# -------------------------------
# 4. TCO Results under Uncertainty Using Monte Carlo Simulation
# -------------------------------
st.header("4. TCO Results under Uncertainty Using Monte Carlo Simulation")
mc_df, mc_summary_df, mc_probability_df, indep_mc_df = cached_mc(int(mc_runs), int(random_seed), uncertainty_overrides_json)

with st.expander("Monte Carlo summary", expanded=False):
    st.dataframe(mc_summary_df, use_container_width=True)
st.subheader("Probability summary")
st.dataframe(mc_probability_df, use_container_width=True)

hist_fig = model.plot_monte_carlo_histograms_by_scenario(mc_df)

st.pyplot(hist_fig, use_container_width=True)

# Driver correlations are shown separately for each subsidy scenario.
input_columns = [
    "expected_station_utilisation",
    "discount_rate",
    "full_loaded_km_per_day",
    "peak_price_per_kwh",
    "off_peak_share",
    "bet_depot_energy_price_per_kwh",
    "bet_public_energy_price_per_kwh",
    "full_loaded_kwh_per_km_year1",
    "battery_recycle_value_ratio",
    "glider_capex",
    "battery_lifetime_cycles",
    "unladen_energy_saving",
    "battery_capacity_kwh",
    "battery_price_per_kwh",
    "expected_annual_return_on_battery_renting",
    "electricity_margin",
    "bet_depot_share",
    "bet_subsidy",
]
available_inputs = [c for c in input_columns if c in mc_df.columns]
st.subheader("Drivers of BET-S - Diesel gap")
# Match the 2405 workflow: calculate one driver-ranking chart from the full MC output,
# rather than splitting by subsidy scenario. This keeps bet_subsidy as a meaningful input driver.
drivers_df = model.get_drivers_of_gap(
    mc_df,
    gap_column="gap_bet_s_diesel",
    input_columns=available_inputs,
)
st.pyplot(model.plot_drivers(drivers_df, gap_name="BET-S - Diesel"), use_container_width=True)

st.subheader("Independent Uncertainty one-at-a-time Monte Carlo Simulation")
if "subsidy_scenario" in indep_mc_df.columns:
    for scenario, sub_df in indep_mc_df.groupby("subsidy_scenario"):
        st.markdown(f"**{scenario}**")
        st.pyplot(model.plot_independent_tco_boxplots(sub_df))
        st.pyplot(model.plot_independent_gap_boxplots(sub_df))
        st.pyplot(model.plot_independent_bets_vs_diesel_boxplot(sub_df))
else:
    st.pyplot(model.plot_independent_tco_boxplots(indep_mc_df))
    st.pyplot(model.plot_independent_gap_boxplots(indep_mc_df))
    st.pyplot(model.plot_independent_bets_vs_diesel_boxplot(indep_mc_df))

# -------------------------------
# 5. AEaaS
# -------------------------------
st.header("5. Asset-and-Energy-as-a-Service")
st.markdown("An asset manager buys battery electric trucks, constructs or outsources energy facilities and provides truck leasing and energy services. An asset manager is assumed to buy the assets and having energy facilities at a lower cost due to the economy of scale and can set a target margin of their business. Fleet managers who require trucks and energy services pay for their actual usage of trucks and energy services.") 

margin_summary_df = cached_margin(int(mc_runs), int(random_seed), uncertainty_overrides_json)
if "subsidy_scenario" in margin_summary_df.columns:
    margin_cols = st.columns(2)
    for col, (scenario, sub_df) in zip(margin_cols, margin_summary_df.groupby("subsidy_scenario")):
        with col:
            st.markdown(f"**{scenario}**")
            st.pyplot(model.plot_margin_vs_freight_all_in_per_km_with_uncertainty(sub_df.sort_values("asset_manager_margin"), title_suffix=f"- {scenario}"))
            st.pyplot(model.plot_margin_vs_gap_with_uncertainty(sub_df.sort_values("asset_manager_margin"), title_suffix=f"- {scenario}"))
else:
    st.pyplot(model.plot_margin_vs_freight_all_in_per_km_with_uncertainty(margin_summary_df))
    st.pyplot(model.plot_margin_vs_gap_with_uncertainty(margin_summary_df))

# -------------------------------
# 6. Projection
# -------------------------------
st.header("6. TCO Projection")
if int(projection_end_year) <= int(projection_start_year):
    st.warning("Projection end year must be later than start year.")
else:
    projection_summary_tco = cached_projection(
        int(projection_start_year),
        int(projection_end_year),
        int(mc_runs),
        int(random_seed),
        uncertainty_overrides_json,
    )
    st.subheader("TCO Projection under Uncertainty")
    if "subsidy_scenario" in projection_summary_tco.columns:
        projection_cols = st.columns(2)
        for col, (scenario, sub_df) in zip(projection_cols, projection_summary_tco.groupby("subsidy_scenario")):
            with col:
                st.pyplot(model.plot_projection_with_uncertainty(sub_df.sort_values("year"), title_suffix=f"- {scenario}"))
    else:
        st.pyplot(model.plot_projection_with_uncertainty(projection_summary_tco))
