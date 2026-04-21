
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from typing import Dict, List
import math
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

########################## Inputs ##########################################
# Shared parameters across all vehicle types 
@dataclass
class SharedInputs:
    years: int = 5
    discount_rate: float = 0.10
    full_loaded_km_per_day: float = 240.0
    unladen_ratio_to_full: float = 3 / 7
    operational_days_per_year: int = 292
    driver_hourly_pay: float = 15.78
    worked_hours_per_week: float = 48.0
    diesel_insurance: float = 10_000.0
    diesel_public_price_per_l: float = 1.48
    diesel_depot_price_per_l: float = 1.05
    diesel_depot_share: float = 0.80
    diesel_bunker_capex_per_l: float = 0.24
    diesel_expected_fleet_size: float = 51.0

    bet_insurance_markup: float = 0.20
    bet_depot_share: float = 0.80
    bet_depot_energy_price_per_kwh: float = 0.22
    bet_public_energy_price_per_kwh: float = 0.39
    bet_subsidy: float = 0.0
    battery_recycle_value_ratio: float = 0.10
    battery_capacity_bet_c_kwh: float = 621.0

    # BET-S specific shared energy pricing assumptions
    off_peak_share: float = 0.50
    peak_price_per_kwh: float = 0.20
    off_peak_price_per_kwh: float = 0.10
    electricity_margin: float = 0.39

    # AEaaS supplier discounts
    aeaas_glider_cost_factor: float = 0.90
    aeaas_baas_cost_factor: float = 0.90

# Diesel-specific technical and cost parameters
@dataclass
class DieselInputs:
    capex: float = 144_900.0
    fixed_depreciation_rate: float = 0.075
    variable_depreciation_per_km: float = 0.095
    annual_service_cost: float = 4_800.0
    lez_operation_percentage: float = 0.10
    lez_charge: float = 50.0
    refuels_per_day: float = 1.0
    fuel_economy_full_loaded_year1_l_per_km: float = 0.35
    fuel_economy_growth_rate: float = 0.011
    unladen_energy_saving: float = 0.25
    litre_to_kwh: float = 3.0

# BET-C (battery electric truck with fixed batteries) parameters
@dataclass
class BETCInputs:
    glider_capex: float = 130_000.0
    battery_capacity_kwh: float = 621.0
    battery_price_per_kwh: float = 148.0
    glider_fixed_depreciation_rate: float = 0.075
    glider_variable_depreciation_per_km: float = 0.095
    battery_eol_ratio: float = 0.50
    battery_recycle_ratio: float = 0.10 #residual value percentage
    truck_lifetime_years: float = 15.0
    battery_lifetime_cycles: float = 2000.0
    annual_service_cost: float = 4_200.0
    fuel_economy_growth_rate: float = 0.014
    full_loaded_kwh_per_km_year1: float = 1.37
    unladen_energy_saving: float = 0.25
    charger_capex_per_kwh: float = 0.12
    site_capex_per_kwh: float = 0.08
    expected_fleet_size: float = 51.0
    recharges_per_day: float = 1.0

# BET-S (battery swapping truck) parameters
@dataclass
class BETSInputs:
    battery_pack_capacity_kwh: float = 171.0
    battery_packs_per_truck: float = 3.0
    battery_price_per_kwh: float = 148.0
    battery_eol_ratio: float = 0.50
    battery_recycle_ratio: float = 0.10
    battery_lifetime_cycles: float = 2000.0
    annual_battery_service_cost: float = 4_200.0
    battery_rent_per_month_ex_depreciation: float = 100.0

    glider_capex: float = 130_000.0
    glider_fixed_depreciation_rate: float = 0.075
    glider_variable_depreciation_per_km: float = 0.095

    full_loaded_kwh_per_km_year1: float = 1.37
    fuel_economy_growth_rate: float = 0.005 #aging
    unladen_energy_saving: float = 0.25

    swaps_per_day: float = 1.0
    swapping_fee_flat: float = 3.0

    station_battery_bays: float = 24.0
    station_capex: float = 1_000_000.0
    site_capex: float = 1_000_000.0
    station_lifetime_years: float = 15.0
    max_station_service_capacity_trucks_per_day: float = 171.0
    expected_station_utilisation: float = 0.30

label_map = {
    "discount_rate": "Discount Rate",
    "full_loaded_km_per_day": "Full-loaded VKT per Day",
    "peak_price_per_kwh": "Peaktime Swapping Price per kWh",
    "off_peak_share": "Off-peak Swapping Percentage",
    "full_loaded_kwh_per_km_year1": "BET Full-loaded kWh per km in Year 1",
    "battery_recycle_ratio": "Battery Residual Percentage",
    "bet_subsidy": "BET Purchase Subsidy",
}

def get_pretty_label(var):
    if var in label_map:
        return label_map[var]
    
    label = var.replace("_", " ").title()

    # Fix common units / acronyms
    label = label.replace("Kwh", "kWh")
    label = label.replace("Km", "km")
    label = label.replace("Vkt", "VKT")
    
    return label

########### Input Calculations ############################################################################################    
# Compute discount factors for each year based on discount rate. All the final results are discounted present values.
def discount_factors(rate: float, years: int) -> List[float]:
    return [1 / (1 + rate) ** y for y in range(1, years + 1)]

# Calculate annual driver salary based on working schedule. Data are from UK gov website.
def annual_driver_salary(days_per_year: float, hours_per_week: float, hourly_pay: float) -> float:
    return days_per_year / (hours_per_week / 9) * hours_per_week * hourly_pay

# Adjust insurance cost for BET relative to diesel
def bet_insurance(diesel_insurance: float, markup: float) -> float:
    return diesel_insurance * (1 + markup)

############################# Energy Consumption & Dynamics Calculations ##################################################
# Compute daily full-loaded and unladen driving distances
def diesel_daily_distances(shared: SharedInputs) -> tuple[float, float]:
    full_loaded = shared.full_loaded_km_per_day
    unladen = full_loaded * shared.unladen_ratio_to_full
    return full_loaded, unladen

# Generate yearly diesel fuel consumption trajectory (degradation)
def diesel_yearly_fuel_economies(inp: DieselInputs) -> List[float]:
    vals = [inp.fuel_economy_full_loaded_year1_l_per_km]
    for _ in range(1, 5):
        vals.append(vals[-1] * (1 + inp.fuel_economy_growth_rate))
    return vals

# Generate yearly BET-C energy consumption trajectory
def betc_yearly_full_loaded_economies(inp: BETCInputs) -> List[float]:
    vals = [inp.full_loaded_kwh_per_km_year1]
    for _ in range(1, 5):
        vals.append(vals[-1] * (1 + inp.fuel_economy_growth_rate))
    return vals

# Generate yearly BET-S energy consumption trajectory
def bets_yearly_full_loaded_economies(inp: BETSInputs) -> List[float]:
    vals = [inp.full_loaded_kwh_per_km_year1]
    for _ in range(1, 5):
        vals.append(vals[-1] * (1 + inp.fuel_economy_growth_rate))
    return vals

################### TCO calculations ######################################################################
def compute_diesel(shared: SharedInputs, inp: DieselInputs) -> Dict[str, float]:
    years = shared.years
    df = discount_factors(shared.discount_rate, years)
    full_km, unladen_km = diesel_daily_distances(shared)
    fuel_full = diesel_yearly_fuel_economies(inp)
    fuel_unladen = [x * (1 - inp.unladen_energy_saving) for x in fuel_full]
    daily_use = [full_km * ff + unladen_km * fu for ff, fu in zip(fuel_full, fuel_unladen)]

    annual_km = (full_km + unladen_km) * shared.operational_days_per_year
    annual_salary = annual_driver_salary(
        shared.operational_days_per_year, shared.worked_hours_per_week, shared.driver_hourly_pay
    )
    annual_operating_cost = (
        inp.annual_service_cost
        + annual_salary
        + shared.diesel_insurance
        + shared.operational_days_per_year * inp.lez_operation_percentage * inp.lez_charge
    )

    truck_residual = (
        inp.capex * (1 - inp.fixed_depreciation_rate) ** years
        - inp.variable_depreciation_per_km * annual_km * years
    )

    annual_depot_demand = [
        daily * inp.refuels_per_day * shared.diesel_expected_fleet_size * shared.operational_days_per_year * shared.diesel_depot_share
        for daily in daily_use
    ]
    annual_public_demand = [
        daily * inp.refuels_per_day * shared.diesel_expected_fleet_size * shared.operational_days_per_year * (1 - shared.diesel_depot_share)
        for daily in daily_use
    ]

    depot_infra_undiscounted = (sum(annual_depot_demand) * shared.diesel_bunker_capex_per_l) / shared.diesel_expected_fleet_size
    depot_infra_discounted = sum(d * shared.diesel_bunker_capex_per_l * w for d, w in zip(annual_depot_demand, df)) / shared.diesel_expected_fleet_size

    energy_costs = [
        shared.diesel_depot_price_per_l * dd + shared.diesel_public_price_per_l * pd
        for dd, pd in zip(
            [daily * inp.refuels_per_day * shared.operational_days_per_year * shared.diesel_depot_share for daily in daily_use],
            [daily * inp.refuels_per_day * shared.operational_days_per_year * (1 - shared.diesel_depot_share) for daily in daily_use],
        )
    ]
    total_energy = sum(energy_costs)
    discounted_energy = sum(c * w for c, w in zip(energy_costs, df))

    tco_undiscounted = (inp.capex - truck_residual) + depot_infra_undiscounted + annual_operating_cost * years + total_energy
    tco_discounted = (inp.capex - truck_residual * df[-1]) + annual_operating_cost * sum(df) + depot_infra_discounted + discounted_energy

    total_5y_diesel_litre = sum(daily_use) * shared.operational_days_per_year
    total_5y_energy_kwh = total_5y_diesel_litre * inp.litre_to_kwh

    return {
        "tco_5y_undiscounted": tco_undiscounted,
        "tco_5y_discounted": tco_discounted,
        "tco_per_year_discounted": tco_discounted / years,
        "tco_per_km_discounted": tco_discounted / (annual_km * years),
        "tco_per_kwh_discounted": tco_discounted / total_5y_energy_kwh,
        "annual_km": annual_km,
        "daily_energy_year1_l": daily_use[0],
        "truck_residual": truck_residual,
    }


def compute_bet_c(shared: SharedInputs, inp: BETCInputs, asset_manager_margin: float = 0.10) -> Dict[str, float]:
    years = shared.years
    df = discount_factors(shared.discount_rate, years)
    full_km, unladen_km = diesel_daily_distances(shared)
    econ_full = betc_yearly_full_loaded_economies(inp)
    econ_unladen = [x * (1 - inp.unladen_energy_saving) for x in econ_full]
    daily_kwh = [full_km * ef + unladen_km * eu for ef, eu in zip(econ_full, econ_unladen)]

    annual_km = (full_km + unladen_km) * shared.operational_days_per_year
    annual_salary = annual_driver_salary(
        shared.operational_days_per_year, shared.worked_hours_per_week, shared.driver_hourly_pay
    )
    insurance = bet_insurance(shared.diesel_insurance, shared.bet_insurance_markup)

    truck_capex = inp.glider_capex + inp.battery_capacity_kwh * inp.battery_price_per_kwh
    glider_residual = (
        inp.glider_capex * (1 - inp.glider_fixed_depreciation_rate) ** years
        - inp.glider_variable_depreciation_per_km * annual_km * years
    )
    battery_value0 = inp.battery_capacity_kwh * inp.battery_price_per_kwh
    battery_residual_eol = battery_value0 - battery_value0 * (1 - inp.battery_eol_ratio) * (shared.operational_days_per_year * years / inp.battery_lifetime_cycles)
    battery_residual_recycle = battery_value0 - battery_value0 * (1 - inp.battery_recycle_ratio) * (shared.operational_days_per_year * years / inp.battery_lifetime_cycles)

    annual_energy_total = [daily * inp.recharges_per_day * inp.expected_fleet_size * shared.operational_days_per_year for daily in daily_kwh]
    annual_depot_demand = [shared.bet_depot_share * x for x in annual_energy_total]
    annual_public_demand = [(1 - shared.bet_depot_share) * x for x in annual_energy_total]

    depot_infra_per_year = [
        ((inp.charger_capex_per_kwh + inp.site_capex_per_kwh) * d) / inp.expected_fleet_size
        for d in annual_depot_demand
    ]
    depot_infra_discounted = sum(c * w for c, w in zip(depot_infra_per_year, df))

    annual_operating_cost = inp.annual_service_cost + annual_salary + insurance
    energy_costs = [
        shared.bet_depot_energy_price_per_kwh * dd / inp.expected_fleet_size
        + shared.bet_public_energy_price_per_kwh * pd / inp.expected_fleet_size
        for dd, pd in zip(annual_depot_demand, annual_public_demand)
    ]
    energy_discounted = sum(c * w for c, w in zip(energy_costs, df))

    tco_discounted_eol = (
        (truck_capex - (glider_residual + battery_residual_eol) * df[-1])
        + depot_infra_discounted
        + annual_operating_cost * sum(df)
        + energy_discounted
        - shared.bet_subsidy
    )
    tco_discounted_recycle = (
        (truck_capex - (glider_residual + battery_residual_recycle) * df[-1])
        + depot_infra_discounted
        + annual_operating_cost * sum(df)
        + energy_discounted
        - shared.bet_subsidy
    )



    return {
        "tco_5y_discounted_eol": tco_discounted_eol,
        "tco_5y_discounted_recycle": tco_discounted_recycle,
        "tco_per_year_discounted_eol": tco_discounted_eol / years,
        "tco_per_km_discounted_eol": tco_discounted_eol / (annual_km * years),
        "tco_per_kwh_discounted_eol": tco_discounted_eol / (shared.operational_days_per_year * sum(daily_kwh)),
        "tco_per_year_discounted_recycle": tco_discounted_recycle / years,
        "tco_per_km_discounted_recycle": tco_discounted_recycle / (annual_km * years),
        "tco_per_kwh_discounted_recycle": tco_discounted_recycle / (shared.operational_days_per_year * sum(daily_kwh)),
        "annual_km": annual_km,
        "daily_energy_year1_kwh": daily_kwh[0],
        "annual_driver_cost": annual_salary,
        "daily_kwh_by_year": daily_kwh,       
    }


def compute_bet_s(shared: SharedInputs, inp: BETSInputs, asset_manager_margin: float = 0.10) -> Dict[str, float]:
    years = shared.years
    df = discount_factors(shared.discount_rate, years)
    full_km, unladen_km = diesel_daily_distances(shared)
    econ_full = bets_yearly_full_loaded_economies(inp)
    econ_unladen = [x * (1 - inp.unladen_energy_saving) for x in econ_full]
    daily_kwh = [full_km * ef + unladen_km * eu for ef, eu in zip(econ_full, econ_unladen)]

    annual_km = (full_km + unladen_km) * shared.operational_days_per_year
    annual_salary = annual_driver_salary(
        shared.operational_days_per_year, shared.worked_hours_per_week, shared.driver_hourly_pay
    )
    insurance = bet_insurance(shared.diesel_insurance, shared.bet_insurance_markup)
    annual_operating_cost = inp.annual_battery_service_cost + annual_salary + insurance

    glider_residual = (
        inp.glider_capex * (1 - inp.glider_fixed_depreciation_rate) ** years
        - inp.glider_variable_depreciation_per_km * annual_km * years
    )

    expected_station_service_demand = inp.max_station_service_capacity_trucks_per_day * inp.expected_station_utilisation
    station_depr_per_truck_5y = ((inp.station_capex + inp.site_capex) / inp.station_lifetime_years) / (expected_station_service_demand * shared.operational_days_per_year) * shared.operational_days_per_year * years

    battery_value_station = inp.station_battery_bays * inp.battery_pack_capacity_kwh * inp.battery_price_per_kwh
    battery_value_truck = expected_station_service_demand * inp.battery_packs_per_truck * inp.battery_pack_capacity_kwh * inp.battery_price_per_kwh
    battery_system_value = battery_value_station + battery_value_truck

    battery_depr_eol_5y = (battery_system_value * (1 - inp.battery_eol_ratio) * (shared.operational_days_per_year * years / inp.battery_lifetime_cycles)) / expected_station_service_demand
    battery_depr_recycle_5y = (battery_system_value * (1 - inp.battery_recycle_ratio) * (shared.operational_days_per_year * years / inp.battery_lifetime_cycles)) / expected_station_service_demand

    battery_service_5y = years * inp.annual_battery_service_cost
    battery_rent_5y = inp.battery_rent_per_month_ex_depreciation * 12 * years
    fixed_swapping_5y = inp.swapping_fee_flat * inp.swaps_per_day * shared.operational_days_per_year * years

    base_energy_price = shared.peak_price_per_kwh * (1 - shared.off_peak_share) + shared.off_peak_price_per_kwh * shared.off_peak_share
    energy_service_costs = [daily * inp.swaps_per_day * shared.operational_days_per_year * base_energy_price for daily in daily_kwh]
    energy_margin_addition = sum(energy_service_costs) * shared.electricity_margin

    discounted_capex = inp.glider_capex - glider_residual * df[-1]
    discounted_operating = annual_operating_cost * sum(df)
    discounted_baas_common = ((station_depr_per_truck_5y + battery_service_5y + battery_rent_5y + fixed_swapping_5y) / years) * sum(df)
    discounted_energy = sum(cost * (1 + shared.electricity_margin) * w for cost, w in zip(energy_service_costs, df))

    tco_discounted_eol = (
        discounted_capex
        + discounted_operating
        + discounted_baas_common
        + (battery_depr_eol_5y / years) * sum(df)
        + discounted_energy
        - shared.bet_subsidy
    )
    tco_discounted_recycle = (
        discounted_capex
        + discounted_operating
        + discounted_baas_common
        + (battery_depr_recycle_5y / years) * sum(df)
        + discounted_energy
        - shared.bet_subsidy
    )

    # ===== AEaaS supplier discounted cost base (BET-S only) =====
    discounted_driver_cost_5y = annual_salary * sum(df)
    discounted_insurance_5y = insurance * sum(df)

    discounted_glider_cost_for_aeaas = (
        inp.glider_capex * shared.aeaas_glider_cost_factor
        - glider_residual * df[-1] * shared.aeaas_glider_cost_factor
    )

    discounted_baas_cost_for_aeaas = (
        (
            (station_depr_per_truck_5y + battery_service_5y + battery_rent_5y + fixed_swapping_5y) / years
        ) * sum(df)
        + (battery_depr_recycle_5y / years) * sum(df)
        + discounted_energy
    ) * shared.aeaas_baas_cost_factor

    # supplier bears everything except driver cost
    aeaas_asset_service_cost_5y = (
        discounted_glider_cost_for_aeaas
        + discounted_insurance_5y
        + discounted_baas_cost_for_aeaas
    )
    asset_service = compute_asset_service_unit_prices(
        asset_service_cost_5y=aeaas_asset_service_cost_5y,
        annual_driver_cost=annual_salary,
        annual_km=annual_km,
        daily_energy_list=daily_kwh,
        shared=shared,
        margin=asset_manager_margin,
    )

    aas_gap_vs_own_tco = asset_service["freight_total_cost_5y"] - tco_discounted_recycle
    
    return {
        "tco_5y_discounted_eol": tco_discounted_eol,
        "tco_5y_discounted_recycle": tco_discounted_recycle,
        "tco_per_year_discounted_eol": tco_discounted_eol / years,
        "tco_per_km_discounted_eol": tco_discounted_eol / (annual_km * years),
        "tco_per_kwh_discounted_eol": tco_discounted_eol / (shared.operational_days_per_year * sum(daily_kwh)),
        "tco_per_year_discounted_recycle": tco_discounted_recycle / years,
        "tco_per_km_discounted_recycle": tco_discounted_recycle / (annual_km * years),
        "tco_per_kwh_discounted_recycle": tco_discounted_recycle / (shared.operational_days_per_year * sum(daily_kwh)),
        "annual_km": annual_km,
        "daily_energy_year1_kwh": daily_kwh[0],
        "energy_margin_addition_5y": energy_margin_addition,
        "annual_driver_cost": annual_salary,
        "daily_kwh_by_year": daily_kwh,
        **asset_service,
        "discounted_glider_cost_for_aeaas": discounted_glider_cost_for_aeaas,
        "discounted_baas_cost_for_aeaas": discounted_baas_cost_for_aeaas,
        "aeaas_asset_service_cost_5y": aeaas_asset_service_cost_5y,
        "aas_gap_vs_own_tco": aas_gap_vs_own_tco,
    }


#  Calculate TCO of diesel, BET-C, and BET-S under given inputs
def run_model(shared=None, diesel_inp=None, betc_inp=None, bets_inp=None, asset_manager_margin: float = 0.10):
    if shared is None:
        shared = SharedInputs()
    if diesel_inp is None:
        diesel_inp = DieselInputs()
    if betc_inp is None:
        betc_inp = BETCInputs(
            battery_recycle_ratio=shared.battery_recycle_value_ratio
        )
    if bets_inp is None:
        bets_inp = BETSInputs(
            battery_recycle_ratio=shared.battery_recycle_value_ratio
        )

    diesel = compute_diesel(shared, diesel_inp)
    bet_c = compute_bet_c(shared, betc_inp, asset_manager_margin=asset_manager_margin)
    bet_s = compute_bet_s(shared, bets_inp, asset_manager_margin=asset_manager_margin)

    return {
        "diesel": diesel,
        "bet_c": bet_c,
        "bet_s": bet_s,
    }


# Extract pairwise TCO gaps between vehicle types
def extract_tco_gaps(results):
    bet_c_vs_diesel = (
        results["bet_c"]["tco_5y_discounted_recycle"]
        - results["diesel"]["tco_5y_discounted"]
    )
    bet_s_vs_diesel = (
        results["bet_s"]["tco_5y_discounted_recycle"]
        - results["diesel"]["tco_5y_discounted"]
    )
    bet_s_vs_bet_c = (
        results["bet_s"]["tco_5y_discounted_recycle"]
        - results["bet_c"]["tco_5y_discounted_recycle"]
    )

    return {
        "bet_c_vs_diesel": bet_c_vs_diesel,
        "bet_s_vs_diesel": bet_s_vs_diesel,
        "bet_s_vs_bet_c": bet_s_vs_bet_c,
    }


############ AEaaS Pricing Model  ################################################################
# Convert asset-service cost into unit prices (per km / per kWh) with margin
def compute_asset_service_unit_prices(
    asset_service_cost_5y: float,
    annual_driver_cost: float,
    annual_km: float,
    daily_energy_list: list[float],
    shared: SharedInputs,
    margin: float = 0.10,
) -> Dict[str, float]:
    years = shared.years
    df = discount_factors(shared.discount_rate, years)

    discounted_driver_cost_5y = annual_driver_cost * sum(df)

    total_5y_km = annual_km * years
    total_5y_kwh = sum(daily_energy_list) * shared.operational_days_per_year

    # asset manager cost base
    unit_cost_per_km = asset_service_cost_5y / total_5y_km
    unit_cost_per_kwh = asset_service_cost_5y / total_5y_kwh

    # asset manager selling price
    price_per_km_with_margin = unit_cost_per_km * (1 + margin)
    price_per_kwh_with_margin = unit_cost_per_kwh * (1 + margin)

    asset_price_5y = asset_service_cost_5y * (1 + margin)

    # driver cost borne by freight company
    driver_cost_per_km = discounted_driver_cost_5y / total_5y_km
    driver_cost_per_kwh = discounted_driver_cost_5y / total_5y_kwh

    # freight company's all-in effective unit cost
    freight_total_cost_per_km = price_per_km_with_margin + driver_cost_per_km
    freight_total_cost_per_kwh = price_per_kwh_with_margin + driver_cost_per_kwh

    # freight company's all-in 5y total cost
    freight_total_cost_5y = asset_price_5y + discounted_driver_cost_5y

    freight_total_cost_5y_from_km = freight_total_cost_per_km * total_5y_km
    freight_total_cost_5y_from_kwh = freight_total_cost_per_kwh * total_5y_kwh

    return {
        "discounted_driver_cost_5y": discounted_driver_cost_5y,
        "asset_service_cost_5y": asset_service_cost_5y,
        "asset_price_5y": asset_price_5y,

        "total_5y_km": total_5y_km,
        "total_5y_kwh": total_5y_kwh,

        "unit_cost_per_km": unit_cost_per_km,
        "unit_cost_per_kwh": unit_cost_per_kwh,

        "price_per_km_with_margin": price_per_km_with_margin,
        "price_per_kwh_with_margin": price_per_kwh_with_margin,

        "driver_cost_per_km": driver_cost_per_km,
        "driver_cost_per_kwh": driver_cost_per_kwh,

        "freight_total_cost_per_km": freight_total_cost_per_km,
        "freight_total_cost_per_kwh": freight_total_cost_per_kwh,

        "freight_total_cost_5y": freight_total_cost_5y,
        "freight_total_cost_5y_from_km": freight_total_cost_5y_from_km,
        "freight_total_cost_5y_from_kwh": freight_total_cost_5y_from_kwh,
    }

# Evaluate effect of asset-manager margin on freight cost (deterministic)
def run_margin_sweep_for_freight_all_in_per_km(
    margins,
    shared=None,
    diesel_inp=None,
    betc_inp=None,
    bets_inp=None,
):
    if shared is None:
        shared = SharedInputs()
    if diesel_inp is None:
        diesel_inp = DieselInputs()
    if betc_inp is None:
        betc_inp = BETCInputs(
            battery_recycle_ratio=shared.battery_recycle_value_ratio
        )
    if bets_inp is None:
        bets_inp = BETSInputs(
            battery_recycle_ratio=shared.battery_recycle_value_ratio
        )

    rows = []

    for margin in margins:
        results = run_model(
            shared=shared,
            diesel_inp=diesel_inp,
            betc_inp=betc_inp,
            bets_inp=bets_inp,
            asset_manager_margin=margin,
        )

        rows.append({
            "asset_manager_margin": margin,
            "diesel_tco_per_km": results["diesel"]["tco_per_km_discounted"],
            "bets_freight_all_in_per_km": results["bet_s"]["freight_total_cost_per_km"],
            "bets_minus_diesel_per_km": results["bet_s"]["freight_total_cost_per_km"] - results["diesel"]["tco_per_km_discounted"],
        })

    return pd.DataFrame(rows)

# margin analysis but with Monte Carlo uncertainty
def run_margin_sweep_with_uncertainty(
    margins,
    n_runs=500,
    random_seed=42,
):
    rng = np.random.default_rng(random_seed)
    rows = []

    for margin in margins:
        for i in range(n_runs):
            # ===== sample uncertain inputs (same logic as baseline Monte Carlo) =====
            sampled_discount_rate = sample_triangular(0.08, 0.10, 0.12, rng)
            sampled_full_loaded_km_per_day = sample_triangular(192.0, 240.0, 288.0, rng)

            sampled_peak_price_per_kwh = sample_triangular(0.16, 0.20, 0.24, rng)
            sampled_off_peak_share = sample_triangular(0.30, 0.50, 0.70, rng)

            sampled_bet_depot_energy_price_per_kwh = sample_triangular(0.18, 0.22, 0.28, rng)
            sampled_bet_public_energy_price_per_kwh = sample_triangular(0.30, 0.39, 0.50, rng)

            sampled_full_loaded_kwh_per_km_year1 = sample_triangular(1.20, 1.37, 1.55, rng)
            sampled_battery_recycle_ratio = sample_triangular(0.05, 0.10, 0.20, rng)
            sampled_battery_lifetime_cycles = sample_triangular(1600.0, 2000.0, 3000.0, rng)
            sampled_unladen_energy_saving = sample_triangular(0.20, 0.25, 0.30, rng)

            sampled_battery_capacity_kwh = sample_triangular(400.0, 621.0, 800.0, rng)

            sampled_bet_subsidy = sample_triangular(0.0, 0.0, 120000.0, rng)

            # ===== build sampled inputs =====
            shared_i = SharedInputs(
                discount_rate=sampled_discount_rate,
                full_loaded_km_per_day=sampled_full_loaded_km_per_day,
                peak_price_per_kwh=sampled_peak_price_per_kwh,
                off_peak_share=sampled_off_peak_share,
                bet_depot_energy_price_per_kwh=sampled_bet_depot_energy_price_per_kwh,
                bet_public_energy_price_per_kwh=sampled_bet_public_energy_price_per_kwh,
                bet_subsidy=sampled_bet_subsidy,
            )

            diesel_i = DieselInputs()

            betc_i = BETCInputs(
                battery_recycle_ratio=sampled_battery_recycle_ratio,
                battery_lifetime_cycles=sampled_battery_lifetime_cycles,
                unladen_energy_saving=sampled_unladen_energy_saving,
                full_loaded_kwh_per_km_year1=sampled_full_loaded_kwh_per_km_year1,
                battery_capacity_kwh=sampled_battery_capacity_kwh,
            )

            bets_i = BETSInputs(
                battery_recycle_ratio=sampled_battery_recycle_ratio,
                battery_lifetime_cycles=sampled_battery_lifetime_cycles,
                unladen_energy_saving=sampled_unladen_energy_saving,
                full_loaded_kwh_per_km_year1=sampled_full_loaded_kwh_per_km_year1,
            )

            results = run_model(
                shared=shared_i,
                diesel_inp=diesel_i,
                betc_inp=betc_i,
                bets_inp=bets_i,
                asset_manager_margin=margin,
            )

            rows.append({
                "asset_manager_margin": margin,
                "iteration": i + 1,
                "diesel_tco_per_km": results["diesel"]["tco_per_km_discounted"],
                "bets_freight_all_in_per_km": results["bet_s"]["freight_total_cost_per_km"],
                "bets_minus_diesel_per_km": results["bet_s"]["freight_total_cost_per_km"] - results["diesel"]["tco_per_km_discounted"],
            })

    return pd.DataFrame(rows)

# Monte Carlo summary showed in the panel (Optional)
def pretty_monte_carlo_summary(summary_df, probability_df) -> str:
    lines = []
    lines.append("Monte Carlo summary")
    lines.append("-" * 80)

    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['metric']}: "
            f"mean = £{row['mean']:,.2f}, "
            f"median = £{row['median']:,.2f}, "
            f"p5 = £{row['p5']:,.2f}, "
            f"p95 = £{row['p95']:,.2f}"
        )

    lines.append("")
    lines.append("Probabilities")
    lines.append("-" * 80)

    for _, row in probability_df.iterrows():
        lines.append(
            f"{row['metric']}: {row['probability']:.2%}"
        )

    return "\n".join(lines)

# Identify what can affect the TCO gap
def get_drivers_of_gap(
    df,
    gap_column="gap_bet_s_diesel",
    input_columns=None,
):
    if input_columns is None:
        input_columns = [
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
        ]

    rows = []
    for col in input_columns:
        corr = df[col].corr(df[gap_column])
        rows.append({
            "variable": col,
            "correlation_with_gap": corr,
            "abs_correlation": abs(corr),
        })

    driver_df = pd.DataFrame(rows)
    driver_df = driver_df.sort_values(
        by="abs_correlation",
        ascending=False
    ).reset_index(drop=True)

    return driver_df

# Impactful variables ranking
def pretty_drivers(driver_df, gap_name="BET-S - Diesel") -> str:
    lines = []
    lines.append(f"Drivers of {gap_name}")
    lines.append("-" * 60)

    for i, row in driver_df.iterrows():
        direction = "positive" if row["correlation_with_gap"] > 0 else "negative"
        lines.append(
            f"{i+1}. {row['variable']}: "
            f"corr = {row['correlation_with_gap']:.3f} "
            f"({direction})"
        )

    return "\n".join(lines)

############################## Deterministic Sensitivity Analysis ##################################################   
# Run one-variable-at-a-time
def run_sensitivity_analysis(
    target_class,
    variable_name,
    base_value,
    changes,
    shared=None,
    diesel_inp=None,
    betc_inp=None,
    bets_inp=None,
):
    if shared is None:
        shared = SharedInputs()
    if diesel_inp is None:
        diesel_inp = DieselInputs()
    if betc_inp is None:
        betc_inp = BETCInputs(
            battery_recycle_ratio=shared.battery_recycle_value_ratio
        )
    if bets_inp is None:
        bets_inp = BETSInputs(
            battery_recycle_ratio=shared.battery_recycle_value_ratio
        )

    # 允许 target_class 既可以是字符串，也可以是列表
    if isinstance(target_class, str):
        target_classes = [target_class]
    else:
        target_classes = target_class

    valid_classes = {"shared", "diesel", "betc", "bets"}
    if not set(target_classes).issubset(valid_classes):
        raise ValueError("target_class must be one of 'shared', 'diesel', 'betc', 'bets', or a list of them")

    labels = []
    bet_c_vs_diesel = []
    bet_s_vs_diesel = []
    bet_s_vs_bet_c = []

    for ch in changes:
        new_value = base_value * (1 + ch)

        shared_i = shared
        diesel_i = diesel_inp
        betc_i = betc_inp
        bets_i = bets_inp

        # 可以同时修改多个 class
        if "shared" in target_classes:
            shared_i = update_input(shared_i, variable_name, new_value)

        if "diesel" in target_classes:
            diesel_i = update_input(diesel_i, variable_name, new_value)

        if "betc" in target_classes:
            betc_i = update_input(betc_i, variable_name, new_value)

        if "bets" in target_classes:
            bets_i = update_input(bets_i, variable_name, new_value)

        # shared 改完后，继续同步 recycle ratio 到 BETC/BETS 默认输入
        if "shared" in target_classes:
            betc_i = update_input(
                betc_i,
                "battery_recycle_ratio",
                shared_i.battery_recycle_value_ratio
            )
            bets_i = update_input(
                bets_i,
                "battery_recycle_ratio",
                shared_i.battery_recycle_value_ratio
            )

        results = run_model(
            shared=shared_i,
            diesel_inp=diesel_i,
            betc_inp=betc_i,
            bets_inp=bets_i,
        )
        gaps = extract_tco_gaps(results)

        if ch > 0:
            label = f"+{int(ch * 100)}%"
        elif ch < 0:
            label = f"{int(ch * 100)}%"
        else:
            label = "0%"

        labels.append(label)
        bet_c_vs_diesel.append(gaps["bet_c_vs_diesel"])
        bet_s_vs_diesel.append(gaps["bet_s_vs_diesel"])
        bet_s_vs_bet_c.append(gaps["bet_s_vs_bet_c"])

    return {
        "target_class": target_classes,
        "variable_name": variable_name,
        "base_value": base_value,
        "labels": labels,
        "bet_c_vs_diesel": bet_c_vs_diesel,
        "bet_s_vs_diesel": bet_s_vs_diesel,
        "bet_s_vs_bet_c": bet_s_vs_bet_c,
    }

# Run multiple sensitivity analyses for different variables
def run_multiple_sensitivity_analyses(specs, changes):
    all_results = []

    for spec in specs:
        result = run_sensitivity_analysis(
            target_class=spec["target_class"],
            variable_name=spec["variable_name"],
            base_value=spec["base_value"],
            changes=changes,
        )
        all_results.append(result)

    return all_results


################ Monte Carlo Sampling ################################
UNCERTAINTY_NOTE = "Shaded area: 5th-95th percentile range across Monte Carlo simulations"

def sample_triangular(left, mode, right, rng):
    return rng.triangular(left, mode, right)

# =========================================================
# Independent-effect Monte Carlo for one-at-a-time boxplots
# =========================================================

# Define uncertain variables and their distributions
def get_uncertainty_specs():
    """
    target_class:
        - "shared"
        - "diesel"
        - "betc"
        - "bets"
        - ["betc", "bets"]  # jointly changed in both BET-C and BET-S
    """
    return [
        {
            "variable": "discount_rate",
            "target_class": "shared",
            "left": 0.08,
            "mode": 0.10,
            "right": 0.12,
        },
        {
            "variable": "full_loaded_km_per_day",
            "target_class": "shared",
            "left": 192.0,
            "mode": 240.0,
            "right": 288.0,
        },
        {
            "variable": "peak_price_per_kwh",
            "target_class": "shared",
            "left": 0.16,
            "mode": 0.20,
            "right": 0.24,
        },
        {
            "variable": "off_peak_share",
            "target_class": "shared",
            "left": 0.30,
            "mode": 0.50,
            "right": 0.70,
        },
        {
            "variable": "bet_depot_energy_price_per_kwh",
            "target_class": "shared",
            "left": 0.18,
            "mode": 0.22,
            "right": 0.28,
        },
        {
            "variable": "bet_public_energy_price_per_kwh",
            "target_class": "shared",
            "left": 0.30,
            "mode": 0.39,
            "right": 0.50,
        },
        {
            "variable": "full_loaded_kwh_per_km_year1",
            "target_class": ["betc", "bets"],
            "left": 1.20,
            "mode": 1.37,
            "right": 1.55,
        },
        {
            "variable": "battery_recycle_ratio",
            "target_class": ["betc", "bets"],
            "left": 0.05,
            "mode": 0.10,
            "right": 0.20,
        },
        {
            "variable": "battery_lifetime_cycles",
            "target_class": ["betc", "bets"],
            "left": 1600.0,
            "mode": 2000.0,
            "right": 3000.0,
        },
        {
            "variable": "unladen_energy_saving",
            "target_class": ["betc", "bets"],
            "left": 0.20,
            "mode": 0.25,
            "right": 0.30,
        },
        {
            "variable": "battery_capacity_kwh",
            "target_class": "betc",
            "left": 400.0,
            "mode": 621.0,
            "right": 800.0,
        },
        {
            "variable": "bet_subsidy",
            "target_class": "shared",
            "left": 0,
            "mode": 0,
            "right": 120000.0,
        },
    ]

# Apply one uncertain variable change for calculations
def apply_single_variable_change(shared, diesel_inp, betc_inp, bets_inp, spec, sampled_value):
    """
    Apply one sampled uncertain variable to the correct input object(s),
    keeping all other inputs at baseline.
    """
    target_class = spec["target_class"]
    variable_name = spec["variable"]

    shared_i = shared
    diesel_i = diesel_inp
    betc_i = betc_inp
    bets_i = bets_inp

    if isinstance(target_class, str):
        target_class = [target_class]

    if "shared" in target_class:
        shared_i = update_input(shared_i, variable_name, sampled_value)

    if "diesel" in target_class:
        diesel_i = update_input(diesel_i, variable_name, sampled_value)

    if "betc" in target_class:
        betc_i = update_input(betc_i, variable_name, sampled_value)

    if "bets" in target_class:
        bets_i = update_input(bets_i, variable_name, sampled_value)

    return shared_i, diesel_i, betc_i, bets_i

# Run one-at-a-time Monte Carlo simulation for each variable
def run_independent_variable_monte_carlo(n_runs=500, random_seed=42):
    """
    For each uncertain variable:
    - vary ONLY that variable according to its triangular distribution
    - keep all other variables at baseline
    - run model n_runs times
    Returns a long dataframe for boxplotting.
    """
    rng = np.random.default_rng(random_seed)
    specs = get_uncertainty_specs()

    base_shared = SharedInputs()
    base_diesel = DieselInputs()
    base_betc = BETCInputs(
        battery_recycle_ratio=base_shared.battery_recycle_value_ratio
    )
    base_bets = BETSInputs(
        battery_recycle_ratio=base_shared.battery_recycle_value_ratio
    )

    rows = []

    for spec in specs:
        var_name = spec["variable"]

        for i in range(n_runs):
            sampled_value = sample_triangular(
                spec["left"], spec["mode"], spec["right"], rng
            )

            # reset to baseline every run
            shared_i = base_shared
            diesel_i = base_diesel
            betc_i = base_betc
            bets_i = base_bets

            shared_i, diesel_i, betc_i, bets_i = apply_single_variable_change(
                shared_i, diesel_i, betc_i, bets_i, spec, sampled_value
            )

            results = run_model(
                shared=shared_i,
                diesel_inp=diesel_i,
                betc_inp=betc_i,
                bets_inp=bets_i,
            )

            diesel_tco = results["diesel"]["tco_5y_discounted"]
            betc_tco = results["bet_c"]["tco_5y_discounted_recycle"]
            bets_tco = results["bet_s"]["tco_5y_discounted_recycle"]

            rows.append({
                "variable": var_name,
                "iteration": i + 1,
                "sampled_value": sampled_value,

                "diesel_tco": diesel_tco,
                "bet_c_tco": betc_tco,
                "bet_s_tco": bets_tco,

                "gap_bet_c_diesel": betc_tco - diesel_tco,
                "gap_bet_s_diesel": bets_tco - diesel_tco,
                "gap_bet_s_bet_c": bets_tco - betc_tco,
            })

    return pd.DataFrame(rows)

def run_monte_carlo_simulation(n_runs=500, random_seed=42):
    rng = np.random.default_rng(random_seed)
    rows = []

    for i in range(n_runs):
        # ===== 1) sample uncertain inputs (triangular distributions) =====
        sampled_discount_rate = sample_triangular(0.08, 0.10, 0.12, rng)

        sampled_full_loaded_km_per_day = sample_triangular(192.0, 240.0, 288.0, rng)

        sampled_peak_price_per_kwh = sample_triangular(0.16, 0.20, 0.24, rng)
        sampled_off_peak_share = sample_triangular(0.30, 0.50, 0.70, rng)

        sampled_bet_depot_energy_price_per_kwh = sample_triangular(0.18, 0.22, 0.28, rng)
        sampled_bet_public_energy_price_per_kwh = sample_triangular(0.30, 0.39, 0.50, rng)

        # BET-C and BET-S jointly changing variables
        sampled_full_loaded_kwh_per_km_year1 = sample_triangular(1.20, 1.37, 1.55, rng)
        sampled_battery_recycle_ratio = sample_triangular(0.05, 0.10, 0.20, rng)
        sampled_battery_lifetime_cycles = sample_triangular(1600.0, 2000.0, 3000.0, rng)
        sampled_unladen_energy_saving = sample_triangular(0.2, 0.25, 0.3, rng)
        sampled_bet_subsidy = sample_triangular(0.0, 0.0, 120000.0, rng)

        # BET-C only
        sampled_battery_capacity_kwh = sample_triangular(400.0, 621.0, 800.0, rng)

        # ===== 2) build sampled inputs =====
        shared_i = SharedInputs(
            discount_rate=sampled_discount_rate,
            full_loaded_km_per_day=sampled_full_loaded_km_per_day,
            peak_price_per_kwh=sampled_peak_price_per_kwh,
            off_peak_share=sampled_off_peak_share,
            bet_depot_energy_price_per_kwh=sampled_bet_depot_energy_price_per_kwh,
            bet_public_energy_price_per_kwh=sampled_bet_public_energy_price_per_kwh,
            bet_subsidy=sampled_bet_subsidy,
        )

        diesel_i = DieselInputs()

        betc_i = BETCInputs(
            battery_recycle_ratio=sampled_battery_recycle_ratio,
            battery_lifetime_cycles=sampled_battery_lifetime_cycles,
            unladen_energy_saving=sampled_unladen_energy_saving,
            full_loaded_kwh_per_km_year1=sampled_full_loaded_kwh_per_km_year1,
            battery_capacity_kwh=sampled_battery_capacity_kwh,
        )

        bets_i = BETSInputs(
            battery_recycle_ratio=sampled_battery_recycle_ratio,
            battery_lifetime_cycles=sampled_battery_lifetime_cycles,
            unladen_energy_saving=sampled_unladen_energy_saving,
            full_loaded_kwh_per_km_year1=sampled_full_loaded_kwh_per_km_year1,
        )

        # ===== 3) run model =====
        diesel = compute_diesel(shared_i, diesel_i)
        bet_c = compute_bet_c(shared_i, betc_i)
        bet_s = compute_bet_s(shared_i, bets_i)

        diesel_tco = diesel["tco_5y_discounted"]
        bet_c_tco = bet_c["tco_5y_discounted_recycle"]
        bet_s_tco = bet_s["tco_5y_discounted_recycle"]

        gap_bet_c_diesel = bet_c_tco - diesel_tco
        gap_bet_s_diesel = bet_s_tco - diesel_tco
        gap_bet_s_bet_c = bet_s_tco - bet_c_tco

        rows.append({
            "iteration": i + 1,

            "discount_rate": sampled_discount_rate,
            "full_loaded_km_per_day": sampled_full_loaded_km_per_day,
            "peak_price_per_kwh": sampled_peak_price_per_kwh,
            "off_peak_share": sampled_off_peak_share,
            "bet_depot_energy_price_per_kwh": sampled_bet_depot_energy_price_per_kwh,
            "bet_public_energy_price_per_kwh": sampled_bet_public_energy_price_per_kwh,

            "full_loaded_kwh_per_km_year1": sampled_full_loaded_kwh_per_km_year1,
            "battery_recycle_ratio": sampled_battery_recycle_ratio,
            "battery_lifetime_cycles": sampled_battery_lifetime_cycles,
            "unladen_energy_saving": sampled_unladen_energy_saving,

            "battery_capacity_kwh": sampled_battery_capacity_kwh,

            "bet_subsidy": sampled_bet_subsidy,

            "diesel_tco": diesel_tco,
            "bet_c_tco": bet_c_tco,
            "bet_s_tco": bet_s_tco,
            "gap_bet_c_diesel": gap_bet_c_diesel,
            "gap_bet_s_diesel": gap_bet_s_diesel,
            "gap_bet_s_bet_c": gap_bet_s_bet_c,
        })

    return pd.DataFrame(rows)

def run_projection_monte_carlo(
    start_year=2026,
    end_year=2040,
    n_runs=500,
    random_seed=42,
):
    """
    For each purchase year:
    1. build projected baseline inputs for that year
    2. run Monte Carlo around that year's projected values
    3. collect TCO distributions
    """
    rng = np.random.default_rng(random_seed)
    rows = []

    for year in range(start_year, end_year + 1):
        shared_base, diesel_base, betc_base, bets_base = build_projected_inputs_for_year(
            target_year=year,
            base_year=start_year,
            shared=SharedInputs(),
            diesel_inp=DieselInputs(),
            betc_inp=BETCInputs(
                battery_recycle_ratio=SharedInputs().battery_recycle_value_ratio
            ),
            bets_inp=BETSInputs(
                battery_recycle_ratio=SharedInputs().battery_recycle_value_ratio
            ),
        )

        for i in range(n_runs):
            # ===== sample around projected-year baseline =====
            sampled_discount_rate = sample_triangular(
                shared_base.discount_rate * 0.8,
                shared_base.discount_rate,
                shared_base.discount_rate * 1.2,
                rng
            )

            sampled_full_loaded_km_per_day = sample_triangular(
                shared_base.full_loaded_km_per_day * 0.8,
                shared_base.full_loaded_km_per_day,
                shared_base.full_loaded_km_per_day * 1.2,
                rng
            )

            sampled_peak_price_per_kwh = sample_triangular(
                shared_base.peak_price_per_kwh * 0.8,
                shared_base.peak_price_per_kwh,
                shared_base.peak_price_per_kwh * 1.2,
                rng
            )

            sampled_off_peak_share = sample_triangular(
                max(0.0, shared_base.off_peak_share * 0.6),
                shared_base.off_peak_share,
                min(1.0, shared_base.off_peak_share * 1.4),
                rng
            )

            sampled_bet_depot_energy_price_per_kwh = sample_triangular(
                shared_base.bet_depot_energy_price_per_kwh * 0.8,
                shared_base.bet_depot_energy_price_per_kwh,
                shared_base.bet_depot_energy_price_per_kwh * 1.25,
                rng
            )

            sampled_bet_public_energy_price_per_kwh = sample_triangular(
                shared_base.bet_public_energy_price_per_kwh * 0.8,
                shared_base.bet_public_energy_price_per_kwh,
                shared_base.bet_public_energy_price_per_kwh * 1.25,
                rng
            )

            sampled_full_loaded_kwh_per_km_year1 = sample_triangular(
                betc_base.full_loaded_kwh_per_km_year1 * 0.88,
                betc_base.full_loaded_kwh_per_km_year1,
                betc_base.full_loaded_kwh_per_km_year1 * 1.13,
                rng
            )

            sampled_battery_recycle_ratio = sample_triangular(
                max(0.0, SharedInputs().battery_recycle_value_ratio * 0.5),
                SharedInputs().battery_recycle_value_ratio,
                min(1.0, SharedInputs().battery_recycle_value_ratio * 2.0),
                rng
            )

            sampled_battery_lifetime_cycles = sample_triangular(
                betc_base.battery_lifetime_cycles * 0.8,
                betc_base.battery_lifetime_cycles,
                betc_base.battery_lifetime_cycles * 1.3,
                rng
            )

            sampled_unladen_energy_saving = sample_triangular(
                0.20,
                0.25,
                0.30,
                rng
            )

            sampled_battery_capacity_kwh = sample_triangular(
                betc_base.battery_capacity_kwh * 0.65,
                betc_base.battery_capacity_kwh,
                betc_base.battery_capacity_kwh * 1.29,
                rng
            )
            
            sampled_bet_subsidy = sample_triangular(0.0, 0.0, 120000.0, rng)
            
            # ===== build sampled inputs =====
            shared_i = replace(
                shared_base,
                discount_rate=sampled_discount_rate,
                full_loaded_km_per_day=sampled_full_loaded_km_per_day,
                peak_price_per_kwh=sampled_peak_price_per_kwh,
                off_peak_share=sampled_off_peak_share,
                bet_depot_energy_price_per_kwh=sampled_bet_depot_energy_price_per_kwh,
                bet_public_energy_price_per_kwh=sampled_bet_public_energy_price_per_kwh,
                bet_subsidy=sampled_bet_subsidy,
            )

            diesel_i = diesel_base

            betc_i = replace(
                betc_base,
                battery_recycle_ratio=sampled_battery_recycle_ratio,
                battery_lifetime_cycles=sampled_battery_lifetime_cycles,
                unladen_energy_saving=sampled_unladen_energy_saving,
                full_loaded_kwh_per_km_year1=sampled_full_loaded_kwh_per_km_year1,
                battery_capacity_kwh=sampled_battery_capacity_kwh,
            )

            bets_i = replace(
                bets_base,
                battery_recycle_ratio=sampled_battery_recycle_ratio,
                battery_lifetime_cycles=sampled_battery_lifetime_cycles,
                unladen_energy_saving=sampled_unladen_energy_saving,
                full_loaded_kwh_per_km_year1=sampled_full_loaded_kwh_per_km_year1,
            )

            results = run_model(
                shared=shared_i,
                diesel_inp=diesel_i,
                betc_inp=betc_i,
                bets_inp=bets_i,
            )

            rows.append({
                "year": year,
                "iteration": i + 1,

                "diesel_tco_5y_discounted": results["diesel"]["tco_5y_discounted"],
                "betc_tco_5y_discounted": results["bet_c"]["tco_5y_discounted_recycle"],
                "bets_tco_5y_discounted": results["bet_s"]["tco_5y_discounted_recycle"],

                "diesel_tco_per_km": results["diesel"]["tco_per_km_discounted"],
                "betc_tco_per_km": results["bet_c"]["tco_per_km_discounted_recycle"],
                "bets_tco_per_km": results["bet_s"]["tco_per_km_discounted_recycle"],

                "diesel_tco_per_kwh": results["diesel"]["tco_per_kwh_discounted"],
                "betc_tco_per_kwh": results["bet_c"]["tco_per_kwh_discounted_recycle"],
                "bets_tco_per_kwh": results["bet_s"]["tco_per_kwh_discounted_recycle"],
            })

    return pd.DataFrame(rows)

# The results of the simulation. Could be printed out
def summarize_monte_carlo_results(df):  
    metrics = [
        "diesel_tco",
        "bet_c_tco",
        "bet_s_tco",
        "gap_bet_c_diesel",
        "gap_bet_s_diesel",
        "gap_bet_s_bet_c",
    ]

    summary_rows = []
    for m in metrics:
        summary_rows.append({
            "metric": m,
            "mean": df[m].mean(),
            "median": df[m].median(),
            "p5": df[m].quantile(0.05),
            "p95": df[m].quantile(0.95),
            "min": df[m].min(),
            "max": df[m].max(),
        })

    summary_df = pd.DataFrame(summary_rows)

    probability_rows = [
        {
            "metric": "P(BET-C Rec - Diesel < 0)",
            "probability": (df["gap_bet_c_diesel"] < 0).mean(),
        },
        {
            "metric": "P(BET-S Rec - Diesel < 0)",
            "probability": (df["gap_bet_s_diesel"] < 0).mean(),
        },
        {
            "metric": "P(BET-S Rec - BET-C Rec < 0)",
            "probability": (df["gap_bet_s_bet_c"] < 0).mean(),
        },
    ]
    probability_df = pd.DataFrame(probability_rows)

    return summary_df, probability_df

################################# Future Cost Projection ############################################
# Update a parameter in a dataclass 
def update_input(obj, field_name, new_value):
    return replace(obj, **{field_name: new_value})

# Apply piecewise annual percentage change to a parameter
def apply_annual_change(base_value, target_year, base_year, rate_to_2030, rate_to_2040):
    """
    Apply piecewise annual percentage change from base_year to target_year.
    Example:
    - 2026->2030 use rate_to_2030 each year
    - 2031->2040 use rate_to_2040 each year
    """
    value = base_value

    for y in range(base_year + 1, target_year + 1):
        if y <= 2030:
            value *= (1 + rate_to_2030)
        elif y <= 2040:
            value *= (1 + rate_to_2040)

    return value

# Apply fixed annual increment to a parameter
def apply_annual_increment(base_value, target_year, base_year, increment_per_year):
    """
    Apply fixed additive annual increment from base_year to target_year.
    """
    years_passed = target_year - base_year
    return base_value + years_passed * increment_per_year

# Projected inputs to change original values for a given future year
def build_projected_inputs_for_year(
    target_year,
    base_year=2026,
    shared=None,
    diesel_inp=None,
    betc_inp=None,
    bets_inp=None,
):
    if shared is None:
        shared = SharedInputs()
    if diesel_inp is None:
        diesel_inp = DieselInputs()
    if betc_inp is None:
        betc_inp = BETCInputs(
            battery_recycle_ratio=shared.battery_recycle_value_ratio
        )
    if bets_inp is None:
        bets_inp = BETSInputs(
            battery_recycle_ratio=shared.battery_recycle_value_ratio
        )

    # ===== Diesel =====
    diesel_proj = replace(
        diesel_inp,
        capex=apply_annual_change(
            diesel_inp.capex,
            target_year,
            base_year,
            rate_to_2030=0.01,   # 2027-2030  +1% each year
            rate_to_2040=0.005,   # 2031-2040  +0.5%
        ),
        fuel_economy_full_loaded_year1_l_per_km=apply_annual_change(
            diesel_inp.fuel_economy_full_loaded_year1_l_per_km,
            target_year,
            base_year,
            rate_to_2030=-0.026,  # each year - 2.6%
            rate_to_2040=-0.013,  # - 1.3%
        ),
    )

    # ===== BET-C =====
    betc_proj = replace(
        betc_inp,
        glider_capex=apply_annual_change(
            betc_inp.glider_capex,
            target_year,
            base_year,
            rate_to_2030=-0.07,    # - 7%
            rate_to_2040=-0.035,   # - 3.5%
        ),
        battery_price_per_kwh=apply_annual_change(
            betc_inp.battery_price_per_kwh,
            target_year,
            base_year,
            rate_to_2030=-0.07,
            rate_to_2040=-0.035,
        ),
        battery_lifetime_cycles=apply_annual_increment(
            betc_inp.battery_lifetime_cycles,
            target_year,
            base_year,
            increment_per_year=200,
        ),
        full_loaded_kwh_per_km_year1=apply_annual_change(
            betc_inp.full_loaded_kwh_per_km_year1,
            target_year,
            base_year,
            rate_to_2030=-0.031,   # - 3.1%
            rate_to_2040=-0.015,   # - 1.5%
        ),
    )

    # ===== BET-S =====
    bets_proj = replace(
        bets_inp,
        glider_capex=apply_annual_change(
            bets_inp.glider_capex,
            target_year,
            base_year,
            rate_to_2030=-0.07,
            rate_to_2040=-0.035,
        ),
        battery_price_per_kwh=apply_annual_change(
            bets_inp.battery_price_per_kwh,
            target_year,
            base_year,
            rate_to_2030=-0.07,
            rate_to_2040=-0.035,
        ),
        battery_lifetime_cycles=apply_annual_increment(
            bets_inp.battery_lifetime_cycles,
            target_year,
            base_year,
            increment_per_year=200,
        ),
        full_loaded_kwh_per_km_year1=apply_annual_change(
            bets_inp.full_loaded_kwh_per_km_year1,
            target_year,
            base_year,
            rate_to_2030=-0.031,
            rate_to_2040=-0.015,
        ),
    )

    return shared, diesel_proj, betc_proj, bets_proj

# Run future TCO projection
def run_tco_projection(
    start_year=2026,
    end_year=2040,
    shared=None,
    diesel_inp=None,
    betc_inp=None,
    bets_inp=None,
):
    rows = []

    if shared is None:
        shared = SharedInputs()
    if diesel_inp is None:
        diesel_inp = DieselInputs()
    if betc_inp is None:
        betc_inp = BETCInputs(
            battery_recycle_ratio=shared.battery_recycle_value_ratio
        )
    if bets_inp is None:
        bets_inp = BETSInputs(
            battery_recycle_ratio=shared.battery_recycle_value_ratio
        )

    for year in range(start_year, end_year + 1):
        shared_i, diesel_i, betc_i, bets_i = build_projected_inputs_for_year(
            target_year=year,
            base_year=start_year,
            shared=shared,
            diesel_inp=diesel_inp,
            betc_inp=betc_inp,
            bets_inp=bets_inp,
        )

        results = run_model(
            shared=shared_i,
            diesel_inp=diesel_i,
            betc_inp=betc_i,
            bets_inp=bets_i,
        )

        rows.append({
            "year": year,

            # ===== 5-year discounted TCO =====
            "diesel_tco_5y_discounted": results["diesel"]["tco_5y_discounted"],
            "betc_tco_5y_discounted": results["bet_c"]["tco_5y_discounted_recycle"],
            "bets_tco_5y_discounted": results["bet_s"]["tco_5y_discounted_recycle"],

            # ===== per km TCO =====
            "diesel_tco_per_km": results["diesel"]["tco_per_km_discounted"],
            "betc_tco_per_km": results["bet_c"]["tco_per_km_discounted_recycle"],
            "bets_tco_per_km": results["bet_s"]["tco_per_km_discounted_recycle"],

            # ===== per kWh TCO =====
            "diesel_tco_per_kwh": results["diesel"]["tco_per_kwh_discounted"],
            "betc_tco_per_kwh": results["bet_c"]["tco_per_kwh_discounted_recycle"],
            "bets_tco_per_kwh": results["bet_s"]["tco_per_kwh_discounted_recycle"],

            # ===== optional: save projected inputs too =====
            "diesel_capex": diesel_i.capex,
            "diesel_year1_l_per_km": diesel_i.fuel_economy_full_loaded_year1_l_per_km,

            "betc_glider_capex": betc_i.glider_capex,
            "betc_battery_price_per_kwh": betc_i.battery_price_per_kwh,
            "betc_battery_lifetime_cycles": betc_i.battery_lifetime_cycles,
            "betc_year1_kwh_per_km": betc_i.full_loaded_kwh_per_km_year1,

            "bets_glider_capex": bets_i.glider_capex,
            "bets_battery_price_per_kwh": bets_i.battery_price_per_kwh,
            "bets_battery_lifetime_cycles": bets_i.battery_lifetime_cycles,
            "bets_year1_kwh_per_km": bets_i.full_loaded_kwh_per_km_year1,
        })

    return pd.DataFrame(rows)



def summarize_projection_uncertainty(df, metric_cols=None):
    if metric_cols is None:
        metric_cols = [
            "diesel_tco_5y_discounted",
            "betc_tco_5y_discounted",
            "bets_tco_5y_discounted",
        ]

    rows = []

    for year, group in df.groupby("year"):
        row = {"year": year}

        for col in metric_cols:
            row[f"{col}_p5"] = group[col].quantile(0.05)
            row[f"{col}_p50"] = group[col].quantile(0.50)
            row[f"{col}_p95"] = group[col].quantile(0.95)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

# Rank the effects of those uncertainties
def summarize_independent_effect_spread(df):
    """
    Optional summary table:
    compare the spread caused by each variable independently.
    """
    rows = []

    for var, group in df.groupby("variable"):
        rows.append({
            "variable": var,

            "diesel_tco_iqr": group["diesel_tco"].quantile(0.75) - group["diesel_tco"].quantile(0.25),
            "bet_c_tco_iqr": group["bet_c_tco"].quantile(0.75) - group["bet_c_tco"].quantile(0.25),
            "bet_s_tco_iqr": group["bet_s_tco"].quantile(0.75) - group["bet_s_tco"].quantile(0.25),

            "gap_bet_c_diesel_iqr": group["gap_bet_c_diesel"].quantile(0.75) - group["gap_bet_c_diesel"].quantile(0.25),
            "gap_bet_s_diesel_iqr": group["gap_bet_s_diesel"].quantile(0.75) - group["gap_bet_s_diesel"].quantile(0.25),
            "gap_bet_s_bet_c_iqr": group["gap_bet_s_bet_c"].quantile(0.75) - group["gap_bet_s_bet_c"].quantile(0.25),
        })

    out = pd.DataFrame(rows)

    out["max_tco_iqr"] = out[["diesel_tco_iqr", "bet_c_tco_iqr", "bet_s_tco_iqr"]].max(axis=1)
    out["max_gap_iqr"] = out[[
        "gap_bet_c_diesel_iqr",
        "gap_bet_s_diesel_iqr",
        "gap_bet_s_bet_c_iqr"
    ]].max(axis=1)

    return out.sort_values("max_gap_iqr", ascending=False).reset_index(drop=True)
