
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

    # ===== AEaaS granular discount factors =====
    aeaas_glider_cost_factor: float = 0.90
    aeaas_insurance_cost_factor: float = 0.90
    aeaas_station_capex_factor: float = 1.00
    aeaas_site_capex_factor: float = 1.00
    aeaas_battery_depr_factor: float = 1.00
    aeaas_battery_service_factor: float = 1.00
    aeaas_battery_rent_factor: float = 0.90
    aeaas_fixed_swapping_fee_factor: float = 0.90
    aeaas_energy_cost_factor: float = 0.90

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
    "bet_public_energy_price_per_kwh": "BET Public Energy Price per kWh",
    "bet_depot_energy_price_per_kwh": "BET Depot Energy Price per kWh",
    "glider_capex": "Electric Glider CAPEX",
    "battery_price_per_kwh": "Battery Price per kWh"
    
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
def diesel_yearly_fuel_economies(inp: DieselInputs, years: int) -> List[float]:
    vals = [inp.fuel_economy_full_loaded_year1_l_per_km]
    for _ in range(1, years):
        vals.append(vals[-1] * (1 + inp.fuel_economy_growth_rate))
    return vals

# Generate yearly BET-C energy consumption trajectory
def betc_yearly_full_loaded_economies(inp: BETCInputs, years: int) -> List[float]:
    vals = [inp.full_loaded_kwh_per_km_year1]
    for _ in range(1, years):
        vals.append(vals[-1] * (1 + inp.fuel_economy_growth_rate))
    return vals

# Generate yearly BET-S energy consumption trajectory
def bets_yearly_full_loaded_economies(inp: BETSInputs, years: int) -> List[float]:
    vals = [inp.full_loaded_kwh_per_km_year1]
    for _ in range(1, years):
        vals.append(vals[-1] * (1 + inp.fuel_economy_growth_rate))
    return vals

################### TCO calculations ######################################################################
def compute_diesel(shared: SharedInputs, inp: DieselInputs) -> Dict[str, float]:
    years = shared.years
    df = discount_factors(shared.discount_rate, years)
    full_km, unladen_km = diesel_daily_distances(shared)
    fuel_full = diesel_yearly_fuel_economies(inp, years)
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

    total_total_diesel_litre = sum(daily_use) * shared.operational_days_per_year
    total_total_energy_kwh = total_total_diesel_litre * inp.litre_to_kwh

    return {
        "tco_undiscounted": tco_undiscounted,
        "tco_discounted": tco_discounted,
        "tco_per_year_discounted": tco_discounted / years,
        "tco_per_km_discounted": tco_discounted / (annual_km * years),
        "tco_per_kwh_discounted": tco_discounted / total_total_energy_kwh,
        "annual_km": annual_km,
        "daily_energy_year1_l": daily_use[0],
        "truck_residual": truck_residual,
    }


def compute_bet_c(shared: SharedInputs, inp: BETCInputs, asset_manager_margin: float = 0.10) -> Dict[str, float]:
    years = shared.years
    df = discount_factors(shared.discount_rate, years)
    full_km, unladen_km = diesel_daily_distances(shared)
    econ_full = betc_yearly_full_loaded_economies(inp, years)
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
        "tco_discounted_eol": tco_discounted_eol,
        "tco_discounted_recycle": tco_discounted_recycle,
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
    econ_full = bets_yearly_full_loaded_economies(inp, years)
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

    station_capex_depr_per_truck_total = (
        (inp.station_capex / inp.station_lifetime_years)
        / (expected_station_service_demand * shared.operational_days_per_year)
        * shared.operational_days_per_year
        * years
    )

    site_capex_depr_per_truck_total = (
        (inp.site_capex / inp.station_lifetime_years)
        / (expected_station_service_demand * shared.operational_days_per_year)
        * shared.operational_days_per_year
        * years
    )

    station_depr_per_truck_total = (
        station_capex_depr_per_truck_total
        + site_capex_depr_per_truck_total
    )
    
    battery_value_station = inp.station_battery_bays * inp.battery_pack_capacity_kwh * inp.battery_price_per_kwh
    battery_value_truck = expected_station_service_demand * inp.battery_packs_per_truck * inp.battery_pack_capacity_kwh * inp.battery_price_per_kwh
    battery_system_value = battery_value_station + battery_value_truck

    battery_depr_eol_total = (battery_system_value * (1 - inp.battery_eol_ratio) * (shared.operational_days_per_year * years / inp.battery_lifetime_cycles)) / expected_station_service_demand
    battery_depr_recycle_total = (battery_system_value * (1 - inp.battery_recycle_ratio) * (shared.operational_days_per_year * years / inp.battery_lifetime_cycles)) / expected_station_service_demand

    battery_service_total = years * inp.annual_battery_service_cost
    battery_rent_total = inp.battery_rent_per_month_ex_depreciation * 12 * years
    fixed_swapping_total = inp.swapping_fee_flat * inp.swaps_per_day * shared.operational_days_per_year * years

    base_energy_price = shared.peak_price_per_kwh * (1 - shared.off_peak_share) + shared.off_peak_price_per_kwh * shared.off_peak_share
    energy_service_costs = [daily * inp.swaps_per_day * shared.operational_days_per_year * base_energy_price for daily in daily_kwh]
    energy_margin_addition = sum(energy_service_costs) * shared.electricity_margin

    discounted_capex = inp.glider_capex - glider_residual * df[-1]
    discounted_operating = annual_operating_cost * sum(df)
    discounted_baas_common = ((station_depr_per_truck_total + battery_service_total + battery_rent_total + fixed_swapping_total) / years) * sum(df)
    discounted_energy = sum(cost * (1 + shared.electricity_margin) * w for cost, w in zip(energy_service_costs, df))

    tco_discounted_eol = (
        discounted_capex
        + discounted_operating
        + discounted_baas_common
        + (battery_depr_eol_total / years) * sum(df)
        + discounted_energy
        - shared.bet_subsidy
    )
    tco_discounted_recycle = (
        discounted_capex
        + discounted_operating
        + discounted_baas_common
        + (battery_depr_recycle_total / years) * sum(df)
        + discounted_energy
        - shared.bet_subsidy
    )

    # ===== AEaaS supplier discounted cost base (BET-S only) =====
    discounted_driver_cost_total = annual_salary * sum(df)
    discounted_insurance_total = insurance * sum(df)

    discounted_glider_cost_for_aeaas = (
        (inp.glider_capex - glider_residual * df[-1])
        * shared.aeaas_glider_cost_factor
    )

    discounted_insurance_for_aeaas = (
        discounted_insurance_total
        * shared.aeaas_insurance_cost_factor
    )

    discounted_station_capex_for_aeaas = (
        ((station_capex_depr_per_truck_total / years) * sum(df))
        * shared.aeaas_station_capex_factor
    )

    discounted_site_capex_for_aeaas = (
        ((site_capex_depr_per_truck_total / years) * sum(df))
        * shared.aeaas_site_capex_factor
    )

    discounted_battery_service_for_aeaas = (
        ((battery_service_total / years) * sum(df))
        * shared.aeaas_battery_service_factor
    )

    discounted_battery_rent_for_aeaas = (
        ((battery_rent_total / years) * sum(df))
        * shared.aeaas_battery_rent_factor
    )

    discounted_fixed_swapping_for_aeaas = (
        ((fixed_swapping_total / years) * sum(df))
        * shared.aeaas_fixed_swapping_fee_factor
    )

    discounted_battery_depr_for_aeaas = (
        ((battery_depr_recycle_total / years) * sum(df))
        * shared.aeaas_battery_depr_factor
    )

    discounted_energy_for_aeaas = (
        discounted_energy
        * shared.aeaas_energy_cost_factor
    )

    # supplier bears everything except driver cost
    aeaas_asset_service_cost_total = (
        discounted_glider_cost_for_aeaas
        + discounted_insurance_for_aeaas
        + discounted_station_capex_for_aeaas
        + discounted_site_capex_for_aeaas
        + discounted_battery_service_for_aeaas
        + discounted_battery_rent_for_aeaas
        + discounted_fixed_swapping_for_aeaas
        + discounted_battery_depr_for_aeaas
        + discounted_energy_for_aeaas
    )

    asset_service = compute_asset_service_unit_prices(
        asset_service_cost_total=aeaas_asset_service_cost_total,
        annual_driver_cost=annual_salary,
        annual_km=annual_km,
        daily_energy_list=daily_kwh,
        shared=shared,
        margin=asset_manager_margin,
    )

    aas_gap_vs_own_tco = asset_service["freight_total_cost_total"] - tco_discounted_recycle
    
    return {
        "tco_discounted_eol": tco_discounted_eol,
        "tco_discounted_recycle": tco_discounted_recycle,
        "tco_per_year_discounted_eol": tco_discounted_eol / years,
        "tco_per_km_discounted_eol": tco_discounted_eol / (annual_km * years),
        "tco_per_kwh_discounted_eol": tco_discounted_eol / (shared.operational_days_per_year * sum(daily_kwh)),
        "tco_per_year_discounted_recycle": tco_discounted_recycle / years,
        "tco_per_km_discounted_recycle": tco_discounted_recycle / (annual_km * years),
        "tco_per_kwh_discounted_recycle": tco_discounted_recycle / (shared.operational_days_per_year * sum(daily_kwh)),
        "annual_km": annual_km,
        "daily_energy_year1_kwh": daily_kwh[0],
        "energy_margin_addition_total": energy_margin_addition,
        "annual_driver_cost": annual_salary,
        "daily_kwh_by_year": daily_kwh,
        **asset_service,
        "discounted_glider_cost_for_aeaas": discounted_glider_cost_for_aeaas,
        "discounted_insurance_for_aeaas": discounted_insurance_for_aeaas,
        "discounted_station_capex_for_aeaas": discounted_station_capex_for_aeaas,
        "discounted_site_capex_for_aeaas": discounted_site_capex_for_aeaas,
        "discounted_battery_service_for_aeaas": discounted_battery_service_for_aeaas,
        "discounted_battery_rent_for_aeaas": discounted_battery_rent_for_aeaas,
        "discounted_fixed_swapping_for_aeaas": discounted_fixed_swapping_for_aeaas,
        "discounted_battery_depr_for_aeaas": discounted_battery_depr_for_aeaas,
        "discounted_energy_for_aeaas": discounted_energy_for_aeaas,
        "aeaas_asset_service_cost_total": aeaas_asset_service_cost_total,
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
        results["bet_c"]["tco_discounted_recycle"]
        - results["diesel"]["tco_discounted"]
    )
    bet_s_vs_diesel = (
        results["bet_s"]["tco_discounted_recycle"]
        - results["diesel"]["tco_discounted"]
    )
    bet_s_vs_bet_c = (
        results["bet_s"]["tco_discounted_recycle"]
        - results["bet_c"]["tco_discounted_recycle"]
    )

    return {
        "bet_c_vs_diesel": bet_c_vs_diesel,
        "bet_s_vs_diesel": bet_s_vs_diesel,
        "bet_s_vs_bet_c": bet_s_vs_bet_c,
    }


############ AEaaS Pricing Model  ################################################################
# Convert asset-service cost into unit prices (per km / per kWh) with margin
def compute_asset_service_unit_prices(
    asset_service_cost_total: float,
    annual_driver_cost: float,
    annual_km: float,
    daily_energy_list: list[float],
    shared: SharedInputs,
    margin: float = 0.10,
) -> Dict[str, float]:
    years = shared.years
    df = discount_factors(shared.discount_rate, years)

    discounted_driver_cost_total = annual_driver_cost * sum(df)

    total_km = annual_km * years
    total_kwh = sum(daily_energy_list) * shared.operational_days_per_year

    # asset manager cost base
    unit_cost_per_km = asset_service_cost_total / total_km
    unit_cost_per_kwh = asset_service_cost_total / total_kwh

    # asset manager selling price
    price_per_km_with_margin = unit_cost_per_km * (1 + margin)
    price_per_kwh_with_margin = unit_cost_per_kwh * (1 + margin)

    asset_price_total = asset_service_cost_total * (1 + margin)

    # driver cost borne by freight company
    driver_cost_per_km = discounted_driver_cost_total / total_km
    driver_cost_per_kwh = discounted_driver_cost_total / total_kwh

    # freight company's all-in effective unit cost
    freight_total_cost_per_km = price_per_km_with_margin + driver_cost_per_km
    freight_total_cost_per_kwh = price_per_kwh_with_margin + driver_cost_per_kwh

    # freight company's all-in total cost
    freight_total_cost_total = asset_price_total + discounted_driver_cost_total

    freight_total_cost_total_from_km = freight_total_cost_per_km * total_km
    freight_total_cost_total_from_kwh = freight_total_cost_per_kwh * total_kwh

    return {
        "discounted_driver_cost_total": discounted_driver_cost_total,
        "asset_service_cost_total": asset_service_cost_total,
        "asset_price_total": asset_price_total,

        "total_km": total_km,
        "total_kwh": total_kwh,

        "unit_cost_per_km": unit_cost_per_km,
        "unit_cost_per_kwh": unit_cost_per_kwh,

        "price_per_km_with_margin": price_per_km_with_margin,
        "price_per_kwh_with_margin": price_per_kwh_with_margin,

        "driver_cost_per_km": driver_cost_per_km,
        "driver_cost_per_kwh": driver_cost_per_kwh,

        "freight_total_cost_per_km": freight_total_cost_per_km,
        "freight_total_cost_per_kwh": freight_total_cost_per_kwh,

        "freight_total_cost_total": freight_total_cost_total,
        "freight_total_cost_total_from_km": freight_total_cost_total_from_km,
        "freight_total_cost_total_from_kwh": freight_total_cost_total_from_kwh,
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
    shared=None,
    diesel_inp=None,
    betc_inp=None,
    bets_inp=None,
    uncertainty_overrides=None,
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

    rng = np.random.default_rng(random_seed)
    specs = get_uncertainty_specs(shared, diesel_inp, betc_inp, bets_inp, uncertainty_overrides)
    rows = []

    for margin in margins:
        for i in range(n_runs):
            shared_i = shared
            diesel_i = diesel_inp
            betc_i = betc_inp
            bets_i = bets_inp

            for spec in specs:
                sampled_value = sample_triangular(spec["left"], spec["mode"], spec["right"], rng)
                shared_i, diesel_i, betc_i, bets_i = apply_single_variable_change(
                    shared_i, diesel_i, betc_i, bets_i, spec, sampled_value
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
            "glider_capex",
            "battery_price_per_kwh",
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

    # 
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

        # 
        if "shared" in target_classes:
            shared_i = update_input(shared_i, variable_name, new_value)

        if "diesel" in target_classes:
            diesel_i = update_input(diesel_i, variable_name, new_value)

        if "betc" in target_classes:
            betc_i = update_input(betc_i, variable_name, new_value)

        if "bets" in target_classes:
            bets_i = update_input(bets_i, variable_name, new_value)

        # 
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


def _apply_uncertainty_overrides(specs, uncertainty_overrides=None):
    """
    Override the Min/Max bounds of Monte Carlo uncertainty specs.

    The Mode is still taken from the current model inputs. If a user-entered
    bound crosses the Mode, it is adjusted so np.random.triangular remains valid.
    """
    if not uncertainty_overrides:
        return specs

    updated_specs = []
    for spec in specs:
        spec_i = dict(spec)
        override = uncertainty_overrides.get(spec_i["variable"], {}) if isinstance(uncertainty_overrides, dict) else {}

        if "left" in override and override["left"] is not None:
            spec_i["left"] = float(override["left"])
        if "right" in override and override["right"] is not None:
            spec_i["right"] = float(override["right"])

        # Keep triangular distribution valid: left <= mode <= right
        if spec_i["left"] > spec_i["mode"]:
            spec_i["left"] = spec_i["mode"]
        if spec_i["right"] < spec_i["mode"]:
            spec_i["right"] = spec_i["mode"]

        updated_specs.append(spec_i)

    return updated_specs

# Define uncertain variables and their distributions
def get_uncertainty_specs(shared=None, diesel_inp=None, betc_inp=None, bets_inp=None, uncertainty_overrides=None):
    """
    Define uncertain variables and triangular distributions.

    When inputs are supplied from the Streamlit app, the Mode values are taken
    from those current inputs so the displayed Monte Carlo range table and the
    Monte Carlo sampling both follow the sidebar assumptions.
    """
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

    def pct_range(mode, low_factor, high_factor, min_value=None, max_value=None):
        left = mode * low_factor
        right = mode * high_factor
        if min_value is not None:
            left = max(min_value, left)
        if max_value is not None:
            right = min(max_value, right)
        if left > mode:
            left = mode
        if right < mode:
            right = mode
        return left, mode, right

    discount_left, discount_mode, discount_right = pct_range(shared.discount_rate, 0.8, 1.2, min_value=0.0)
    vkt_left, vkt_mode, vkt_right = pct_range(shared.full_loaded_km_per_day, 0.8, 1.2, min_value=0.0)
    peak_left, peak_mode, peak_right = pct_range(shared.peak_price_per_kwh, 0.8, 1.2, min_value=0.0)
    offpeak_left, offpeak_mode, offpeak_right = pct_range(shared.off_peak_share, 0.6, 1.4, min_value=0.0, max_value=1.0)
    depot_left, depot_mode, depot_right = pct_range(shared.bet_depot_energy_price_per_kwh, 0.8, 1.25, min_value=0.0)
    public_left, public_mode, public_right = pct_range(shared.bet_public_energy_price_per_kwh, 0.8, 1.25, min_value=0.0)
    kwh_left, kwh_mode, kwh_right = pct_range(betc_inp.full_loaded_kwh_per_km_year1, 0.88, 1.13, min_value=0.0)
    glider_left, glider_mode, glider_right = pct_range(betc_inp.glider_capex, 0.8, 1.2, min_value=0.0)
    battery_price_left, battery_price_mode, battery_price_right = pct_range(betc_inp.battery_price_per_kwh, 0.8, 1.2, min_value=0.0)
    recycle_left, recycle_mode, recycle_right = pct_range(betc_inp.battery_recycle_ratio, 0.5, 2.0, min_value=0.0, max_value=1.0)
    cycle_left, cycle_mode, cycle_right = pct_range(betc_inp.battery_lifetime_cycles, 0.8, 1.5, min_value=1.0)
    unladen_left, unladen_mode, unladen_right = pct_range(betc_inp.unladen_energy_saving, 0.8, 1.2, min_value=0.0, max_value=1.0)
    cap_left, cap_mode, cap_right = pct_range(betc_inp.battery_capacity_kwh, 400.0 / 621.0, 800.0 / 621.0, min_value=1.0)
    subsidy_left = 0.0
    subsidy_mode = shared.bet_subsidy
    subsidy_right = max(120000.0, subsidy_mode * 1.2)

    specs = [
        {"variable": "discount_rate", "target_class": "shared", "left": discount_left, "mode": discount_mode, "right": discount_right},
        {"variable": "full_loaded_km_per_day", "target_class": "shared", "left": vkt_left, "mode": vkt_mode, "right": vkt_right},
        {"variable": "peak_price_per_kwh", "target_class": "shared", "left": peak_left, "mode": peak_mode, "right": peak_right},
        {"variable": "off_peak_share", "target_class": "shared", "left": offpeak_left, "mode": offpeak_mode, "right": offpeak_right},
        {"variable": "bet_depot_energy_price_per_kwh", "target_class": "shared", "left": depot_left, "mode": depot_mode, "right": depot_right},
        {"variable": "bet_public_energy_price_per_kwh", "target_class": "shared", "left": public_left, "mode": public_mode, "right": public_right},
        {"variable": "full_loaded_kwh_per_km_year1", "target_class": ["betc", "bets"], "left": kwh_left, "mode": kwh_mode, "right": kwh_right},
        {"variable": "glider_capex", "target_class": ["betc", "bets"], "left": glider_left, "mode": glider_mode, "right": glider_right},
        {"variable": "battery_price_per_kwh", "target_class": ["betc", "bets"], "left": battery_price_left, "mode": battery_price_mode, "right": battery_price_right},
        {"variable": "battery_recycle_ratio", "target_class": ["betc", "bets"], "left": recycle_left, "mode": recycle_mode, "right": recycle_right},
        {"variable": "battery_lifetime_cycles", "target_class": ["betc", "bets"], "left": cycle_left, "mode": cycle_mode, "right": cycle_right},
        {"variable": "unladen_energy_saving", "target_class": ["betc", "bets"], "left": unladen_left, "mode": unladen_mode, "right": unladen_right},
        {"variable": "battery_capacity_kwh", "target_class": "betc", "left": cap_left, "mode": cap_mode, "right": cap_right},
        {"variable": "bet_subsidy", "target_class": "shared", "left": subsidy_left, "mode": subsidy_mode, "right": subsidy_right},
    ]

    return _apply_uncertainty_overrides(specs, uncertainty_overrides)

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
def run_independent_variable_monte_carlo(
    n_runs=500,
    random_seed=42,
    shared=None,
    diesel_inp=None,
    betc_inp=None,
    bets_inp=None,
    uncertainty_overrides=None,
):
    """
    For each uncertain variable:
    - vary ONLY that variable according to its triangular distribution
    - keep all other variables at the current baseline inputs
    - run model n_runs times
    Returns a long dataframe for boxplotting.
    """
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

    rng = np.random.default_rng(random_seed)
    specs = get_uncertainty_specs(shared, diesel_inp, betc_inp, bets_inp, uncertainty_overrides)

    rows = []

    for spec in specs:
        var_name = spec["variable"]

        for i in range(n_runs):
            sampled_value = sample_triangular(spec["left"], spec["mode"], spec["right"], rng)

            shared_i, diesel_i, betc_i, bets_i = apply_single_variable_change(
                shared, diesel_inp, betc_inp, bets_inp, spec, sampled_value
            )

            results = run_model(shared=shared_i, diesel_inp=diesel_i, betc_inp=betc_i, bets_inp=bets_i)

            diesel_tco = results["diesel"]["tco_discounted"]
            betc_tco = results["bet_c"]["tco_discounted_recycle"]
            bets_tco = results["bet_s"]["tco_discounted_recycle"]

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

def run_monte_carlo_simulation(
    n_runs=500,
    random_seed=42,
    shared=None,
    diesel_inp=None,
    betc_inp=None,
    bets_inp=None,
    uncertainty_overrides=None,
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

    rng = np.random.default_rng(random_seed)
    specs = get_uncertainty_specs(shared, diesel_inp, betc_inp, bets_inp, uncertainty_overrides)
    rows = []

    for i in range(n_runs):
        shared_i = shared
        diesel_i = diesel_inp
        betc_i = betc_inp
        bets_i = bets_inp
        sampled = {}

        for spec in specs:
            sampled_value = sample_triangular(spec["left"], spec["mode"], spec["right"], rng)
            sampled[spec["variable"]] = sampled_value
            shared_i, diesel_i, betc_i, bets_i = apply_single_variable_change(
                shared_i, diesel_i, betc_i, bets_i, spec, sampled_value
            )

        results = run_model(shared=shared_i, diesel_inp=diesel_i, betc_inp=betc_i, bets_inp=bets_i)

        diesel_tco = results["diesel"]["tco_discounted"]
        bet_c_tco = results["bet_c"]["tco_discounted_recycle"]
        bet_s_tco = results["bet_s"]["tco_discounted_recycle"]

        rows.append({
            "iteration": i + 1,
            "discount_rate": sampled.get("discount_rate"),
            "full_loaded_km_per_day": sampled.get("full_loaded_km_per_day"),
            "peak_price_per_kwh": sampled.get("peak_price_per_kwh"),
            "off_peak_share": sampled.get("off_peak_share"),
            "bet_depot_energy_price_per_kwh": sampled.get("bet_depot_energy_price_per_kwh"),
            "bet_public_energy_price_per_kwh": sampled.get("bet_public_energy_price_per_kwh"),
            "full_loaded_kwh_per_km_year1": sampled.get("full_loaded_kwh_per_km_year1"),
            "glider_capex": sampled.get("glider_capex"),
            "battery_price_per_kwh": sampled.get("battery_price_per_kwh"),
            "battery_recycle_ratio": sampled.get("battery_recycle_ratio"),
            "battery_lifetime_cycles": sampled.get("battery_lifetime_cycles"),
            "unladen_energy_saving": sampled.get("unladen_energy_saving"),
            "battery_capacity_kwh": sampled.get("battery_capacity_kwh"),
            "bet_subsidy": sampled.get("bet_subsidy"),
            "diesel_tco": diesel_tco,
            "bet_c_tco": bet_c_tco,
            "bet_s_tco": bet_s_tco,
            "gap_bet_c_diesel": bet_c_tco - diesel_tco,
            "gap_bet_s_diesel": bet_s_tco - diesel_tco,
            "gap_bet_s_bet_c": bet_s_tco - bet_c_tco,
        })

    return pd.DataFrame(rows)

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

def run_projection_monte_carlo(
    start_year=2026,
    end_year=2040,
    n_runs=500,
    random_seed=42,
    shared=None,
    diesel_inp=None,
    betc_inp=None,
    bets_inp=None,
    uncertainty_overrides=None,
):
    """
    For each purchase year:
    1. build projected baseline inputs for that year
    2. run Monte Carlo around that year's projected values
    3. collect TCO distributions
    """
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

    rng = np.random.default_rng(random_seed)
    rows = []

    for year in range(start_year, end_year + 1):
        shared_base, diesel_base, betc_base, bets_base = build_projected_inputs_for_year(
            target_year=year,
            base_year=start_year,
            shared=shared,
            diesel_inp=diesel_inp,
            betc_inp=betc_inp,
            bets_inp=bets_inp,
        )

        specs = get_uncertainty_specs(shared_base, diesel_base, betc_base, bets_base, uncertainty_overrides)

        for i in range(n_runs):
            shared_i = shared_base
            diesel_i = diesel_base
            betc_i = betc_base
            bets_i = bets_base

            for spec in specs:
                sampled_value = sample_triangular(spec["left"], spec["mode"], spec["right"], rng)
                shared_i, diesel_i, betc_i, bets_i = apply_single_variable_change(
                    shared_i, diesel_i, betc_i, bets_i, spec, sampled_value
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

                "diesel_tco_discounted": results["diesel"]["tco_discounted"],
                "betc_tco_discounted": results["bet_c"]["tco_discounted_recycle"],
                "bets_tco_discounted": results["bet_s"]["tco_discounted_recycle"],

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


def summarize_projection_uncertainty(df, metric_cols=None):
    if metric_cols is None:
        metric_cols = [
            "diesel_tco_discounted",
            "betc_tco_discounted",
            "bets_tco_discounted",
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
            "diesel_tco_discounted": results["diesel"]["tco_discounted"],
            "betc_tco_discounted": results["bet_c"]["tco_discounted_recycle"],
            "bets_tco_discounted": results["bet_s"]["tco_discounted_recycle"],

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

################## Visualisations #################################################################
def plot_tco_comparison(results):
    labels = ["Diesel", "BET-C", "BET-S"]
    values = [
        results["diesel"]["tco_discounted"],
        results["bet_c"]["tco_discounted_recycle"],
        results["bet_s"]["tco_discounted_recycle"],
    ]

    plt.figure()
    bars = plt.bar(labels, values)

    plt.title("Discounted TCO Comparison")
    plt.ylabel("TCO (£)")
    plt.xlabel("Truck Type")

    for bar, v in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:,.0f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()


def plot_tco_gap(results):
    labels = [
        "BET-C - Diesel",
        "BET-S - Diesel",
        "BET-S - BET-C"
    ]

    bet_c_gap = (
        results["bet_c"]["tco_discounted_recycle"]
        - results["diesel"]["tco_discounted"]
    )

    bet_s_gap = (
        results["bet_s"]["tco_discounted_recycle"]
        - results["diesel"]["tco_discounted"]
    )

    bet_s_vs_bet_c_gap = (
        results["bet_s"]["tco_discounted_recycle"]
        - results["bet_c"]["tco_discounted_recycle"]
    )

    values = [bet_c_gap, bet_s_gap, bet_s_vs_bet_c_gap]

    plt.figure()
    bars = plt.bar(labels, values)

    plt.title("TCO Gaps")
    plt.ylabel("Difference (£)")
    plt.xlabel("Comparison")
    plt.axhline(0)

    for bar, v in zip(bars, values):
        if v >= 0:
            va = "bottom"
        else:
            va = "top"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:,.0f}",
            ha="center",
            va=va
        )

    plt.tight_layout()
    plt.show()


# Plot projected TCO
def plot_tco_projection(df):
    plt.figure(figsize=(10, 6))

    plt.plot(df["year"], df["diesel_tco_discounted"], marker="o", label="Diesel")
    plt.plot(df["year"], df["betc_tco_discounted"], marker="o", label="BET-C")
    plt.plot(df["year"], df["bets_tco_discounted"], marker="o", label="BET-S")

    plt.title("Projected Discounted TCO (2026-2040)")
    plt.xlabel("Year")
    plt.ylabel("Discounted TCO (£)")
    plt.xticks(df["year"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_tco_per_km_projection(df):
    plt.figure(figsize=(10, 6))

    plt.plot(df["year"], df["diesel_tco_per_km"], marker="o", label="Diesel")
    plt.plot(df["year"], df["betc_tco_per_km"], marker="o", label="BET-C")
    plt.plot(df["year"], df["bets_tco_per_km"], marker="o", label="BET-S")

    plt.title("Projected Discounted TCO per km (2026-2040)")
    plt.xlabel("Purchase Year")
    plt.ylabel("Discounted TCO (£/km)")
    plt.xticks(df["year"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_tco_per_kwh_projection(df):
    plt.figure(figsize=(10, 6))

    plt.plot(df["year"], df["diesel_tco_per_kwh"], marker="o", label="Diesel")
    plt.plot(df["year"], df["betc_tco_per_kwh"], marker="o", label="BET-C")
    plt.plot(df["year"], df["bets_tco_per_kwh"], marker="o", label="BET-S")

    plt.title("Projected Discounted TCO per kWh (2026-2040)")
    plt.xlabel("Purchase Year")
    plt.ylabel("Discounted TCO (£/kWh)")
    plt.xticks(df["year"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sensitivity_bar(sensitivity_results, title=None):

    name_map = {
                "battery_price_per_kwh": "Battery price (£/kWh)",
                "full_loaded_km_per_day": "Full-loaded daily mileage (km/day)",
                "diesel_public_price_per_l": "Diesel price (£/L)",
                "discount_rate": "Discount rate (%)",
                "bet_depot_energy_price_per_kwh": "Electricity price (£/kWh)",
    }
            
    labels = sensitivity_results["labels"]
    bet_c_vs_diesel = sensitivity_results["bet_c_vs_diesel"]
    bet_s_vs_diesel = sensitivity_results["bet_s_vs_diesel"]
    bet_s_vs_bet_c = sensitivity_results["bet_s_vs_bet_c"]

    x = range(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(
        [i - width for i in x],
        bet_c_vs_diesel,
        width=width,
        label="BET-C - Diesel"
    )
    bars2 = plt.bar(
        x,
        bet_s_vs_diesel,
        width=width,
        label="BET-S - Diesel"
    )
    bars3 = plt.bar(
        [i + width for i in x],
        bet_s_vs_bet_c,
        width=width,
        label="BET-S - BET-C"
    )

    plt.axhline(0)
    plt.xticks(list(x), labels)
    plt.xlabel("Change from baseline")
    plt.ylabel("TCO Gap (£)")

    if title is None:
        var_name = sensitivity_results["variable_name"]
        base_value = sensitivity_results["base_value"]

        display_name = name_map.get(var_name, var_name)

        if "price" in var_name or "cost" in var_name:
            base_str = f"£{base_value:.2f}"
        elif "rate" in var_name:
            base_str = f"{base_value*100:.1f}%"
        elif "km" in var_name:
            base_str = f"{base_value:.0f} km"
        else:
            base_str = f"{base_value}"

        title = f"Sensitivity Analysis: {display_name} (base = {base_str})"

    plt.title(title)

    plt.legend()

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            v = bar.get_height()
            va = "bottom" if v >= 0 else "top"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                v,
                f"{v:,.0f}",
                ha="center",
                va=va,
                fontsize=8
            )

    plt.tight_layout()
    plt.show()

# Summarize uncertainty results using percentiles for plots
def summarize_margin_uncertainty(df):
    rows = []

    for margin, group in df.groupby("asset_manager_margin"):
        rows.append({
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
        })

    return pd.DataFrame(rows).sort_values("asset_manager_margin").reset_index(drop=True)

# Plot margin vs cost with uncertainty bands
def plot_margin_vs_freight_all_in_per_km_with_uncertainty(summary_df):
    plt.figure(figsize=(10, 6))

    x = summary_df["asset_manager_margin"] * 100

    # Diesel
    plt.plot(
        x,
        summary_df["diesel_p50"],
        marker="o",
        color="tab:blue",
        label="Diesel truck TCO per km (median)"
    )
    plt.fill_between(
        x,
        summary_df["diesel_p5"],
        summary_df["diesel_p95"],
        color="tab:blue",
        alpha=0.2
    )

    # BET-S AEaaS
    plt.plot(
        x,
        summary_df["bets_p50"],
        marker="o",
        color="tab:green",
        label="BET-S AEaaS cost per km (median)"
    )
    plt.fill_between(
        x,
        summary_df["bets_p5"],
        summary_df["bets_p95"],
        color="tab:green",
        alpha=0.2
    )

    plt.xlabel("Asset-manager margin (%)")
    plt.ylabel("Cost (£/km)")
    plt.title("Impact of Asset-manager Margin on Freight Cost per km with Uncertainty")
    plt.legend()
    plt.text(
        0.01, 0.98,
        UNCERTAINTY_NOTE,
        transform=plt.gca().transAxes,
        ha="left",
        va="top"
    )
    plt.tight_layout()
    plt.show()

# Plot margin vs cost gap with uncertainty bands
def plot_margin_vs_gap_with_uncertainty(summary_df):
    plt.figure(figsize=(10, 6))

    x = summary_df["asset_manager_margin"] * 100

    plt.plot(
        x,
        summary_df["gap_p50"],
        marker="o",
        label="BET-S AEaaS - Diesel (median)"
    )
    plt.fill_between(
        x,
        summary_df["gap_p5"],
        summary_df["gap_p95"],
        alpha=0.2
    )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Asset-manager margin (%)")
    plt.ylabel("Cost Gap (£/km)")
    plt.title("Effect of Asset-manager Margin on BET-S AEaaS - Diesel Gap with Uncertainty")
    plt.legend()
    plt.text(
        0.01, 0.98,
        UNCERTAINTY_NOTE,
        transform=plt.gca().transAxes,
        ha="left",
        va="top"
    )
    plt.tight_layout()
    plt.show()

def plot_projection_with_uncertainty(summary_df):
    plt.figure(figsize=(10, 6))

    specs = [
        ("diesel_tco_discounted", "Diesel", "tab:blue"),
        ("betc_tco_discounted", "BET-C", "tab:orange"),
        ("bets_tco_discounted", "BET-S", "tab:green"),
    ]

    for metric, label, color in specs:
        years = summary_df["year"]
        p5 = summary_df[f"{metric}_p5"]
        p50 = summary_df[f"{metric}_p50"]
        p95 = summary_df[f"{metric}_p95"]

        plt.plot(years, p50, marker="o", color=color, label=f"{label} median")
        plt.fill_between(years, p5, p95, color=color, alpha=0.2)

    plt.title("Projected Discounted TCO with Uncertainty")
    plt.xlabel("Purchase Year")
    plt.ylabel("Discounted TCO (£)")
    plt.xticks(summary_df["year"], rotation=45)
    plt.legend()
    plt.text(
        0.01, 0.98,
        UNCERTAINTY_NOTE,
        transform=plt.gca().transAxes,
        ha="left",
        va="top"
    )
    plt.tight_layout()
    plt.show()
    



def plot_monte_carlo_histograms(df):
    histogram_specs = [
        ("diesel_tco", "Monte Carlo: Diesel Truck Discounted TCO", "TCO (£)", "tab:blue"),
        ("bet_c_tco", "Monte Carlo: BET-C Discounted TCO", "TCO (£)", "tab:orange"),
        ("bet_s_tco", "Monte Carlo: BET-S Discounted TCO", "TCO (£)", "tab:green"),
        ("gap_bet_c_diesel", "Monte Carlo: BET-C - Diesel", "TCO Gap (£)", "tab:purple"),
        ("gap_bet_s_diesel", "Monte Carlo: BET-S - Diesel", "TCO Gap (£)", "tab:red"),
        ("gap_bet_s_bet_c", "Monte Carlo: BET-S - BET-C", "TCO Gap (£)", "tab:brown"),
    ]

    for col, title, xlabel, color in histogram_specs:
        plt.figure(figsize=(8, 5))
        plt.hist(df[col], bins=20, color=color, )
        plt.axvline(df[col].mean(), color = "black", linestyle="--", label="Mean")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()



# Bar chart of the ranked driver correlations.
def plot_drivers(driver_df, gap_name="BET-S - Diesel"):
    labels = [get_pretty_label(v) for v in driver_df["variable"]]
    values = driver_df["correlation_with_gap"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values)

    plt.axhline(0)
    plt.title(f"Drivers of {gap_name} Gap")
    plt.xlabel("Input variable")
    plt.ylabel("Correlation with the gap")
    plt.xticks(rotation=30, ha="right")

    for bar, v in zip(bars, values):
        va = "bottom" if v >= 0 else "top"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.2f}",
            ha="center",
            va=va
        )

    plt.tight_layout()
    plt.show()
    
##################### Boxplot Visualisation ####################################

def plot_independent_tco_boxplots(df, figsize=(24, 8)):
    """
    One figure:
    for each uncertain variable, show 3 boxplots:
    Diesel / BET-C / BET-S
    """
    variable_order = list(df["variable"].drop_duplicates())

    fig, ax = plt.subplots(figsize=figsize)

    positions = []
    data = []

    gap_between_groups = 2.0
    start = 1.0

    group_centers = []
    group_boundaries = []

    for g, var in enumerate(variable_order):
        base = start + g * (3 + gap_between_groups)

        diesel_data = df.loc[df["variable"] == var, "diesel_tco"].dropna()
        betc_data = df.loc[df["variable"] == var, "bet_c_tco"].dropna()
        bets_data = df.loc[df["variable"] == var, "bet_s_tco"].dropna()

        data.extend([diesel_data, betc_data, bets_data])
        positions.extend([base, base + 1, base + 2])

        # Name
        group_centers.append(base + 1)

        # Line
        if g < len(variable_order) - 1:
            next_base = start + (g + 1) * (3 + gap_between_groups)
            boundary = (base + 2 + next_base) / 2
            group_boundaries.append(boundary)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )

    
    color_map = {
        "diesel": "tab:blue",   
        "betc": "tab:orange",     
        "bets": "tab:green",     
    }

    colors = (
        [color_map["diesel"], color_map["betc"], color_map["bets"]]
        * len(variable_order)
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)
        ax.set_xticks(group_centers)
        pretty_labels = [get_pretty_label(v) for v in variable_order]
        ax.set_xticklabels(pretty_labels, rotation=35, ha="right")
        ax.set_ylabel("Discounted TCO (£)")
        ax.set_title("Independent Impact of each Uncertain Variable on TCO")

    for x in group_boundaries:
        ax.axvline(x=x, linestyle="--", linewidth=1)
    

    legend_patches = [
        mpatches.Patch(color=color_map["diesel"], label="Diesel"),
        mpatches.Patch(color=color_map["betc"], label="BET-C"),
        mpatches.Patch(color=color_map["bets"], label="BET-S"),
    ]

    ax.legend(handles=legend_patches, loc="upper right")
    
    plt.tight_layout()
    plt.show()


def plot_independent_gap_boxplots(df, figsize=(24, 8)):
    """
    One figure:
    for each uncertain variable, show 3 gap boxplots:
    BET-C - Diesel / BET-S - Diesel / BET-S - BET-C
    """
    variable_order = list(df["variable"].drop_duplicates())

    fig, ax = plt.subplots(figsize=figsize)

    positions = []
    data = []
    gap_between_groups = 2.0
    start = 1.0

    group_boundaries = []
    group_centers = []

    for g, var in enumerate(variable_order):
        base = start + g * (3 + gap_between_groups)

        gap1 = df.loc[df["variable"] == var, "gap_bet_c_diesel"].dropna()
        gap2 = df.loc[df["variable"] == var, "gap_bet_s_diesel"].dropna()
        gap3 = df.loc[df["variable"] == var, "gap_bet_s_bet_c"].dropna()

        data.extend([gap1, gap2, gap3])
        positions.extend([base, base + 1, base + 2])

        # 每组中心位置
        group_centers.append(base + 1)

        # 分隔线位置：本组最后一个箱线和下一组第一个箱线的中点
        if g < len(variable_order) - 1:
            next_base = start + (g + 1) * (3 + gap_between_groups)
            boundary = (base + 2 + next_base) / 2
            group_boundaries.append(boundary)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )


    color_map = {
        "betc_diesel": "tab:purple", 
        "bets_diesel": "tab:red", 
        "bets_betc": "tab:brown",
    }

    colors = (
        [color_map["betc_diesel"], color_map["bets_diesel"], color_map["bets_betc"]]
        * len(variable_order)
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_xticks(group_centers)
    pretty_labels = [get_pretty_label(v) for v in variable_order]
    ax.set_xticklabels(pretty_labels, rotation=35, ha="right")
    ax.set_ylabel("TCO Gap (£)")
    ax.set_title("Independent Impact of each Uncertain Variable on TCO Gaps")
    ax.axhline(0, linewidth=1)

    for x in group_boundaries:
        ax.axvline(x=x, linestyle="--", linewidth=1)

    legend_patches = [
        mpatches.Patch(color=color_map["betc_diesel"], label="BET-C - Diesel"),
        mpatches.Patch(color=color_map["bets_diesel"], label="BET-S - Diesel"),
        mpatches.Patch(color=color_map["bets_betc"], label="BET-S - BET-C"),
    ]
    ax.legend(handles=legend_patches, loc="upper right")

    ax.set_xlim(min(positions) - 1, max(positions) + 1)

    plt.tight_layout()
    plt.show()




def plot_independent_bets_vs_diesel_boxplot(df, figsize=(18, 7)):
    """
    One figure:
    for each uncertain variable, show only BET-S - Diesel gap boxplot
    """
    exclude_vars = [
        "bet_depot_energy_price_per_kwh",
        "bet_public_energy_price_per_kwh",
        "battery_capacity_kwh"
    ]

    variable_order = [
        v for v in df["variable"].drop_duplicates()
        if v not in exclude_vars
    ]

    fig, ax = plt.subplots(figsize=figsize)

    data = []
    positions = []

    for i, var in enumerate(variable_order, start=1):
        gap_data = df.loc[df["variable"] == var, "gap_bet_s_diesel"].dropna()
        data.append(gap_data)
        positions.append(i)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("tab:red")

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)
        
    ax.set_xticks(positions)
    pretty_labels = [get_pretty_label(v) for v in variable_order]
    ax.set_xticklabels(pretty_labels, rotation=35, ha="right")
    ax.set_ylabel("BET-S - Diesel TCO Gap (£)")
    ax.set_title("Independent Impact of each Uncertain Variable on BET-S - Diesel Gap")
    ax.axhline(0, linewidth=1)

    plt.tight_layout()
    plt.show()

###################### Summary Outputs ########################################################
# Make the numbers readable
def format_base_value(var, value):
    if "price" in var or "cost" in var:
        return f"£{value:.2f}"
    elif "rate" in var:
        return f"{value*100:.1f}%"
    elif "km" in var:
        return f"{value:.0f} km"
    else:
        return f"{value}"
    
# TCO results   
def pretty_summary() -> str:
    shared = SharedInputs()
    diesel = compute_diesel(shared, DieselInputs())
    bet_c = compute_bet_c(shared, BETCInputs(battery_recycle_ratio=shared.battery_recycle_value_ratio))
    bet_s = compute_bet_s(shared, BETSInputs(battery_recycle_ratio=shared.battery_recycle_value_ratio))

    lines = []
    lines.append("Discounted TCO summary")
    lines.append("-" * 40)
    lines.append(
        f"Diesel : £{diesel['tco_discounted']:,.2f} | "
        f"£{diesel['tco_per_km_discounted']:.4f}/km | "
        f"£{diesel['tco_per_kwh_discounted']:.4f}/kWh"
    )
    #lines.append(f"BET-C EO1L: £{bet_c['tco_discounted_eol']:,.2f} | £{bet_c['tco_per_km_discounted_eol']:.4f}/km")
    lines.append(
        f"BET-C  : £{bet_c['tco_discounted_recycle']:,.2f} | "
        f"£{bet_c['tco_per_km_discounted_recycle']:.4f}/km | "
        f"£{bet_c['tco_per_kwh_discounted_recycle']:.4f}/kWh"
    )
    #lines.append(f"BET-S EO1L: £{bet_s['tco_discounted_eol']:,.2f} | £{bet_s['tco_per_km_discounted_eol']:.4f}/km")
    lines.append(
        f"BET-S  : £{bet_s['tco_discounted_recycle']:,.2f} | "
        f"£{bet_s['tco_per_km_discounted_recycle']:.4f}/km | "
        f"£{bet_s['tco_per_kwh_discounted_recycle']:.4f}/kWh"
    )
    lines.append("")
    lines.append("TCO gaps vs Diesel (discounted, Duration)")
    #lines.append(f"BET-C EO1L - Diesel: £{bet_c['tco_discounted_eol'] - diesel['tco_discounted']:,.2f}")
    lines.append(f"BET-C. - Diesel: £{bet_c['tco_discounted_recycle'] - diesel['tco_discounted']:,.2f}")
    #lines.append(f"BET-S EO1L - Diesel: £{bet_s['tco_discounted_eol'] - diesel['tco_discounted']:,.2f}")
    lines.append(f"BET-S. - Diesel: £{bet_s['tco_discounted_recycle'] - diesel['tco_discounted']:,.2f}")
    lines.append(f"BET-S. - BET-C.: £{bet_s['tco_discounted_recycle'] - bet_c['tco_discounted_recycle']:,.2f}")
    lines.append("")
    return "\n".join(lines)

# Generate AEaaS pricing and cost breakdown summary
def pretty_aeaas_summary(results):
    lines = []
    lines.append("AEaaS cost summary for freight company")
    lines.append("(BET-S only; 10% asset-manager margin; driver cost added back)")
    lines.append("-" * 90)

    r = results["bet_s"]

    lines.append("BET-S")
    lines.append(
        f"  Asset & Energy sell price : £{r['price_per_km_with_margin']:.4f}/km | "
        f"£{r['price_per_kwh_with_margin']:.4f}/kWh"
    )
    lines.append(
        f"  Driver cost               : £{r['driver_cost_per_km']:.4f}/km | "
        f"£{r['driver_cost_per_kwh']:.4f}/kWh"
    )
    lines.append(
        f"  Freight all-in            : £{r['freight_total_cost_per_km']:.4f}/km | "
        f"£{r['freight_total_cost_per_kwh']:.4f}/kWh"
    )
    lines.append(
        f"  AEaaS total               : £{r['freight_total_cost_total']:,.2f}"
    )
    lines.append(
        f"  Own TCO                   : £{r['tco_discounted_recycle']:,.2f}"
    )
    lines.append(
        f"  Gap AEaaS - own TCO       : £{r['aas_gap_vs_own_tco']:,.2f}"
    )
    lines.append(
        f"  AEaaS discounted glider   : £{r['discounted_glider_cost_for_aeaas']:,.2f}"
    )
    lines.append(
        f"  AEaaS asset service cost  : £{r['aeaas_asset_service_cost_total']:,.2f}"
    )

    lines.append("")

    return "\n".join(lines)

# Format sensitivity results into readable text summary
def pretty_sensitivity_summary(sensitivity_results) -> str:
    lines = []
    target_class_text = "+".join(sensitivity_results["target_class"])
    lines.append(
        f"Sensitivity analysis: {target_class_text}.{sensitivity_results['variable_name']}"
    )
    lines.append(f"Base value: {sensitivity_results['base_value']}")
    lines.append("-" * 70)

    for i, label in enumerate(sensitivity_results["labels"]):
        lines.append(
            f"{label:>5} | "
            f"BET-C - Diesel: £{sensitivity_results['bet_c_vs_diesel'][i]:,.2f} | "
            f"BET-S - Diesel: £{sensitivity_results['bet_s_vs_diesel'][i]:,.2f} | "
            f"BET-S - BET-C: £{sensitivity_results['bet_s_vs_bet_c'][i]:,.2f}"
        )

    return "\n".join(lines)
    





######################################## The main processes of this code  ######################################################
