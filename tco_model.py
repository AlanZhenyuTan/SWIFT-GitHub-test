

from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from typing import Dict, List
import math
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

from pathlib import Path
import re


########################## Inputs ##########################################

@dataclass
class SharedInputs:
    # General TCO horizon and discounting
    years: int = 5
    discount_rate: float = 0.10

    # Operation profile
    full_loaded_km_per_day: float = 240.0
    unladen_ratio_to_full: float = 3 / 7
    operational_days_per_year: int = 292
    shift_per_day: float = 1.0

     # Personnel##################################################################################
    driver_hourly_pay: float = 15.78
    worked_hours_per_week: float = 48.0 # Personnel##################################################################################

    # Common financing assumptions from All-in-1-sheet rows 16-18
    cost_of_capital: float = 0.125
    upfront_payment_percentage: float = 0.30
    loan_term_years: int = 5
    aeaas_cost_of_capital: float = 0.10

    # Shared BET assumptions
    bet_insurance_markup: float = 0.20
    battery_recycle_value_ratio: float = 0.10
    bet_subsidy: float = 0.0

    # BET-C depot/public charging assumptions
    bet_depot_share: float = 0.80
    bet_depot_energy_price_per_kwh: float = 0.22
    bet_public_energy_price_per_kwh: float = 0.39
    battery_capacity_bet_c_kwh: float = 513.0

    # Diesel depot/public refuelling assumptions
    diesel_insurance: float = 10_000.0
    diesel_depot_share: float = 0.80
    diesel_depot_price_per_l: float = 1.05
    diesel_public_price_per_l: float = 1.48
    diesel_bunker_capex_per_l: float = 0.24
    diesel_expected_fleet_size: float = 51.0

    # BET-S / BaaS provider assumptions
    off_peak_share: float = 0.50
    peak_price_per_kwh: float = 0.20       # provider pays, peak
    off_peak_price_per_kwh: float = 0.10   # provider pays, off-peak
    electricity_margin: float = 1.00       # provider earns

    # AEaaS granular cost factors retained for optional margin/uncertainty plots.
    aeaas_glider_cost_factor: float = 0.90
    aeaas_insurance_cost_factor: float = 0.90
    aeaas_annual_service_cost_factor:float = 1.00
    aeaas_station_capex_factor: float = 1.00
    aeaas_station_opex_factor: float = 1.00
    aeaas_battery_depr_factor: float = 1.00
    aeaas_battery_service_factor: float = 1.00
    aeaas_battery_rent_factor: float = 0.90
    aeaas_fixed_swapping_fee_factor: float = 0.90
    aeaas_energy_cost_factor: float = 0.90


@dataclass
class DieselInputs:
    # Diesel truck specifications
    capex: float = 144_900.0
    fixed_depreciation_rate: float = 0.075
    variable_depreciation_per_km: float = 0.095
    truck_lifetime_years: float = 15.0

    # Operation and cost
    annual_service_cost: float = 4_800.0
    fuel_economy_full_loaded_year1_l_per_km: float = 0.35
    fuel_economy_growth_rate: float = 0.011
    unladen_energy_saving: float = 0.25
    refuels_per_day: float = 1.0
    lez_operation_percentage: float = 0.10
    lez_charge: float = 50.0
    litre_to_kwh: float = 3.0


@dataclass
class BETCInputs:
    # BET-C truck specifications
    glider_capex: float = 130_000.0
    battery_capacity_kwh: float = 513.0
    battery_price_per_kwh: float = 148.0
    glider_fixed_depreciation_rate: float = 0.075
    glider_variable_depreciation_per_km: float = 0.095
    glider_residual_value_percentage: float = 0.20
    glider_lifetime_years: float = 15.0

    # Battery and operation
    battery_recycle_value_ratio: float = 0.10
    battery_lifetime_cycles: float = 2200.0
    annual_service_cost: float = 4_200.0
    fuel_economy_growth_rate: float = 0.014
    full_loaded_kwh_per_km_year1: float = 1.37
    unladen_energy_saving: float = 0.25
    recharges_per_day: float | None = None  # None = calculate from daily distance need and single-charge range

    # Depot charging infrastructure
    charger_capex_per_kwh: float = 0.12
    charger_lifetime_years: float = 8.0
    site_capex_per_kwh: float = 0.08
    site_lifetime_years: float = 24.0
    expected_fleet_size: float = 51.0


@dataclass
class BETSInputs:
    # BET-S truck and battery specifications
    glider_capex: float = 130_000.0
    battery_pack_capacity_kwh: float = 171.0
    battery_packs_per_truck: float = 3.0
    battery_price_per_kwh: float = 148.0
    battery_recycle_value_ratio: float = 0.10
    battery_lifetime_cycles: float = 2200.0
    glider_fixed_depreciation_rate: float = 0.075
    glider_variable_depreciation_per_km: float = 0.095
    glider_residual_value_percentage: float = 0.20
    glider_lifetime_years: float = 15.0

    # Truck operation
    annual_service_cost: float = 4_200.0
    full_loaded_kwh_per_km_year1: float = 1.37
    fuel_economy_growth_rate: float = 0.005
    unladen_energy_saving: float = 0.25
    swaps_per_day: float | None = None  # None = calculate from daily distance need and swappable-battery range

    # Station / BaaS provider structure
    station_battery_bays: float = 24.0
    station_capex: float = 2_000_000.0
    site_capex: float = 1_000_000.0
    station_lifetime_years: float = 15.0
    max_station_service_capacity_trucks_per_day: float = 171.0
    expected_station_utilisation: float = 0.30
    station_annual_staff_costs: float = 216_000.0
    station_annual_other_service_costs: float = 48_705.0
    expected_annual_return_on_battery_renting: float = 0.15
    swapping_fee_flat: float = 3.0



label_map = {
    "expected_station_utilisation": "Expected Station Utilisation",
    "discount_rate": "Discount Rate",
    "cost_of_capital": "Cost of Capital",
    "upfront_payment_percentage": "Upfront Payment Percentage",
    "loan_term_years": "Loan Term",
    "full_loaded_km_per_day": "Full-loaded VKT per Day",
    "peak_price_per_kwh": "Peak Retail Energy Price - BaaS Provider Pays",
    "off_peak_price_per_kwh": "Off-peak Retail Energy Price - BaaS Provider Pays",
    "electricity_margin": "Target Electricity Margin",
    "off_peak_share": "Off-peak Swapping Percentage",
    "full_loaded_kwh_per_km_year1": "BET Full-loaded kWh per km in Year 1",
    "battery_recycle_value_ratio": "Battery Residual Percentage",
    "bet_subsidy": "BET Purchase Subsidy",
    "bet_public_energy_price_per_kwh": "Public Hub Electricity Price",
    "bet_depot_energy_price_per_kwh": "On-depot Retail Energy Price",
    "glider_capex": "Electric Glider Price",
    "battery_price_per_kwh": "Initial Battery Price",
    "battery_capacity_kwh": "BET-C Total Battery Capacity",
    "station_capex": "Station CAPEX",
    "site_capex": "Site CAPEX",
    "station_annual_staff_costs": "Station Annual Staff Costs",
    "station_annual_other_service_costs": "Other Station Annual Service Costs",
    "years": "TCO Horizon",
    "battery_lifetime_cycles": "Battery Lifetime Cycles",

}


def get_pretty_label(var):
    if var in label_map:
        return label_map[var]
    label = var.replace("_", " ").title()
    label = label.replace("Kwh", "kWh")
    label = label.replace("Km", "km")
    label = label.replace("Vkt", "VKT")
    return label


########### Input Calculations ############################################################################################
def discount_factors(rate: float, years: int) -> List[float]:
    return [1 / (1 + rate) ** y for y in range(1, years + 1)]


def annual_driver_salary(days_per_year: float, hours_per_week: float, hourly_pay: float, shift_per_day: float = 1.0) -> float:
    """Annual driver/personnel cost per truck.

    Excel first calculates one-shift driver salary as:
        operational days / (worked hours per week / 9) * worked hours per week * hourly pay
    and then multiplies it by Shift per day in the fixed operating cost rows
    (All-in-1-sheet rows C39/G39/K37).
    """
    one_shift_salary = days_per_year / (hours_per_week / 9) * hours_per_week * hourly_pay
    return one_shift_salary * shift_per_day


def bet_insurance(diesel_insurance: float, markup: float) -> float:
    return diesel_insurance * (1 + markup)


def daily_distances(shared: SharedInputs) -> tuple[float, float]:
    full_loaded = shared.full_loaded_km_per_day
    unladen = full_loaded * shared.unladen_ratio_to_full
    return full_loaded, unladen


# Backwards-compatible alias used elsewhere in the old code. ?????????????????????????????????????????
def diesel_daily_distances(shared: SharedInputs) -> tuple[float, float]:
    return daily_distances(shared)
#####################################################????????????????????????????????????????????????


def annual_km(shared: SharedInputs) -> float:
    full_km, unladen_km = daily_distances(shared)
    return (full_km + unladen_km) * shared.operational_days_per_year * shared.shift_per_day

def pmt(rate: float, nper: int, pv: float, fv: float = 0.0, when: int = 0) -> float:
    """Excel PMT equivalent. Returns Excel-sign convention value."""
    if rate == 0:
        return -(pv + fv) / nper
    factor = (1 + rate) ** nper
    return -(rate * (pv * factor + fv)) / ((1 + rate * when) * (factor - 1))


def annual_loan_payment(rate: float, nper: int, pv: float) -> float:
    """Positive amortised annual loan payment, equivalent to Excel: -PMT(rate, nper, pv)."""
    return -pmt(rate, nper, pv)


def financed_acquisition_npv(capex: float, shared: SharedInputs) -> float:
    """Upfront payment + discounted loan repayments within loan term only."""
    years_for_loan_payment = min(shared.years, shared.loan_term_years)

    df = discount_factors(shared.discount_rate, shared.years)
    loan_df = df[:years_for_loan_payment]

    upfront_payment = capex * shared.upfront_payment_percentage
    loan_amount = capex * (1 - shared.upfront_payment_percentage)

    annual_payment = annual_loan_payment(
        shared.cost_of_capital,
        shared.loan_term_years,
        loan_amount,
    )

    return upfront_payment + annual_payment * sum(loan_df)

def financed_acquisition_npv_with_rate(
    capex: float,
    shared: SharedInputs,
    cost_of_capital: float,
) -> float:
    years_for_loan_payment = min(shared.years, shared.loan_term_years)

    df = discount_factors(shared.discount_rate, shared.years)
    loan_df = df[:years_for_loan_payment]

    upfront_payment = capex * shared.upfront_payment_percentage
    loan_amount = capex * (1 - shared.upfront_payment_percentage)

    annual_payment = annual_loan_payment(
        cost_of_capital,
        shared.loan_term_years,
        loan_amount,
    )

    return upfront_payment + annual_payment * sum(loan_df)


def yearly_growth_series(start_value: float, growth_rate: float, years: int) -> List[float]:   #????????????????????????????????????
    vals = [start_value]
    for _ in range(1, years):
        vals.append(vals[-1] * (1 + growth_rate))
    return vals


def diesel_yearly_fuel_economies(inp: DieselInputs, years: int) -> List[float]:
    return yearly_growth_series(inp.fuel_economy_full_loaded_year1_l_per_km, inp.fuel_economy_growth_rate, years)


def betc_yearly_full_loaded_economies(inp: BETCInputs, years: int) -> List[float]:
    return yearly_growth_series(inp.full_loaded_kwh_per_km_year1, inp.fuel_economy_growth_rate, years)


def bets_yearly_full_loaded_economies(inp: BETSInputs, years: int) -> List[float]:
    return yearly_growth_series(inp.full_loaded_kwh_per_km_year1, inp.fuel_economy_growth_rate, years)


def daily_energy_from_full_unladen(shared: SharedInputs, full_loaded_series: List[float], unladen_saving: float) -> List[float]:
    full_km, unladen_km = daily_distances(shared)
    unladen_series = [x * (1 - unladen_saving) for x in full_loaded_series]
    return [
        (full_km * f + unladen_km * u) * shared.shift_per_day
        for f, u in zip(full_loaded_series, unladen_series)
    ]


def calculated_recharges_per_day(shared: SharedInputs, inp: BETCInputs) -> int:
    full_km, unladen_km = daily_distances(shared)

    return max(
        1,
        math.ceil(
            ((full_km + unladen_km) * shared.shift_per_day)
            /
            (
                inp.battery_capacity_kwh
                / inp.full_loaded_kwh_per_km_year1
            )
        )
    )


def get_recharges_per_day(shared: SharedInputs, inp: BETCInputs) -> float:
    return calculated_recharges_per_day(shared, inp) if inp.recharges_per_day is None else inp.recharges_per_day


################### TCO calculations ######################################################################
def compute_diesel(shared: SharedInputs, inp: DieselInputs) -> Dict[str, float]:
    """Diesel TCO translated from All-in-1-sheet formulas."""
    years = shared.years
    df = discount_factors(shared.discount_rate, years)
    df_sum = sum(df)
    daily_litres = daily_energy_from_full_unladen(
        shared,
        diesel_yearly_fuel_economies(inp, years),
        inp.unladen_energy_saving,
    )
    akm = annual_km(shared)
    annual_salary = annual_driver_salary(
        shared.operational_days_per_year,
        shared.worked_hours_per_week,
        shared.driver_hourly_pay,
        shared.shift_per_day,
    )

    truck_acquisition_cost_npv = financed_acquisition_npv(inp.capex, shared)
    truck_residual_value = max(
        0.0,
        inp.capex * (1 - inp.fixed_depreciation_rate) ** years
        - inp.variable_depreciation_per_km * akm * years,
    )
    truck_residual_value_npv = truck_residual_value * df[-1]
    fixed_operating_cost_npv = (
        inp.annual_service_cost
        + annual_salary
        + shared.diesel_insurance
        + shared.operational_days_per_year * inp.lez_operation_percentage * inp.lez_charge
    ) * df_sum

    annual_fleet_diesel = [
        daily * shared.diesel_expected_fleet_size * shared.operational_days_per_year
        for daily in daily_litres
    ]
    annual_on_depot = [shared.diesel_depot_share * x for x in annual_fleet_diesel]

    depot_infra_per_truck_npv = sum(
        demand * shared.diesel_bunker_capex_per_l / shared.diesel_expected_fleet_size * w
        for demand, w in zip(annual_on_depot, df)
    )
    energy_cost_npv = sum(
        (
            shared.diesel_depot_price_per_l * daily * shared.diesel_depot_share * shared.operational_days_per_year
            + shared.diesel_public_price_per_l * daily * (1 - shared.diesel_depot_share) * shared.operational_days_per_year
        ) * w
        for daily, w in zip(daily_litres, df)
    )

    tco_discounted = (
        truck_acquisition_cost_npv
        - truck_residual_value_npv
        + depot_infra_per_truck_npv
        + fixed_operating_cost_npv
        + energy_cost_npv
    )
    total_litres = sum(daily_litres) * shared.operational_days_per_year
    total_energy_kwh = total_litres * inp.litre_to_kwh

    return {
        "tco_undiscounted": tco_discounted,
        "tco_discounted": tco_discounted,
        "tco_per_year_discounted": tco_discounted / years,
        "tco_per_km_discounted": tco_discounted / (akm * years),
        "tco_per_kwh_discounted": tco_discounted / total_energy_kwh if total_energy_kwh else math.nan,
        "annual_km": akm,
        "daily_energy_year1_l": daily_litres[0],
        "daily_litres_year1": daily_litres[0],
        "daily_litres_by_year": daily_litres,
        "truck_acquisition_cost_npv": truck_acquisition_cost_npv,
        "truck_residual": truck_residual_value,
        "truck_residual_value_npv": truck_residual_value_npv,
        "fixed_operating_cost_npv": fixed_operating_cost_npv,
        "truck_fixed_operating_cost_npv": fixed_operating_cost_npv,
        "depot_infra_per_truck_npv": depot_infra_per_truck_npv,
        "depot_infrastructure_cost_npv": depot_infra_per_truck_npv,
        "energy_cost_npv": energy_cost_npv,
        "total_energy_service_cost_npv": energy_cost_npv,
        "annual_driver_cost": annual_salary,
    }


def compute_bet_c(shared: SharedInputs, inp: BETCInputs, asset_manager_margin: float = 0.10) -> Dict[str, float]:
    """BET-C TCO translated from All-in-1-sheet formulas."""
    years = shared.years
    df = discount_factors(shared.discount_rate, years)
    df_sum = sum(df)
    daily_kwh = daily_energy_from_full_unladen(
        shared,
        betc_yearly_full_loaded_economies(inp, years),
        inp.unladen_energy_saving,
    )
    akm = annual_km(shared)
    annual_salary = annual_driver_salary(
        shared.operational_days_per_year,
        shared.worked_hours_per_week,
        shared.driver_hourly_pay,
        shared.shift_per_day,
    )
    insurance = bet_insurance(shared.diesel_insurance, shared.bet_insurance_markup)

    truck_capex = inp.glider_capex + inp.battery_capacity_kwh * inp.battery_price_per_kwh
    truck_acquisition_cost_npv = financed_acquisition_npv(truck_capex, shared)
    glider_residual_value = max(
        0.0,
        inp.glider_capex * (1 - inp.glider_fixed_depreciation_rate) ** years
        - inp.glider_variable_depreciation_per_km * akm * years,
    )
    glider_residual_value_npv = glider_residual_value * df[-1]
    battery_value = inp.battery_capacity_kwh * inp.battery_price_per_kwh
    battery_residual_value_npv = max(
        battery_value * inp.battery_recycle_value_ratio,
        (
            battery_value
            - battery_value
            * (1 - inp.battery_recycle_value_ratio)
            * (shared.operational_days_per_year * years * shared.shift_per_day / inp.battery_lifetime_cycles)
        )
        * df[-1],
    )
    truck_residual_value_npv = glider_residual_value_npv + battery_residual_value_npv

    fixed_operating_cost_npv = (inp.annual_service_cost + annual_salary + insurance) * df_sum
    recharges_per_day = get_recharges_per_day(shared, inp)

    annual_energy_requirement = [
        daily * inp.expected_fleet_size * shared.operational_days_per_year
        for daily in daily_kwh
    ]
    annual_on_depot = [shared.bet_depot_share * x for x in annual_energy_requirement]
    depot_infra_per_truck_npv = sum(
        demand * (inp.charger_capex_per_kwh + inp.site_capex_per_kwh) / inp.expected_fleet_size * w
        for demand, w in zip(annual_on_depot, df)
    )
    energy_cost_npv = sum(
        (
            shared.bet_depot_energy_price_per_kwh * daily * shared.bet_depot_share * shared.operational_days_per_year
            + shared.bet_public_energy_price_per_kwh * daily * (1 - shared.bet_depot_share) * shared.operational_days_per_year
        ) * w
        for daily, w in zip(daily_kwh, df)
    )
    tco_discounted = (
        truck_acquisition_cost_npv
        - truck_residual_value_npv
        + depot_infra_per_truck_npv
        + fixed_operating_cost_npv
        + energy_cost_npv
        - shared.bet_subsidy
    )

    total_kwh = sum(daily_kwh) * shared.operational_days_per_year
    return {
        "tco_discounted": tco_discounted,
        "tco_per_year_discounted": tco_discounted / years,
        "tco_per_km_discounted": tco_discounted / (akm * years),
        "tco_per_kwh_discounted": tco_discounted / total_kwh if total_kwh else math.nan,
        "tco_discounted_eol": tco_discounted,
        "tco_discounted_recycle": tco_discounted,
        "tco_per_year_discounted_eol": tco_discounted / years,
        "tco_per_km_discounted_eol": tco_discounted / (akm * years),
        "tco_per_kwh_discounted_eol": tco_discounted / total_kwh if total_kwh else math.nan,
        "tco_per_year_discounted_recycle": tco_discounted / years,
        "tco_per_km_discounted_recycle": tco_discounted / (akm * years),
        "tco_per_kwh_discounted_recycle": tco_discounted / total_kwh if total_kwh else math.nan,
        "annual_km": akm,
        "daily_energy_year1_kwh": daily_kwh[0],
        "daily_kwh_year1": daily_kwh[0],
        "daily_kwh_by_year": daily_kwh,
        "recharges_per_day": recharges_per_day,
        "charges_per_day": recharges_per_day,
        "annual_driver_cost": annual_salary,
        "truck_capex": truck_capex,
        "truck_acquisition_cost_npv": truck_acquisition_cost_npv,
        "glider_residual_value_npv": glider_residual_value_npv,
        "battery_residual_value_npv": battery_residual_value_npv,
        "truck_residual_value_npv": truck_residual_value_npv,
        "fixed_operating_cost_npv": fixed_operating_cost_npv,
        "truck_fixed_operating_cost_npv": fixed_operating_cost_npv,
        "depot_infra_per_truck_npv": depot_infra_per_truck_npv,
        "depot_infrastructure_cost_npv": depot_infra_per_truck_npv,
        "energy_cost_npv": energy_cost_npv,
        "energy_service_total_cost_npv": energy_cost_npv,
    }


def compute_bet_s(shared: SharedInputs, inp: BETSInputs, asset_manager_margin: float = 0.10) -> Dict[str, float]:
    """BET-S TCO translated from All-in-1-sheet formulas, with old AEaaS outputs preserved."""
    years = shared.years
    df = discount_factors(shared.discount_rate, years)
    df_sum = sum(df)
    daily_kwh = daily_energy_from_full_unladen(
        shared,
        bets_yearly_full_loaded_economies(inp, years),
        inp.unladen_energy_saving,
    )
    akm = annual_km(shared)
    annual_salary = annual_driver_salary(
        shared.operational_days_per_year,
        shared.worked_hours_per_week,
        shared.driver_hourly_pay,
        shared.shift_per_day,
    )
    insurance = bet_insurance(shared.diesel_insurance, shared.bet_insurance_markup)

    full_km, unladen_km = daily_distances(shared)
    calculated_swaps_per_day_value = max(
        1,
        math.ceil(
            ((full_km + unladen_km) * shared.shift_per_day)
            / (inp.battery_pack_capacity_kwh * inp.battery_packs_per_truck / inp.full_loaded_kwh_per_km_year1)
        ),
    )
    swaps_per_day = calculated_swaps_per_day_value if inp.swaps_per_day is None else inp.swaps_per_day

    glider_acquisition_cost_npv = financed_acquisition_npv(inp.glider_capex, shared)
    glider_residual_value = max(
        0.0,
        inp.glider_capex * (1 - inp.glider_fixed_depreciation_rate) ** years
        - inp.glider_variable_depreciation_per_km * akm * years,
    )
    glider_residual_value_npv = glider_residual_value * df[-1]
    fixed_operating_cost_npv = (inp.annual_service_cost + annual_salary + insurance) * df_sum

    expected_station_service_demand = round(
        inp.max_station_service_capacity_trucks_per_day * inp.expected_station_utilisation
    )
    battery_capex = inp.battery_pack_capacity_kwh * inp.battery_price_per_kwh * (
        inp.station_battery_bays + expected_station_service_demand * inp.battery_packs_per_truck
    )
    annual_allocated_station_operating_cost_per_truck = (
        inp.station_annual_staff_costs + inp.station_annual_other_service_costs
    ) / expected_station_service_demand
    annual_allocated_infrastructure_depreciation_per_truck = (
        (inp.station_capex + inp.site_capex) / inp.station_lifetime_years
    ) / expected_station_service_demand
    annual_allocated_battery_depreciation_per_truck = (
        battery_capex
        * (1 - inp.battery_recycle_value_ratio)
        * (shared.operational_days_per_year / inp.battery_lifetime_cycles)
    ) / expected_station_service_demand
    basic_annual_rent_to_cover_baas_costs = (
        annual_allocated_station_operating_cost_per_truck
        + annual_allocated_infrastructure_depreciation_per_truck
        + annual_allocated_battery_depreciation_per_truck
    )
    annual_extra_rent_as_profit = (
        inp.battery_price_per_kwh
        * inp.battery_packs_per_truck
        * inp.battery_pack_capacity_kwh
        * inp.expected_annual_return_on_battery_renting
    )
    annual_rent_fleet_manager_pays = basic_annual_rent_to_cover_baas_costs + annual_extra_rent_as_profit
    rent_fleet_manager_pays_npv = annual_rent_fleet_manager_pays * df_sum
    fixed_swapping_fees_npv = inp.swapping_fee_flat * swaps_per_day * shared.operational_days_per_year * df_sum

    fleet_peak_price = shared.peak_price_per_kwh * (1 + shared.electricity_margin)
    fleet_off_peak_price = shared.off_peak_price_per_kwh * (1 + shared.electricity_margin)
    provider_base_energy_price = shared.peak_price_per_kwh * (1 - shared.off_peak_share) + shared.off_peak_price_per_kwh * shared.off_peak_share
    fleet_energy_price = fleet_peak_price * (1 - shared.off_peak_share) + fleet_off_peak_price * shared.off_peak_share
    electricity_service_costs = [daily * shared.operational_days_per_year * fleet_energy_price for daily in daily_kwh]
    electricity_service_costs_npv = sum(cost * w for cost, w in zip(electricity_service_costs, df))
    energy_service_total_cost_npv = rent_fleet_manager_pays_npv + fixed_swapping_fees_npv + electricity_service_costs_npv

    tco_discounted = (
        glider_acquisition_cost_npv
        - glider_residual_value_npv
        + fixed_operating_cost_npv
        + energy_service_total_cost_npv
        - shared.bet_subsidy
    )

    # AEaaS cost base:
    # AEaaS is assumed to follow the same BET-S cost structure,
    # but each cost element is reduced by AEaaS scale-economy factors.
    # Driver cost is excluded from the AEaaS asset-service cost base
    # and added back separately for the freight-company all-in cost.

    discounted_driver_cost_total = annual_salary * df_sum

    # 1. Glider cost with AEaaS lower cost of capital
    aeaas_glider_acquisition_cost_npv = financed_acquisition_npv_with_rate(
        inp.glider_capex,
        shared,
        shared.aeaas_cost_of_capital,
    )

    discounted_glider_cost_for_aeaas = (
        aeaas_glider_acquisition_cost_npv
        - glider_residual_value_npv
    ) * shared.aeaas_glider_cost_factor

    # 2. Annual service cost
    discounted_service_cost_for_aeaas = (
        inp.annual_service_cost
        * df_sum
        * shared.aeaas_annual_service_cost_factor
    )

    # 3. Insurance
    discounted_insurance_for_aeaas = (
        insurance
        * df_sum
        * shared.aeaas_insurance_cost_factor
    )

    # 4. Station operating cost
    discounted_station_operating_cost_for_aeaas = (
        annual_allocated_station_operating_cost_per_truck
        * df_sum
        * shared.aeaas_station_opex_factor
    )

    # 5. Infrastructure depreciation
    discounted_infrastructure_depr_for_aeaas = (
        annual_allocated_infrastructure_depreciation_per_truck
        * df_sum
        * shared.aeaas_station_capex_factor
    )

    # 6. Battery depreciation
    discounted_battery_depr_for_aeaas = (
        annual_allocated_battery_depreciation_per_truck
        * df_sum
        * shared.aeaas_battery_depr_factor
    )

    # 7. Battery rent profit / return component
    discounted_battery_rent_for_aeaas = (
        annual_extra_rent_as_profit
        * df_sum
        * shared.aeaas_battery_rent_factor
    )

    # 8. Fixed swapping fees
    discounted_fixed_swapping_for_aeaas = (
        fixed_swapping_fees_npv
        * shared.aeaas_fixed_swapping_fee_factor
    )

    # 9. Electricity service cost based on fleet_energy_price, not provider_base_energy_price
    discounted_energy_for_aeaas = (
        electricity_service_costs_npv
        * shared.aeaas_energy_cost_factor
    )

    aeaas_asset_service_cost_total_before_subsidy = (
        discounted_glider_cost_for_aeaas
        + discounted_service_cost_for_aeaas
        + discounted_insurance_for_aeaas
        + discounted_station_operating_cost_for_aeaas
        + discounted_infrastructure_depr_for_aeaas
        + discounted_battery_depr_for_aeaas
        + discounted_battery_rent_for_aeaas
        + discounted_fixed_swapping_for_aeaas
        + discounted_energy_for_aeaas
    )

    aeaas_subsidy_for_provider = shared.bet_subsidy

    aeaas_asset_service_cost_total = (
        aeaas_asset_service_cost_total_before_subsidy
        - aeaas_subsidy_for_provider
    )

    
    asset_service = compute_asset_service_unit_prices(
        asset_service_cost_total=aeaas_asset_service_cost_total,
        annual_driver_cost=annual_salary,
        annual_km=akm,
        daily_energy_list=daily_kwh,
        shared=shared,
        margin=asset_manager_margin,
    )
    aas_gap_vs_own_tco = asset_service["freight_total_cost_total"] - tco_discounted
    total_kwh = sum(daily_kwh) * shared.operational_days_per_year

    return {
        "tco_discounted": tco_discounted,
        "tco_per_year_discounted": tco_discounted / years,
        "tco_per_km_discounted": tco_discounted / (akm * years),
        "tco_per_kwh_discounted": tco_discounted / total_kwh if total_kwh else math.nan,
        "tco_discounted_eol": tco_discounted,
        "tco_discounted_recycle": tco_discounted,
        "tco_per_year_discounted_eol": tco_discounted / years,
        "tco_per_km_discounted_eol": tco_discounted / (akm * years),
        "tco_per_kwh_discounted_eol": tco_discounted / total_kwh if total_kwh else math.nan,
        "tco_per_year_discounted_recycle": tco_discounted / years,
        "tco_per_km_discounted_recycle": tco_discounted / (akm * years),
        "tco_per_kwh_discounted_recycle": tco_discounted / total_kwh if total_kwh else math.nan,
        "annual_km": akm,
        "daily_energy_year1_kwh": daily_kwh[0],
        "daily_kwh_year1": daily_kwh[0],
        "daily_kwh_by_year": daily_kwh,
        "annual_driver_cost": annual_salary,
        "calculated_swaps_per_day": calculated_swaps_per_day_value,
        "swaps_per_day": swaps_per_day,
        "glider_acquisition_cost_npv": glider_acquisition_cost_npv,
        "glider_total_cost_npv": glider_acquisition_cost_npv,
        "glider_residual_value_npv": glider_residual_value_npv,
        "fixed_operating_cost_npv": fixed_operating_cost_npv,
        "truck_fixed_operating_cost_npv": fixed_operating_cost_npv,
        "battery_capex": battery_capex,
        "expected_station_service_demand": expected_station_service_demand,
        "annual_allocated_station_operating_cost_per_truck": annual_allocated_station_operating_cost_per_truck,
        "annual_allocated_infrastructure_depreciation_per_truck": annual_allocated_infrastructure_depreciation_per_truck,
        "annual_allocated_battery_depreciation_per_truck": annual_allocated_battery_depreciation_per_truck,
        "basic_annual_rent_to_cover_baas_costs": basic_annual_rent_to_cover_baas_costs,
        "basic_monthly_rent_to_cover_costs": basic_annual_rent_to_cover_baas_costs / 12,
        "annual_extra_rent_as_profit": annual_extra_rent_as_profit,
        "extra_monthly_rent": annual_extra_rent_as_profit / 12,
        "annual_rent_fleet_manager_pays": annual_rent_fleet_manager_pays,
        "rent_fleet_manager_pays_npv": rent_fleet_manager_pays_npv,
        "fixed_swapping_fees_npv": fixed_swapping_fees_npv,
        "electricity_service_costs_npv": electricity_service_costs_npv,
        "energy_service_total_cost_npv": energy_service_total_cost_npv,
        "provider_base_energy_price": provider_base_energy_price,
        "fleet_energy_price": fleet_energy_price,
        **asset_service,
        "discounted_glider_cost_for_aeaas": discounted_glider_cost_for_aeaas,
        "discounted_insurance_for_aeaas": discounted_insurance_for_aeaas,
        
        "aeaas_glider_acquisition_cost_npv": aeaas_glider_acquisition_cost_npv,
        "discounted_service_cost_for_aeaas": discounted_service_cost_for_aeaas,
        "discounted_station_operating_cost_for_aeaas": discounted_station_operating_cost_for_aeaas,
        "discounted_infrastructure_depr_for_aeaas": discounted_infrastructure_depr_for_aeaas,

        "discounted_battery_rent_for_aeaas": discounted_battery_rent_for_aeaas,
        "discounted_fixed_swapping_for_aeaas": discounted_fixed_swapping_for_aeaas,
        "discounted_battery_depr_for_aeaas": discounted_battery_depr_for_aeaas,
        "discounted_energy_for_aeaas": discounted_energy_for_aeaas,
        "aeaas_asset_service_cost_total_before_subsidy": aeaas_asset_service_cost_total_before_subsidy,
        "aeaas_subsidy_for_provider": aeaas_subsidy_for_provider,
        "aeaas_asset_service_cost_total": aeaas_asset_service_cost_total,
        "aas_gap_vs_own_tco": aas_gap_vs_own_tco,
    }

# =========================================================
# BaaS provider IRR and payback analysis
# =========================================================

def npv_from_cashflows(rate: float, cashflows: list[float]) -> float:
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))


def compute_irr_bisection(cashflows: list[float], low=-0.99, high=5.0, tol=1e-6, max_iter=200):
    """
    Return IRR as decimal, e.g. 0.15 = 15%.
    Returns NaN if IRR cannot be solved.
    """
    npv_low = npv_from_cashflows(low, cashflows)
    npv_high = npv_from_cashflows(high, cashflows)

    if npv_low * npv_high > 0:
        return math.nan

    for _ in range(max_iter):
        mid = (low + high) / 2
        npv_mid = npv_from_cashflows(mid, cashflows)

        if abs(npv_mid) < tol:
            return mid

        if npv_low * npv_mid <= 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid

    return (low + high) / 2


def compute_payback_period(cashflows: list[float]):
    """
    Simple payback period, not discounted.
    Return years. If never paid back, return NaN.
    """
    cumulative = cashflows[0]

    if cumulative >= 0:
        return 0.0

    for year in range(1, len(cashflows)):
        previous = cumulative
        cumulative += cashflows[year]

        if cumulative >= 0:
            needed = -previous
            annual_cf = cashflows[year]
            return (year - 1) + needed / annual_cf

    return math.nan


def compute_baas_provider_cashflows(
    shared: SharedInputs,
    inp: BETSInputs,
) -> dict:
    """
    Station/BaaS provider cashflow model.

    Year 0:
    - station CAPEX
    - site CAPEX
    - station batteries + truck batteries provided by BaaS provider

    Annual cash inflow:
    - battery renting income
    - fixed swapping fee income
    - electricity sales income

    Annual cash outflow:
    - station staff and service costs
    - electricity procurement cost

    Note:
    - depreciation is not treated as a cash outflow.
    - terminal/residual value is not included here.
    """
    years = shared.years

    daily_kwh = daily_energy_from_full_unladen(
        shared,
        bets_yearly_full_loaded_economies(inp, years),
        inp.unladen_energy_saving,
    )

    full_km, unladen_km = daily_distances(shared)

    calculated_swaps_per_day_value = max(
        1,
        math.ceil(
            ((full_km + unladen_km) * shared.shift_per_day)
            /
            (
                inp.battery_pack_capacity_kwh
                * inp.battery_packs_per_truck
                / inp.full_loaded_kwh_per_km_year1
            )
        )
    )

    swaps_per_day = (
        calculated_swaps_per_day_value
        if inp.swaps_per_day is None
        else inp.swaps_per_day
    )

    expected_station_service_demand = round(
        inp.max_station_service_capacity_trucks_per_day
        * inp.expected_station_utilisation
    )

    station_battery_capex = (
        inp.station_battery_bays
        * inp.battery_pack_capacity_kwh
        * inp.battery_price_per_kwh
    )

    truck_battery_capex = (
        expected_station_service_demand
        * inp.battery_packs_per_truck
        * inp.battery_pack_capacity_kwh
        * inp.battery_price_per_kwh
    )

    battery_capex = station_battery_capex + truck_battery_capex

    initial_investment = -(
        inp.station_capex
        + inp.site_capex
        + battery_capex
    )

    annual_allocated_station_operating_cost_per_truck = (
        inp.station_annual_staff_costs
        + inp.station_annual_other_service_costs
    ) / expected_station_service_demand

    annual_allocated_infrastructure_depreciation_per_truck = (
        (inp.station_capex + inp.site_capex)
        / inp.station_lifetime_years
    ) / expected_station_service_demand

    annual_allocated_battery_depreciation_per_truck = (
        battery_capex
        * (1 - inp.battery_recycle_value_ratio)
        * (shared.operational_days_per_year / inp.battery_lifetime_cycles)
    ) / expected_station_service_demand

    basic_annual_rent_to_cover_baas_costs = (
        annual_allocated_station_operating_cost_per_truck
        + annual_allocated_infrastructure_depreciation_per_truck
        + annual_allocated_battery_depreciation_per_truck
    )

    annual_extra_rent_as_profit = (
        inp.battery_price_per_kwh
        * inp.battery_packs_per_truck
        * inp.battery_pack_capacity_kwh
        * inp.expected_annual_return_on_battery_renting
    )

    annual_rent_per_truck = (
        basic_annual_rent_to_cover_baas_costs
        + annual_extra_rent_as_profit
    )

    provider_base_energy_price = (
        shared.peak_price_per_kwh * (1 - shared.off_peak_share)
        + shared.off_peak_price_per_kwh * shared.off_peak_share
    )

    fleet_energy_price = provider_base_energy_price * (1 + shared.electricity_margin)

    annual_cashflows = []

    for daily_energy in daily_kwh:
        annual_rent_income = annual_rent_per_truck * expected_station_service_demand

        annual_swapping_fee_income = (
            inp.swapping_fee_flat
            * swaps_per_day
            * shared.operational_days_per_year
            * expected_station_service_demand
        )

        annual_electricity_sales_income = (
            daily_energy
            * swaps_per_day
            * shared.operational_days_per_year
            * expected_station_service_demand
            * fleet_energy_price
        )

        annual_electricity_procurement_cost = (
            daily_energy
            * shared.operational_days_per_year
            * expected_station_service_demand
            * provider_base_energy_price
        )

        annual_station_operating_cost = (
            inp.station_annual_staff_costs
            + inp.station_annual_other_service_costs
        )

        annual_net_cashflow = (
            annual_rent_income
            + annual_swapping_fee_income
            + annual_electricity_sales_income
            - annual_electricity_procurement_cost
            - annual_station_operating_cost
        )

        annual_cashflows.append(annual_net_cashflow)

    cashflows = [initial_investment] + annual_cashflows

    irr = compute_irr_bisection(cashflows)
    payback = compute_payback_period(cashflows)

    return {
        "cashflows": cashflows,
        "irr": irr,
        "payback_period": payback,
        "initial_investment": initial_investment,
        "expected_station_service_demand": expected_station_service_demand,
        "battery_capex": battery_capex,
        "annual_rent_per_truck": annual_rent_per_truck,
        "swaps_per_day": swaps_per_day,
    }


def run_baas_viability_grid(
    shared=None,
    bets_inp=None,
    swapping_fees=None,
    electricity_margins=None,
    battery_rent_returns=None,
):
    if shared is None:
        shared = SharedInputs()

    if bets_inp is None:
        bets_inp = BETSInputs(
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
        )

    if swapping_fees is None:
        swapping_fees = np.arange(3, 6, 1)

    if electricity_margins is None:
        electricity_margins = np.arange(1.0, -0.01, -0.1)

    if battery_rent_returns is None:
        battery_rent_returns = np.arange(0.0, 0.151, 0.05)

    rows = []

    for fee in swapping_fees:
        for margin in electricity_margins:
            for rent_return in battery_rent_returns:
                shared_i = replace(
                    shared,
                    electricity_margin=float(margin),
                )

                bets_i = replace(
                    bets_inp,
                    swapping_fee_flat=float(fee),
                    expected_annual_return_on_battery_renting=float(rent_return),
                )

                result = compute_baas_provider_cashflows(shared_i, bets_i)

                rows.append({
                    "swapping_fee": fee,
                    "electricity_margin": margin,
                    "battery_rent_return": rent_return,
                    "irr": result["irr"],
                    "payback_period": result["payback_period"],
                    "initial_investment": result["initial_investment"],
                    "expected_station_service_demand": result["expected_station_service_demand"],
                    "annual_rent_per_truck": result["annual_rent_per_truck"],
                    "swaps_per_day": result["swaps_per_day"],
                })

    return pd.DataFrame(rows)


def plot_baas_irr_payback_heatmaps(
    df,
    save_path=None,
):
    swapping_fees = sorted(df["swapping_fee"].unique())
    n_cols = len(swapping_fees)

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(4.2 * n_cols, 8),
        constrained_layout=True,
    )

    irr_vmin = df["irr"].min(skipna=True) * 100
    irr_vmax = df["irr"].max(skipna=True) * 100

    payback_vmin = df["payback_period"].min(skipna=True)
    payback_vmax = df["payback_period"].max(skipna=True)

    for col, fee in enumerate(swapping_fees):
        sub = df[df["swapping_fee"] == fee]

        irr_matrix = sub.pivot(
            index="electricity_margin",
            columns="battery_rent_return",
            values="irr",
        ).sort_index(ascending=False)

        payback_matrix = sub.pivot(
            index="electricity_margin",
            columns="battery_rent_return",
            values="payback_period",
        ).sort_index(ascending=False)

        x_labels = [f"{x:.0%}" for x in irr_matrix.columns]
        y_labels = [f"{y:.0%}" for y in irr_matrix.index]

        ax_irr = axes[0, col]
        im1 = ax_irr.imshow(
            irr_matrix.values * 100,
            aspect="auto",
            vmin=irr_vmin,
            vmax=irr_vmax,
        )

        ax_irr.set_title(f"IRR | Swapping fee = £{fee:.0f}")
        ax_irr.set_xticks(range(len(x_labels)))
        ax_irr.set_xticklabels(x_labels)
        ax_irr.set_yticks(range(len(y_labels)))
        ax_irr.set_yticklabels(y_labels)

        if col == 0:
            ax_irr.set_ylabel("Target electricity margin")

        ax_irr.set_xlabel("Battery rent annual return")

        for i in range(irr_matrix.shape[0]):
            for j in range(irr_matrix.shape[1]):
                value = irr_matrix.values[i, j]
                label = "" if np.isnan(value) else f"{value * 100:.1f}%"
                ax_irr.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if value <= irr_matrix.values.mean() else "black")

        ax_payback = axes[1, col]
        im2 = ax_payback.imshow(
            payback_matrix.values,
            aspect="auto",
            vmin=payback_vmin,
            vmax=payback_vmax,
            cmap="viridis_r",
        )

        ax_payback.set_title(f"Payback | Swapping fee = £{fee:.0f}")
        ax_payback.set_xticks(range(len(x_labels)))
        ax_payback.set_xticklabels(x_labels)
        ax_payback.set_yticks(range(len(y_labels)))
        ax_payback.set_yticklabels(y_labels)

        if col == 0:
            ax_payback.set_ylabel("Target electricity margin")

        ax_payback.set_xlabel("Battery rent annual return")

        for i in range(payback_matrix.shape[0]):
            for j in range(payback_matrix.shape[1]):
                value = payback_matrix.values[i, j]
                label = "N/A" if np.isnan(value) else f"{value:.1f}"
                ax_payback.text(j, i, label, ha="center", va="center", fontsize=8, color="black" if value <= 4 else "white")

    fig.colorbar(im1, ax=axes[0, :], shrink=0.75, label="IRR (%)")
    fig.colorbar(im2, ax=axes[1, :], shrink=0.75, label="Payback period (years)")

    fig.suptitle(
        "BaaS Provider Financial Viability: IRR and Payback Period",
        fontsize=16,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig

def run_baas_utilisation_viability_grid(
    shared=None,
    bets_inp=None,
    expected_station_utilisations=None,
    electricity_margins=None,
    battery_rent_returns=None,
    fixed_swapping_fee=3.0,
):
    if shared is None:
        shared = SharedInputs()

    if bets_inp is None:
        bets_inp = BETSInputs(
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
        )

    if expected_station_utilisations is None:
        expected_station_utilisations = np.arange(0.20, 0.50, 0.10)

    if electricity_margins is None:
        electricity_margins = np.arange(1.0, -0.01, -0.1)

    if battery_rent_returns is None:
        battery_rent_returns = np.arange(0.0, 0.151, 0.05)

    rows = []

    for util in expected_station_utilisations:
        for margin in electricity_margins:
            for rent_return in battery_rent_returns:
                shared_i = replace(
                    shared,
                    electricity_margin=float(margin),
                )

                bets_i = replace(
                    bets_inp,
                    swapping_fee_flat=float(fixed_swapping_fee),
                    expected_station_utilisation=float(util),
                    expected_annual_return_on_battery_renting=float(rent_return),
                )

                result = compute_baas_provider_cashflows(shared_i, bets_i)

                rows.append({
                    "expected_station_utilisation": util,
                    "electricity_margin": margin,
                    "battery_rent_return": rent_return,
                    "swapping_fee": fixed_swapping_fee,
                    "irr": result["irr"],
                    "payback_period": result["payback_period"],
                    "initial_investment": result["initial_investment"],
                    "expected_station_service_demand": result["expected_station_service_demand"],
                    "annual_rent_per_truck": result["annual_rent_per_truck"],
                    "swaps_per_day": result["swaps_per_day"],
                })

    return pd.DataFrame(rows)


def plot_baas_utilisation_irr_payback_heatmaps(df):
    utilisations = sorted(df["expected_station_utilisation"].unique())
    n_cols = len(utilisations)

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(4.2 * n_cols, 8),
        constrained_layout=True,
    )

    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    irr_vmin = df["irr"].min(skipna=True) * 100
    irr_vmax = df["irr"].max(skipna=True) * 100

    payback_vmin = df["payback_period"].min(skipna=True)
    payback_vmax = df["payback_period"].max(skipna=True)

    for col, util in enumerate(utilisations):
        sub = df[df["expected_station_utilisation"] == util]

        irr_matrix = sub.pivot(
            index="electricity_margin",
            columns="battery_rent_return",
            values="irr",
        ).sort_index(ascending=False)

        payback_matrix = sub.pivot(
            index="electricity_margin",
            columns="battery_rent_return",
            values="payback_period",
        ).sort_index(ascending=False)

        x_labels = [f"{x:.0%}" for x in irr_matrix.columns]
        y_labels = [f"{y:.0%}" for y in irr_matrix.index]

        ax_irr = axes[0, col]
        im1 = ax_irr.imshow(
            irr_matrix.values * 100,
            aspect="auto",
            vmin=irr_vmin,
            vmax=irr_vmax,
        )

        ax_irr.set_title(f"IRR | Utilisation = {util:.0%}")
        ax_irr.set_xticks(range(len(x_labels)))
        ax_irr.set_xticklabels(x_labels)
        ax_irr.set_yticks(range(len(y_labels)))
        ax_irr.set_yticklabels(y_labels)

        if col == 0:
            ax_irr.set_ylabel("Target electricity margin")

        ax_irr.set_xlabel("Battery rent annual return")

        for i in range(irr_matrix.shape[0]):
            for j in range(irr_matrix.shape[1]):
                value = irr_matrix.values[i, j]
                label = "" if np.isnan(value) else f"{value * 100:.1f}%"
                ax_irr.text(
                    j, i, label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value <= irr_matrix.values.mean() else "black"
                )

        ax_payback = axes[1, col]
        im2 = ax_payback.imshow(
            payback_matrix.values,
            aspect="auto",
            vmin=payback_vmin,
            vmax=payback_vmax,
            cmap="viridis_r",
        )

        ax_payback.set_title(f"Payback | Utilisation = {util:.0%}")
        ax_payback.set_xticks(range(len(x_labels)))
        ax_payback.set_xticklabels(x_labels)
        ax_payback.set_yticks(range(len(y_labels)))
        ax_payback.set_yticklabels(y_labels)

        if col == 0:
            ax_payback.set_ylabel("Target electricity margin")

        ax_payback.set_xlabel("Battery rent annual return")

        for i in range(payback_matrix.shape[0]):
            for j in range(payback_matrix.shape[1]):
                value = payback_matrix.values[i, j]
                label = "N/A" if np.isnan(value) else f"{value:.1f}"
                ax_payback.text(
                    j, i, label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if value <= 4 else "white"
                )

    fig.colorbar(im1, ax=axes[0, :], shrink=0.75, label="IRR (%)")
    fig.colorbar(im2, ax=axes[1, :], shrink=0.75, label="Payback period (years)")

    fig.suptitle(
        "BaaS Provider Financial Viability: IRR and Payback Period by Station Utilisation",
        fontsize=16,
    )
    return plt.gcf()

def run_baas_utilisation_tco_gap_grid(
    shared=None,
    diesel_inp=None,
    bets_inp=None,
    expected_station_utilisations=None,
    electricity_margins=None,
    battery_rent_returns=None,
    fixed_swapping_fee=3.0,
):
    if shared is None:
        shared = SharedInputs()

    if diesel_inp is None:
        diesel_inp = DieselInputs()

    if bets_inp is None:
        bets_inp = BETSInputs(
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
        )

    if expected_station_utilisations is None:
        expected_station_utilisations = np.arange(0.20, 0.50, 0.10)

    if electricity_margins is None:
        electricity_margins = np.arange(1.0, -0.01, -0.1)

    if battery_rent_returns is None:
        battery_rent_returns = np.arange(0.0, 0.151, 0.05)

    rows = []

    for util in expected_station_utilisations:
        for margin in electricity_margins:
            for rent_return in battery_rent_returns:

                shared_i = replace(
                    shared,
                    electricity_margin=float(margin),
                )

                bets_i = replace(
                    bets_inp,
                    swapping_fee_flat=float(fixed_swapping_fee),
                    expected_station_utilisation=float(util),
                    expected_annual_return_on_battery_renting=float(rent_return),
                )

                diesel_result = compute_diesel(shared_i, diesel_inp)
                bets_result = compute_bet_s(shared_i, bets_i)

                rows.append({
                    "expected_station_utilisation": util,
                    "electricity_margin": margin,
                    "battery_rent_return": rent_return,
                    "swapping_fee": fixed_swapping_fee,
                    "bets_tco_discounted": bets_result["tco_discounted"],
                    "diesel_tco_discounted": diesel_result["tco_discounted"],
                    "gap_bets_diesel": bets_result["tco_discounted"] - diesel_result["tco_discounted"],
                    "gap_bets_diesel_per_km": bets_result["tco_per_km_discounted"] - diesel_result["tco_per_km_discounted"],
                })

    return pd.DataFrame(rows)
def plot_baas_utilisation_tco_gap_heatmaps(
    df,
    gap_column="gap_bets_diesel",
    title="TCO Gap between BET-S and Diesel Trucks by Station Utilisation",
):
    utilisations = sorted(df["expected_station_utilisation"].unique())
    n_cols = len(utilisations)

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(4.2 * n_cols, 4.5),
        constrained_layout=True,
    )

    if n_cols == 1:
        axes = np.array([axes])

    gap_vmin = df[gap_column].min(skipna=True)
    gap_vmax = df[gap_column].max(skipna=True)

    abs_max = max(abs(gap_vmin), abs(gap_vmax))

    for col, util in enumerate(utilisations):
        sub = df[df["expected_station_utilisation"] == util]

        gap_matrix = sub.pivot(
            index="electricity_margin",
            columns="battery_rent_return",
            values=gap_column,
        ).sort_index(ascending=False)

        x_labels = [f"{x:.0%}" for x in gap_matrix.columns]
        y_labels = [f"{y:.0%}" for y in gap_matrix.index]

        ax = axes[col]

        im = ax.imshow(
            gap_matrix.values,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-abs_max,
            vmax=abs_max,
        )

        ax.set_title(f"Utilisation = {util:.0%}")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

        ax.set_xlabel("Battery rent annual return")

        if col == 0:
            ax.set_ylabel("Target electricity margin")

        for i in range(gap_matrix.shape[0]):
            for j in range(gap_matrix.shape[1]):
                value = gap_matrix.values[i, j]

                if np.isnan(value):
                    label = "N/A"
                    text_color = "black"
                else:
                    label = f"{value / 1000:.0f}k"
                    text_color = "white" if abs(value) > abs_max * 0.5 else "black"

                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    fig.colorbar(
        im,
        ax=axes,
        shrink=0.75,
        label="BET-S - Diesel discounted TCO (£)",
    )

    fig.suptitle(title, fontsize=16)


    
def run_model(shared=None, diesel_inp=None, betc_inp=None, bets_inp=None, asset_manager_margin: float = 0.10):
    if shared is None:
        shared = SharedInputs()
    if diesel_inp is None:
        diesel_inp = DieselInputs()
    if betc_inp is None:
        betc_inp = BETCInputs(battery_recycle_value_ratio=shared.battery_recycle_value_ratio)
    if bets_inp is None:
        bets_inp = BETSInputs(battery_recycle_value_ratio=shared.battery_recycle_value_ratio)

    diesel = compute_diesel(shared, diesel_inp)
    bet_c = compute_bet_c(shared, betc_inp, asset_manager_margin=asset_manager_margin)
    bet_s = compute_bet_s(shared, bets_inp, asset_manager_margin=asset_manager_margin)

    return {"diesel": diesel, "bet_c": bet_c, "bet_s": bet_s}


def extract_tco_gaps(results):
    bet_c_vs_diesel = results["bet_c"]["tco_discounted"] - results["diesel"]["tco_discounted"]
    bet_s_vs_diesel = results["bet_s"]["tco_discounted"] - results["diesel"]["tco_discounted"]
    bet_s_vs_bet_c = results["bet_s"]["tco_discounted"] - results["bet_c"]["tco_discounted"]
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
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
        )
    if bets_inp is None:
        bets_inp = BETSInputs(
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
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
    include_subsidy_uncertainty=True,
):
    rng = np.random.default_rng(random_seed)
    rows = []
    scenario = subsidy_scenario_label(include_subsidy_uncertainty)

    for margin in margins:
        for i in range(n_runs):
            # ===== sample uncertain inputs (same logic as baseline Monte Carlo) =====
            sampled_discount_rate = sample_uncertain("discount_rate", 0.08, 0.10, 0.12, rng)
            sampled_full_loaded_km_per_day = sample_uncertain("full_loaded_km_per_day", 192.0, 240.0, 288.0, rng)

            sampled_peak_price_per_kwh = sample_uncertain("peak_price_per_kwh", 0.16, 0.20, 0.24, rng)
            sampled_off_peak_share = sample_uncertain("off_peak_share", 0.30, 0.50, 0.70, rng)

            sampled_bet_depot_energy_price_per_kwh = sample_uncertain("bet_depot_energy_price_per_kwh", 0.18, 0.22, 0.28, rng)
            sampled_bet_public_energy_price_per_kwh = sample_uncertain("bet_public_energy_price_per_kwh", 0.30, 0.39, 0.50, rng)

            sampled_full_loaded_kwh_per_km_year1 = sample_uncertain("full_loaded_kwh_per_km_year1", 1.20, 1.37, 1.55, rng)
            sampled_battery_recycle_value_ratio = sample_uncertain("battery_recycle_value_ratio", 0.05, 0.10, 0.20, rng)
            sampled_glider_capex = sample_uncertain("glider_capex", 104000.0, 130000.0, 156000.0, rng)
            sampled_battery_price_per_kwh = sample_uncertain("battery_price_per_kwh", 118.4, 148.0, 177.6, rng)
            sampled_battery_lifetime_cycles = sample_uncertain("battery_lifetime_cycles", 1500.0, 2200.0, 3000.0, rng)
            sampled_unladen_energy_saving = sample_uncertain("unladen_energy_saving", 0.20, 0.25, 0.30, rng)

            sampled_battery_capacity_kwh = sample_uncertain("battery_capacity_kwh", 400.0, 513.0, 800.0, rng)
            sampled_bet_depot_share = sample_uncertain("bet_depot_share", 0, 0.8, 1,rng)

            sampled_expected_station_utilisation = sample_uncertain("expected_station_utilisation", 0.20, 0.30, 0.50, rng)
            sampled_expected_annual_return_on_battery_renting = sample_uncertain("expected_annual_return_on_battery_renting", 0.05, 0.15, 0.25, rng)
            sampled_electricity_margin = sample_uncertain("electricity_margin", 0.2, 1, 1.5, rng)

            sampled_bet_subsidy = sample_bet_subsidy(
                rng,
                include_subsidy_uncertainty=include_subsidy_uncertainty,
            )

            # ===== build sampled inputs =====
            shared_i = SharedInputs(
                discount_rate=sampled_discount_rate,
                full_loaded_km_per_day=sampled_full_loaded_km_per_day,
                peak_price_per_kwh=sampled_peak_price_per_kwh,
                off_peak_share=sampled_off_peak_share,
                bet_depot_energy_price_per_kwh=sampled_bet_depot_energy_price_per_kwh,
                bet_public_energy_price_per_kwh=sampled_bet_public_energy_price_per_kwh,
                bet_subsidy=sampled_bet_subsidy,
                bet_depot_share=sampled_bet_depot_share,
                electricity_margin=sampled_electricity_margin,
            )

            diesel_i = DieselInputs()

            betc_i = BETCInputs(
                battery_recycle_value_ratio=sampled_battery_recycle_value_ratio,
                glider_capex=sampled_glider_capex,
                battery_lifetime_cycles=sampled_battery_lifetime_cycles,
                unladen_energy_saving=sampled_unladen_energy_saving,
                full_loaded_kwh_per_km_year1=sampled_full_loaded_kwh_per_km_year1,
                battery_capacity_kwh=sampled_battery_capacity_kwh,
            )

            bets_i = BETSInputs(
                battery_recycle_value_ratio=sampled_battery_recycle_value_ratio,
                glider_capex=sampled_glider_capex,
                battery_lifetime_cycles=sampled_battery_lifetime_cycles,
                unladen_energy_saving=sampled_unladen_energy_saving,
                full_loaded_kwh_per_km_year1=sampled_full_loaded_kwh_per_km_year1,
                expected_station_utilisation=sampled_expected_station_utilisation,
                expected_annual_return_on_battery_renting=sampled_expected_annual_return_on_battery_renting,
            )

            results = run_model(
                shared=shared_i,
                diesel_inp=diesel_i,
                betc_inp=betc_i,
                bets_inp=bets_i,
                asset_manager_margin=margin,
            )

            rows.append({
                "subsidy_scenario": scenario,
                "asset_manager_margin": margin,
                "iteration": i + 1,
                "diesel_tco_per_km": results["diesel"]["tco_per_km_discounted"],
                "bets_freight_all_in_per_km": results["bet_s"]["freight_total_cost_per_km"],
                "bets_minus_diesel_per_km": results["bet_s"]["freight_total_cost_per_km"] - results["diesel"]["tco_per_km_discounted"],
            })

    return pd.DataFrame(rows)


def run_margin_sweep_with_and_without_subsidy_uncertainty(
    margins,
    n_runs=500,
    random_seed=42,
):
    """Run margin uncertainty twice and return one combined dataframe."""
    with_subsidy = run_margin_sweep_with_uncertainty(
        margins=margins,
        n_runs=n_runs,
        random_seed=random_seed,
        include_subsidy_uncertainty=True,
    )
    no_subsidy = run_margin_sweep_with_uncertainty(
        margins=margins,
        n_runs=n_runs,
        random_seed=random_seed,
        include_subsidy_uncertainty=False,
    )
    return pd.concat([with_subsidy, no_subsidy], ignore_index=True)

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
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
        )
    if bets_inp is None:
        bets_inp = BETSInputs(
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
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

    x_labels = []
    
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
                "battery_recycle_value_ratio",
                shared_i.battery_recycle_value_ratio
            )
            bets_i = update_input(
                bets_i,
                "battery_recycle_value_ratio",
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

        if "rate" in variable_name or "utilisation" in variable_name or "share" in variable_name or "ratio" in variable_name or "return" in variable_name or "margin" in variable_name: #改单位
            x_labels.append(f"{new_value:.0%}")
        elif "km" in variable_name:
            x_labels.append(f"{new_value:.0f}")
        elif "price" in variable_name or "cost" in variable_name:
            x_labels.append(f"{new_value:.2f}")
        else:
            x_labels.append(f"{new_value:g}")

    return {
        "target_class": target_classes,
        "variable_name": variable_name,
        "base_value": base_value,
        "labels": labels,
        "bet_c_vs_diesel": bet_c_vs_diesel,
        "bet_s_vs_diesel": bet_s_vs_diesel,
        "bet_s_vs_bet_c": bet_s_vs_bet_c,
        "x_labels": x_labels,
    }

# Run multiple sensitivity analyses for different variables
def run_multiple_sensitivity_analyses(specs):
    all_results = []

    for spec in specs:
        
        if "direct_values" in spec:
            result = run_sensitivity_analysis_direct_values(
                target_class=spec["target_class"],
                variable_name=spec["variable_name"],
                values=spec["direct_values"],
                base_value=spec.get("base_value"),
            )
        else:
            changes = spec.get("changes", [-0.40, -0.20, 0.0, 0.20, 0.40])


            result = run_sensitivity_analysis(
                target_class=spec["target_class"],
                variable_name=spec["variable_name"],
                base_value=spec["base_value"],
                changes=changes,
            )
        all_results.append(result)

    return all_results
    # ===== multiple sensitivity analyses =====

sensitivity_specs = [
    {
        "target_class": "shared",
        "variable_name": "full_loaded_km_per_day",
        "base_value": 240.0,
        "changes": [-0.50, -0.20, 0.0, 0.40, 0.80],
    },
    {
        "target_class": "shared",
        "variable_name": "battery_recycle_value_ratio",
        "base_value": 0.10,
        "changes": [-0.50, 0.0, 1.00, 2.00, 3.00],
    },
      
    {
        "target_class": ["betc", "bets"],
        "variable_name": "battery_price_per_kwh",
        "base_value": 148.0,
        "changes": [-0.40, -0.20, 0.0, 0.10, 0.20],
    },
    
    {
        "target_class": ["betc", "bets"],
        "variable_name": "battery_lifetime_cycles",
        "base_value": 2200.0,
        "changes": [-0.20, 0.0, 0.40, 0.80, 1.00],
    },
    {
        "target_class": ["shared"],
        "variable_name": "off_peak_share",
        "base_value": 0.50,
        "changes": [-1.00, -0.50, 0.00, 0.50, 1.00],
    },
    {
        "target_class": ["bets"],
        "variable_name": "expected_station_utilisation",
        "base_value": 0.30,
        "changes": [-0.50, 0.0, 0.50, 1.00, 1.50],
    },
    {
        "target_class": ["bets"],
        "variable_name": "expected_annual_return_on_battery_renting",
        "base_value": 0.15,
        "changes": [-1.0, -0.50, 0, 0.50, 1.00],
    },
    {
        "target_class": ["shared"],
        "variable_name": "electricity_margin",
        "base_value": 1.00,
        "changes": [-0.90, -0.60,-0.30, 0, 0.50],
    },
    {
        "target_class": ["shared"],
        "variable_name": "bet_subsidy",
        "base_value": 0.00,
        "direct_values": [0,10000,30000,60000,90000],
    },
    {
        "target_class": ["shared"],
        "variable_name": "bet_depot_share",
        "base_value": 0.80,
        "changes": [-1.0 ,-0.625, -0.25, 0, 0.25],
    },
    {
        "target_class": ["shared"],
        "variable_name": "years",
        "base_value": 5.00,
        "direct_values": [3 ,4, 5, 6, 7],
    },
    {
        "target_class": ["shared"],
        "variable_name": "shift_per_day",
        "base_value": 1.00,
        "direct_values": [1, 2],
    },

    
    
    
    
]

def run_sensitivity_analysis_direct_values(
    target_class,
    variable_name,
    values,
    base_value=None,
):
    if isinstance(target_class, str):
        target_classes = [target_class]
    else:
        target_classes = target_class

    bet_c_vs_diesel = []
    bet_s_vs_diesel = []
    bet_s_vs_bet_c = []
    x_labels = []

    for value in values:
        shared_i = SharedInputs()
        diesel_i = DieselInputs()
        betc_i = BETCInputs(
            battery_recycle_value_ratio=shared_i.battery_recycle_value_ratio
        )
        bets_i = BETSInputs(
            battery_recycle_value_ratio=shared_i.battery_recycle_value_ratio
        )

        # ===== update variable =====
        if "shared" in target_classes:
            shared_i = update_input(shared_i, variable_name, value)

        if "diesel" in target_classes:
            diesel_i = update_input(diesel_i, variable_name, value)

        if "betc" in target_classes:
            betc_i = update_input(betc_i, variable_name, value)

        if "bets" in target_classes:
            bets_i = update_input(bets_i, variable_name, value)

        results = run_model(
            shared=shared_i,
            diesel_inp=diesel_i,
            betc_inp=betc_i,
            bets_inp=bets_i,
        )

        gaps = extract_tco_gaps(results)

        bet_c_vs_diesel.append(gaps["bet_c_vs_diesel"])
        bet_s_vs_diesel.append(gaps["bet_s_vs_diesel"])
        bet_s_vs_bet_c.append(gaps["bet_s_vs_bet_c"])

        x_labels.append(f"{value:,.0f}")

    return {
        "target_class": target_classes,
        "variable_name": variable_name,
        "base_value": base_value,
        "labels": x_labels,
        "x_labels": x_labels,
        "bet_c_vs_diesel": bet_c_vs_diesel,
        "bet_s_vs_diesel": bet_s_vs_diesel,
        "bet_s_vs_bet_c": bet_s_vs_bet_c,
    }


################ Monte Carlo Sampling ################################
UNCERTAINTY_NOTE = "Shaded area: 5th-95th percentile range across Monte Carlo simulations"

def sample_triangular(left, mode, right, rng):
    return rng.triangular(left, mode, right)


# Sidebar-editable Monte Carlo uncertainty ranges.
# The app can set this before running cached simulations.
UNCERTAINTY_OVERRIDES = {}


def set_uncertainty_overrides(overrides=None):
    global UNCERTAINTY_OVERRIDES
    UNCERTAINTY_OVERRIDES = overrides or {}


def _uncertainty_bounds(variable, left, mode, right):
    override = UNCERTAINTY_OVERRIDES.get(variable, {}) if isinstance(UNCERTAINTY_OVERRIDES, dict) else {}
    left = float(override.get("left", left))
    mode = float(override.get("mode", mode))
    right = float(override.get("right", right))
    if left > right:
        left, right = right, left
    mode = min(max(mode, left), right)
    return left, mode, right


def sample_uncertain(variable, left, mode, right, rng):
    left, mode, right = _uncertainty_bounds(variable, left, mode, right)
    return sample_triangular(left, mode, right, rng)


def sample_bet_subsidy(rng, include_subsidy_uncertainty=True):
    """Sample BET purchase subsidy for Monte Carlo runs.

    include_subsidy_uncertainty=True  -> triangular(0, 0, 120000)
    include_subsidy_uncertainty=False -> fixed 0
    """
    if include_subsidy_uncertainty:
        return sample_uncertain("bet_subsidy", 0.0, 0.0, 120000.0, rng)
    return 0.0


def subsidy_scenario_label(include_subsidy_uncertainty=True):
    return "With subsidy uncertainty" if include_subsidy_uncertainty else "No subsidy"

# =========================================================
# Independent-effect Monte Carlo for one-at-a-time boxplots
# =========================================================

# Define uncertain variables and their distributions
def get_uncertainty_specs(include_subsidy_uncertainty=True, uncertainty_overrides=None):
    """
    target_class:
        - "shared"
        - "diesel"
        - "betc"
        - "bets"
        - ["betc", "bets"]  # jointly changed in both BET-C and BET-S
    """
    specs = [
        {
            "variable": "expected_station_utilisation",
            "target_class": "bets",
            "left": 0.20,
            "mode": 0.30,
            "right": 0.50,
        },
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
            "variable": "battery_recycle_value_ratio",
            "target_class": ["betc", "bets"],
            "left": 0.05,
            "mode": 0.10,
            "right": 0.20,
        },
        {
            "variable": "battery_lifetime_cycles",
            "target_class": ["betc", "bets"],
            "left": 1600.0,
            "mode": 2200.0,
            "right": 3000.0,
        },
        {
            "variable": "glider_capex",
            "target_class": ["betc", "bets"],
            "left": 104000.0,
            "mode": 130000.0,
            "right": 156000.0,
        },
        {
            "variable": "battery_price_per_kwh",
            "target_class": ["betc", "bets"],
            "left": 118.4,
            "mode": 148.0,
            "right": 177.6,
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
            "mode": 513.0,
            "right": 800.0,
        },
        {
            "variable": "expected_annual_return_on_battery_renting",
            "target_class": "bets",
            "left": 0.05,
            "mode": 0.15,
            "right": 0.25,
        },
        {
            "variable": "electricity_margin",
            "target_class": "shared",
            "left": 0.50,
            "mode": 1.00,
            "right": 1.50,
        },
        {
            "variable": "bet_depot_share",
            "target_class": "shared",
            "left": 0.00,
            "mode": 0.80,
            "right": 1.00,
        },
        
        
        
    ]

    if include_subsidy_uncertainty:
        specs.append({
            "variable": "bet_subsidy",
            "target_class": "shared",
            "left": 0,
            "mode": 0,
            "right": 120000.0,
        })

    overrides = uncertainty_overrides if uncertainty_overrides is not None else UNCERTAINTY_OVERRIDES
    if overrides:
        for spec in specs:
            variable = spec["variable"]
            if variable in overrides:
                left, mode, right = _uncertainty_bounds(
                    variable, spec["left"], spec["mode"], spec["right"]
                )
                spec["left"], spec["mode"], spec["right"] = left, mode, right

    return specs

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
def run_independent_variable_monte_carlo(n_runs=500, random_seed=42, include_subsidy_uncertainty=True):
    """
    For each uncertain variable:
    - vary ONLY that variable according to its triangular distribution
    - keep all other variables at baseline
    - run model n_runs times
    Returns a long dataframe for boxplotting.
    """
    rng = np.random.default_rng(random_seed)
    scenario = subsidy_scenario_label(include_subsidy_uncertainty)
    specs = get_uncertainty_specs(include_subsidy_uncertainty=include_subsidy_uncertainty)

    base_shared = SharedInputs()
    base_diesel = DieselInputs()
    base_betc = BETCInputs(
        battery_recycle_value_ratio=base_shared.battery_recycle_value_ratio
    )
    base_bets = BETSInputs(
        battery_recycle_value_ratio=base_shared.battery_recycle_value_ratio
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

            diesel_tco = results["diesel"]["tco_discounted"]
            betc_tco = results["bet_c"]["tco_discounted"]
            bets_tco = results["bet_s"]["tco_discounted"]

            rows.append({
                "subsidy_scenario": scenario,
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


def run_independent_variable_monte_carlo_with_and_without_subsidy(
    n_runs=500,
    random_seed=42,
):
    """Run independent-variable MC twice and return one combined dataframe."""
    with_subsidy = run_independent_variable_monte_carlo(
        n_runs=n_runs,
        random_seed=random_seed,
        include_subsidy_uncertainty=True,
    )
    no_subsidy = run_independent_variable_monte_carlo(
        n_runs=n_runs,
        random_seed=random_seed,
        include_subsidy_uncertainty=False,
    )
    return pd.concat([with_subsidy, no_subsidy], ignore_index=True)


def run_monte_carlo_simulation(n_runs=500, random_seed=42, include_subsidy_uncertainty=True):
    rng = np.random.default_rng(random_seed)
    rows = []
    scenario = subsidy_scenario_label(include_subsidy_uncertainty)

    for i in range(n_runs):
        # ===== 1) sample uncertain inputs (triangular distributions) =====
        sampled_discount_rate = sample_uncertain("discount_rate", 0.08, 0.10, 0.12, rng)

        sampled_full_loaded_km_per_day = sample_uncertain("full_loaded_km_per_day", 192.0, 240.0, 288.0, rng)

        sampled_peak_price_per_kwh = sample_uncertain("peak_price_per_kwh", 0.16, 0.20, 0.24, rng)
        sampled_off_peak_share = sample_uncertain("off_peak_share", 0.30, 0.50, 0.70, rng)

        sampled_bet_depot_energy_price_per_kwh = sample_uncertain("bet_depot_energy_price_per_kwh", 0.18, 0.22, 0.28, rng)
        sampled_bet_public_energy_price_per_kwh = sample_uncertain("bet_public_energy_price_per_kwh", 0.30, 0.39, 0.50, rng)

        # BET-C and BET-S jointly changing variables
        sampled_full_loaded_kwh_per_km_year1 = sample_uncertain("full_loaded_kwh_per_km_year1", 1.20, 1.37, 1.55, rng)
        sampled_battery_recycle_value_ratio = sample_uncertain("battery_recycle_value_ratio", 0.05, 0.10, 0.20, rng)
        sampled_glider_capex = sample_uncertain("glider_capex", 104000.0, 130000.0, 156000.0, rng)
        sampled_battery_price_per_kwh = sample_uncertain("battery_price_per_kwh", 118.4, 148.0, 177.6, rng)
        sampled_battery_lifetime_cycles = sample_uncertain("battery_lifetime_cycles", 1600.0, 2200.0, 3000.0, rng)
        sampled_unladen_energy_saving = sample_uncertain("unladen_energy_saving", 0.2, 0.25, 0.3, rng)
        sampled_bet_subsidy = sample_bet_subsidy(
            rng,
            include_subsidy_uncertainty=include_subsidy_uncertainty,
        )

        # BET-C only
        sampled_battery_capacity_kwh = sample_uncertain("battery_capacity_kwh", 400.0, 513.0, 800.0, rng)
        sampled_bet_depot_share = sample_uncertain("bet_depot_share", 0, 0.8, 1,rng)
        # BET-S only


        sampled_expected_station_utilisation = sample_uncertain("expected_station_utilisation", 0.20, 0.30, 0.50, rng)
        sampled_expected_annual_return_on_battery_renting = sample_uncertain("expected_annual_return_on_battery_renting", 0.05, 0.15, 0.25, rng)
        sampled_electricity_margin = sample_uncertain("electricity_margin", 0.2, 1, 1.5, rng)


        # ===== 2) build sampled inputs =====
        shared_i = SharedInputs(
            discount_rate=sampled_discount_rate,
            full_loaded_km_per_day=sampled_full_loaded_km_per_day,
            peak_price_per_kwh=sampled_peak_price_per_kwh,
            off_peak_share=sampled_off_peak_share,
            bet_depot_energy_price_per_kwh=sampled_bet_depot_energy_price_per_kwh,
            bet_public_energy_price_per_kwh=sampled_bet_public_energy_price_per_kwh,
            bet_subsidy=sampled_bet_subsidy,
            bet_depot_share=sampled_bet_depot_share,
            electricity_margin=sampled_electricity_margin,
        )

        diesel_i = DieselInputs()

        betc_i = BETCInputs(
            battery_recycle_value_ratio=sampled_battery_recycle_value_ratio,
            battery_lifetime_cycles=sampled_battery_lifetime_cycles,
            glider_capex=sampled_glider_capex,
            battery_price_per_kwh=sampled_battery_price_per_kwh,
            unladen_energy_saving=sampled_unladen_energy_saving,
            full_loaded_kwh_per_km_year1=sampled_full_loaded_kwh_per_km_year1,
            battery_capacity_kwh=sampled_battery_capacity_kwh,
        )

        bets_i = BETSInputs(
            battery_recycle_value_ratio=sampled_battery_recycle_value_ratio,
            battery_lifetime_cycles=sampled_battery_lifetime_cycles,
            glider_capex=sampled_glider_capex,
            battery_price_per_kwh=sampled_battery_price_per_kwh,
            unladen_energy_saving=sampled_unladen_energy_saving,
            full_loaded_kwh_per_km_year1=sampled_full_loaded_kwh_per_km_year1,
            expected_station_utilisation=sampled_expected_station_utilisation,
            expected_annual_return_on_battery_renting=sampled_expected_annual_return_on_battery_renting,
            
        )

        # ===== 3) run model =====
        diesel = compute_diesel(shared_i, diesel_i)
        bet_c = compute_bet_c(shared_i, betc_i)
        bet_s = compute_bet_s(shared_i, bets_i)

        diesel_tco = diesel["tco_discounted"]
        bet_c_tco = bet_c["tco_discounted"]
        bet_s_tco = bet_s["tco_discounted"]

        gap_bet_c_diesel = bet_c_tco - diesel_tco
        gap_bet_s_diesel = bet_s_tco - diesel_tco
        gap_bet_s_bet_c = bet_s_tco - bet_c_tco

        rows.append({
            "subsidy_scenario": scenario,
            "iteration": i + 1,

            "discount_rate": sampled_discount_rate,
            "full_loaded_km_per_day": sampled_full_loaded_km_per_day,
            "peak_price_per_kwh": sampled_peak_price_per_kwh,
            "off_peak_share": sampled_off_peak_share,
            "bet_depot_energy_price_per_kwh": sampled_bet_depot_energy_price_per_kwh,
            "bet_public_energy_price_per_kwh": sampled_bet_public_energy_price_per_kwh,

            "full_loaded_kwh_per_km_year1": sampled_full_loaded_kwh_per_km_year1,
            "battery_recycle_value_ratio": sampled_battery_recycle_value_ratio,
            "battery_lifetime_cycles": sampled_battery_lifetime_cycles,
            "glider_capex": sampled_glider_capex,
            "battery_price_per_kwh": sampled_battery_price_per_kwh,
            "unladen_energy_saving": sampled_unladen_energy_saving,

            "battery_capacity_kwh": sampled_battery_capacity_kwh,

            "expected_station_utilisation": sampled_expected_station_utilisation,
            "expected_annual_return_on_battery_renting": sampled_expected_annual_return_on_battery_renting,
            "electricity_margin":sampled_electricity_margin,

            "bet_depot_share":sampled_bet_depot_share,

            "bet_subsidy": sampled_bet_subsidy,

            "diesel_tco": diesel_tco,
            "bet_c_tco": bet_c_tco,
            "bet_s_tco": bet_s_tco,
            "gap_bet_c_diesel": gap_bet_c_diesel,
            "gap_bet_s_diesel": gap_bet_s_diesel,
            "gap_bet_s_bet_c": gap_bet_s_bet_c,
        })

    return pd.DataFrame(rows)


def run_monte_carlo_simulation_with_and_without_subsidy(
    n_runs=500,
    random_seed=42,
):
    """Run full Monte Carlo twice and return one combined dataframe."""
    with_subsidy = run_monte_carlo_simulation(
        n_runs=n_runs,
        random_seed=random_seed,
        include_subsidy_uncertainty=True,
    )
    no_subsidy = run_monte_carlo_simulation(
        n_runs=n_runs,
        random_seed=random_seed,
        include_subsidy_uncertainty=False,
    )
    return pd.concat([with_subsidy, no_subsidy], ignore_index=True)

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
    include_subsidy_uncertainty=True,
):
    """
    For each purchase year:
    1. build projected baseline inputs for that year
    2. run Monte Carlo around that year's projected values
    3. collect TCO distributions
    """
    rng = np.random.default_rng(random_seed)
    rows = []
    scenario = subsidy_scenario_label(include_subsidy_uncertainty)

    for year in range(start_year, end_year + 1):
        shared_base, diesel_base, betc_base, bets_base = build_projected_inputs_for_year(
            target_year=year,
            base_year=start_year,
            shared=SharedInputs(),
            diesel_inp=DieselInputs(),
            betc_inp=BETCInputs(
                battery_recycle_value_ratio=SharedInputs().battery_recycle_value_ratio
            ),
            bets_inp=BETSInputs(
                battery_recycle_value_ratio=SharedInputs().battery_recycle_value_ratio
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

            sampled_battery_recycle_value_ratio = sample_triangular(
                max(0.0, SharedInputs().battery_recycle_value_ratio * 0.5),
                SharedInputs().battery_recycle_value_ratio,
                min(1.0, SharedInputs().battery_recycle_value_ratio * 2.0),
                rng
            )

            sampled_battery_price_per_kwh = sample_triangular(
                betc_base.battery_price_per_kwh * 0.8,
                betc_base.battery_price_per_kwh,
                betc_base.battery_price_per_kwh * 1.2,
                rng
            )
            sampled_glider_capex = sample_triangular(
                betc_base.glider_capex * 0.8,
                betc_base.glider_capex,
                betc_base.glider_capex * 1.2,
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
            
            sampled_bet_subsidy = sample_bet_subsidy(
                rng,
                include_subsidy_uncertainty=include_subsidy_uncertainty,
            )
            
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
                battery_recycle_value_ratio=sampled_battery_recycle_value_ratio,
                glider_capex=sampled_glider_capex,
                battery_price_per_kwh=sampled_battery_price_per_kwh,
                battery_lifetime_cycles=sampled_battery_lifetime_cycles,
                unladen_energy_saving=sampled_unladen_energy_saving,
                full_loaded_kwh_per_km_year1=sampled_full_loaded_kwh_per_km_year1,
                battery_capacity_kwh=sampled_battery_capacity_kwh,
            )

            bets_i = replace(
                bets_base,
                battery_recycle_value_ratio=sampled_battery_recycle_value_ratio,
                glider_capex=sampled_glider_capex,
                battery_price_per_kwh=sampled_battery_price_per_kwh,
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
                "subsidy_scenario": scenario,
                "year": year,
                "iteration": i + 1,

                "diesel_tco_discounted": results["diesel"]["tco_discounted"],
                "betc_tco_discounted": results["bet_c"]["tco_discounted"],
                "bets_tco_discounted": results["bet_s"]["tco_discounted"],

                "diesel_tco_per_km": results["diesel"]["tco_per_km_discounted"],
                "betc_tco_per_km": results["bet_c"]["tco_per_km_discounted"],
                "bets_tco_per_km": results["bet_s"]["tco_per_km_discounted"],

                "diesel_tco_per_kwh": results["diesel"]["tco_per_kwh_discounted"],
                "betc_tco_per_kwh": results["bet_c"]["tco_per_kwh_discounted"],
                "bets_tco_per_kwh": results["bet_s"]["tco_per_kwh_discounted"],
            })

    return pd.DataFrame(rows)


def run_projection_monte_carlo_with_and_without_subsidy(
    start_year=2026,
    end_year=2040,
    n_runs=500,
    random_seed=42,
):
    with_subsidy = run_projection_monte_carlo(
        start_year=start_year,
        end_year=end_year,
        n_runs=n_runs,
        random_seed=random_seed,
        include_subsidy_uncertainty=True,
    )
    no_subsidy = run_projection_monte_carlo(
        start_year=start_year,
        end_year=end_year,
        n_runs=n_runs,
        random_seed=random_seed,
        include_subsidy_uncertainty=False,
    )
    return pd.concat([with_subsidy, no_subsidy], ignore_index=True)

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

    group_cols = [None]
    if "subsidy_scenario" in df.columns:
        group_cols = list(df["subsidy_scenario"].drop_duplicates())

    summary_rows = []
    probability_rows = []

    for scenario in group_cols:
        group = df if scenario is None else df[df["subsidy_scenario"] == scenario]
        for m in metrics:
            row = {
                "metric": m,
                "mean": group[m].mean(),
                "median": group[m].median(),
                "p5": group[m].quantile(0.05),
                "p95": group[m].quantile(0.95),
                "min": group[m].min(),
                "max": group[m].max(),
            }
            if scenario is not None:
                row["subsidy_scenario"] = scenario
            summary_rows.append(row)

        for metric, col in [
            ("P(BET-C - Diesel < 0)", "gap_bet_c_diesel"),
            ("P(BET-S - Diesel < 0)", "gap_bet_s_diesel"),
            ("P(BET-S - BET-C < 0)", "gap_bet_s_bet_c"),
        ]:
            row = {"metric": metric, "probability": (group[col] < 0).mean()}
            if scenario is not None:
                row["subsidy_scenario"] = scenario
            probability_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(probability_rows)

def summarize_projection_uncertainty(df, metric_cols=None):
    if metric_cols is None:
        metric_cols = [
            "diesel_tco_discounted",
            "betc_tco_discounted",
            "bets_tco_discounted",
        ]

    rows = []

    group_cols = ["year"]
    if "subsidy_scenario" in df.columns:
        group_cols = ["subsidy_scenario", "year"]

    for keys, group in df.groupby(group_cols):
        if "subsidy_scenario" in df.columns:
            scenario, year = keys
            row = {"subsidy_scenario": scenario, "year": year}
        else:
            year = keys
            row = {"year": year}

        for col in metric_cols:
            row[f"{col}_p5"] = group[col].quantile(0.05)
            row[f"{col}_p50"] = group[col].quantile(0.50)
            row[f"{col}_p95"] = group[col].quantile(0.95)
            row[f"{col}_mean"] = group[col].mean()

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
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
        )
    if bets_inp is None:
        bets_inp = BETSInputs(
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
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
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
        )
    if bets_inp is None:
        bets_inp = BETSInputs(
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
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
            "betc_tco_discounted": results["bet_c"]["tco_discounted"],
            "bets_tco_discounted": results["bet_s"]["tco_discounted"],

            # ===== per km TCO =====
            "diesel_tco_per_km": results["diesel"]["tco_per_km_discounted"],
            "betc_tco_per_km": results["bet_c"]["tco_per_km_discounted"],
            "bets_tco_per_km": results["bet_s"]["tco_per_km_discounted"],

            # ===== per kWh TCO =====
            "diesel_tco_per_kwh": results["diesel"]["tco_per_kwh_discounted"],
            "betc_tco_per_kwh": results["bet_c"]["tco_per_kwh_discounted"],
            "bets_tco_per_kwh": results["bet_s"]["tco_per_kwh_discounted"],

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
        results["bet_c"]["tco_discounted"],
        results["bet_s"]["tco_discounted"],
    ]

    plt.figure()
    bars = plt.bar(
        labels,
        values,
        color=["tab:blue", "tab:orange", "tab:green"]
    )

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
    return plt.gcf()

def plot_tco_per_km_comparison(results):
    labels = ["Diesel", "BET-C", "BET-S"]
    values = [
        results["diesel"]["tco_per_km_discounted"],
        results["bet_c"]["tco_per_km_discounted"],
        results["bet_s"]["tco_per_km_discounted"],
    ]

    plt.figure()
    bars = plt.bar(
        labels,
        values,
        color=["tab:blue", "tab:orange", "tab:green"]
    )

    plt.title("Discounted TCO per km Comparison")
    plt.ylabel("TCO (£/km)")
    plt.xlabel("Truck Type")

    for bar, v in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:,.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    return plt.gcf()
    
def plot_tco_gap(results):
    labels = [
        "BET-C - Diesel",
        "BET-S - Diesel",
        "BET-S - BET-C"
    ]

    bet_c_gap = (
        results["bet_c"]["tco_discounted"]
        - results["diesel"]["tco_discounted"]
    )

    bet_s_gap = (
        results["bet_s"]["tco_discounted"]
        - results["diesel"]["tco_discounted"]
    )

    bet_s_vs_bet_c_gap = (
        results["bet_s"]["tco_discounted"]
        - results["bet_c"]["tco_discounted"]
    )

    values = [bet_c_gap, bet_s_gap, bet_s_vs_bet_c_gap]

    plt.figure()
    bars = plt.bar(
        labels,
        values,
        color=["tab:purple", "tab:red", "tab:brown"]
    )

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
    return plt.gcf()

def plot_tco_per_km_gap(results):
    labels = [
        "BET-C - Diesel",
        "BET-S - Diesel",
        "BET-S - BET-C"
    ]

    bet_c_gap_per_km = (
        results["bet_c"]["tco_per_km_discounted"]
        - results["diesel"]["tco_per_km_discounted"]
    )

    bet_s_gap_per_km = (
        results["bet_s"]["tco_per_km_discounted"]
        - results["diesel"]["tco_per_km_discounted"]
    )

    bet_s_vs_bet_c_gap_per_km = (
        results["bet_s"]["tco_per_km_discounted"]
        - results["bet_c"]["tco_per_km_discounted"]
    )

    values = [bet_c_gap_per_km, bet_s_gap_per_km, bet_s_vs_bet_c_gap_per_km]

    plt.figure()
    bars = plt.bar(
        labels,
        values,
        color=["tab:purple", "tab:red", "tab:brown"]
    )

    plt.title("TCO per km Gaps")
    plt.ylabel("Difference (£/km)")
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
            f"{v:,.3f}",
            ha="center",
            va=va
        )

    plt.tight_layout()
    return plt.gcf()

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
    return plt.gcf()

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
    return plt.gcf()
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
    return plt.gcf()


def plot_sensitivity_bar(sensitivity_results, title=None):    #画敏感性分析图改名字

    name_map = {
                "battery_price_per_kwh": "Battery price (£/kWh)",
                "battery_recycle_value_ratio": "Battery residual percentage",
                "full_loaded_km_per_day": "Full-loaded daily mileage (km/day)",
                "diesel_public_price_per_l": "Diesel price (£/L)",
                "discount_rate": "Discount rate (%)",
                "bet_depot_energy_price_per_kwh": "Electricity price (£/kWh)",
                "expected_station_utilisation": "Expected Station Utilisation",
                "expected_annual_return_on_battery_renting": "Expected Annual Return on Battery Renting",
                "electricity_margin": "Target Electricity Margin",
                "bet_subsidy": "BET Purchase Subsidy",
                "bet_depot_share": "Depot Slow Charging Percentage",
                "shift_per_day": "Shift per Day",
                "off_peak_share": "Off-peak Swapping Percentage",
                "years": "TCO Horizon",
                "battery_lifetime_cycles": "Battery Lifetime Cycles",
    }
            
    if "x_labels" in sensitivity_results:
        labels = sensitivity_results["x_labels"]
    else:
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
    plt.xlabel(name_map.get(sensitivity_results["variable_name"], sensitivity_results["variable_name"]))
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
    return plt.gcf()

# Summarize uncertainty results using percentiles for plots
def summarize_margin_uncertainty(df):
    rows = []

    group_cols = ["asset_manager_margin"]
    if "subsidy_scenario" in df.columns:
        group_cols = ["subsidy_scenario", "asset_manager_margin"]

    for keys, group in df.groupby(group_cols):
        if "subsidy_scenario" in df.columns:
            scenario, margin = keys
            row = {
                "subsidy_scenario": scenario,
                "asset_manager_margin": margin,
            }
        else:
            margin = keys
            row = {
                "asset_manager_margin": margin,
            }

        row.update({

            "diesel_p5": group["diesel_tco_per_km"].quantile(0.05),
            "diesel_p50": group["diesel_tco_per_km"].quantile(0.50),
            "diesel_p95": group["diesel_tco_per_km"].quantile(0.95),
            "diesel_mean": group["diesel_tco_per_km"].mean(),

            "bets_p5": group["bets_freight_all_in_per_km"].quantile(0.05),
            "bets_p50": group["bets_freight_all_in_per_km"].quantile(0.50),
            "bets_p95": group["bets_freight_all_in_per_km"].quantile(0.95),
            "bets_mean": group["bets_freight_all_in_per_km"].mean(),

            "gap_p5": group["bets_minus_diesel_per_km"].quantile(0.05),
            "gap_p50": group["bets_minus_diesel_per_km"].quantile(0.50),
            "gap_p95": group["bets_minus_diesel_per_km"].quantile(0.95),
            "gap_mean": group["bets_minus_diesel_per_km"].mean(),
        })
        rows.append(row)

    sort_cols = ["asset_manager_margin"]
    if rows and "subsidy_scenario" in rows[0]:
        sort_cols = ["subsidy_scenario", "asset_manager_margin"]
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)

# Plot margin vs cost with uncertainty bands
def plot_margin_vs_freight_all_in_per_km_with_uncertainty(summary_df, title_suffix=""):
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

    # Diesel mean
    plt.plot(
        x,
        summary_df["diesel_mean"],
        linestyle="--",
        linewidth=2,
        color="tab:blue",
        alpha=0.8,
        label="Diesel truck TCO per km (mean)"
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

    # BET-S mean
    plt.plot(
        x,
        summary_df["bets_mean"],
        linestyle="--",
        linewidth=2,
        color="tab:green",
        alpha=0.8,
        label="BET-S AEaaS cost per km (mean)"
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
    plt.title(f"Impact of Asset-manager Margin on Freight Cost per km with Uncertainty {title_suffix}".strip())
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.text(
        0.01, 0.98,
        UNCERTAINTY_NOTE,
        transform=plt.gca().transAxes,
        ha="left",
        va="top"
    )
    plt.tight_layout()
    return plt.gcf()

# Plot margin vs cost gap with uncertainty bands
def plot_margin_vs_gap_with_uncertainty(summary_df, title_suffix=""):
    plt.figure(figsize=(10, 6))

    x = summary_df["asset_manager_margin"] * 100
    mean = summary_df["gap_mean"]

    plt.plot(
        x,
        summary_df["gap_p50"],
        marker="o",
        label="BET-S AEaaS - Diesel (median)"
    )

    plt.plot(
        x,
        mean,
        linestyle="--",
        linewidth=2,
        color="tab:blue",
        alpha=0.8,
        label="Mean"
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
    plt.title(f"Effect of Asset-manager Margin on BET-S AEaaS - Diesel Gap with Uncertainty {title_suffix}".strip())
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.text(
        0.01, 0.98,
        UNCERTAINTY_NOTE,
        transform=plt.gca().transAxes,
        ha="left",
        va="top"
    )
    plt.tight_layout()
    return plt.gcf()

def plot_projection_with_uncertainty(summary_df, title_suffix=""):
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
        mean = summary_df[f"{metric}_mean"]

        plt.plot(years, p50, marker="o", color=color, label=f"{label} median")
        plt.fill_between(years, p5, p95, color=color, alpha=0.2)
        plt.plot(
            years,
            mean,
            linestyle="--",
            linewidth=2,
            color=color,
            alpha=0.9,
            label=f"{label} mean"
        )

    plt.title(f"Projected Discounted TCO with Uncertainty {title_suffix}".strip())
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
    return plt.gcf()
    



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
        mean_value = df[col].mean()
        plt.axvline(df[col].mean(), color = "black", linestyle="--", label="Mean")
        plt.text(mean_value,plt.ylim()[1] * 0.9,f"Mean = {mean_value:,.0f}",va="top",ha="right")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.tight_layout()
        return plt.gcf()



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
    return plt.gcf()
    
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

        # 每组中心位置，用来放变量名
        group_centers.append(base + 1)

        # 只给前 n-1 组画右侧分隔线
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
    # ===== add mean markers =====
        means = [np.mean(d) for d in data]

        ax.scatter(
            positions,
            means,
            color="tab:blue",
            marker="D",   # diamond
            s=40,
            zorder=3
        )
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
    ax.text(
        0.02,
        0.05,
        "Black line in the boxes = median\nBlue diamond = mean",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="none"
        )
    )
    
    plt.tight_layout()
    return plt.gcf()


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

        # ===== add mean markers =====
        means = [np.mean(d) for d in data]

        ax.scatter(
            positions,
            means,
            color="tab:blue",
            marker="D",   # diamond
            s=40,
            zorder=3
        )

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
    ax.text(
        0.02,
        0.05,
        "Black line in the boxes = median\nBlue diamond = mean",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="none"
        )
    )

    ax.set_xlim(min(positions) - 1, max(positions) + 1)

    plt.tight_layout()
    return plt.gcf()




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

    # ===== add mean markers =====
        means = [np.mean(d) for d in data]

        ax.scatter(
            positions,
            means,
            color="tab:blue",
            marker="D",   # diamond
            s=40,
            zorder=3
        )
        
    ax.set_xticks(positions)
    pretty_labels = [get_pretty_label(v) for v in variable_order]
    ax.set_xticklabels(pretty_labels, rotation=35, ha="right")
    ax.set_ylabel("BET-S - Diesel TCO Gap (£)")
    ax.set_title("Independent Impact of each Uncertain Variable on BET-S - Diesel Gap")
    ax.axhline(0, linewidth=1)
    ax.text(
        0.02,
        0.05,
        "Black line in the boxes = median\nBlue diamond = mean",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="none"
        )
    )


    plt.tight_layout()
    return plt.gcf()



def plot_projection_with_uncertainty_by_scenario(summary_df):
    """Plot projection uncertainty as separate figures for each subsidy scenario."""
    if "subsidy_scenario" not in summary_df.columns:
        return plot_projection_with_uncertainty(summary_df)

    for scenario, sub_df in summary_df.groupby("subsidy_scenario"):
        plot_projection_with_uncertainty(
            sub_df.sort_values("year"),
            title_suffix=f"- {scenario}",
        )


def plot_margin_vs_freight_all_in_per_km_by_scenario(summary_df):
    """Plot AEaaS per-km uncertainty as separate figures for each subsidy scenario."""
    if "subsidy_scenario" not in summary_df.columns:
        return plot_margin_vs_freight_all_in_per_km_with_uncertainty(summary_df)

    for scenario, sub_df in summary_df.groupby("subsidy_scenario"):
        plot_margin_vs_freight_all_in_per_km_with_uncertainty(
            sub_df.sort_values("asset_manager_margin"),
            title_suffix=f"- {scenario}",
        )


def plot_margin_vs_gap_by_scenario(summary_df):
    """Plot AEaaS-minus-Diesel gap uncertainty as separate figures for each subsidy scenario."""
    if "subsidy_scenario" not in summary_df.columns:
        return plot_margin_vs_gap_with_uncertainty(summary_df)

    for scenario, sub_df in summary_df.groupby("subsidy_scenario"):
        plot_margin_vs_gap_with_uncertainty(
            sub_df.sort_values("asset_manager_margin"),
            title_suffix=f"- {scenario}",
        )


def plot_monte_carlo_histograms_by_scenario(df):
    """Overlay with-subsidy and no-subsidy MC histograms in one 2x3 figure."""
    if "subsidy_scenario" not in df.columns:
        return plot_monte_carlo_histograms(df)

    histogram_specs = [
        ("diesel_tco", "Diesel discounted TCO", "TCO (£)"),
        ("bet_c_tco", "BET-C discounted TCO", "TCO (£)"),
        ("bet_s_tco", "BET-S discounted TCO", "TCO (£)"),
        ("gap_bet_c_diesel", "BET-C - Diesel", "TCO gap (£)"),
        ("gap_bet_s_diesel", "BET-S - Diesel", "TCO gap (£)"),
        ("gap_bet_s_bet_c", "BET-S - BET-C", "TCO gap (£)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))
    axes = axes.flatten()

    for ax, (col, title, xlabel) in zip(axes, histogram_specs):
        for scenario, sub_df in df.groupby("subsidy_scenario"):
            values = sub_df[col].dropna()
            ax.hist(values, bins=25, alpha=0.42, label=scenario)
            mean_value = values.mean()
            ax.axvline(mean_value, linestyle="--", linewidth=1.4)

            label = "No subsidy" if scenario.lower().startswith("no") else "With subsidy"
            y = 0.88 if label == "No subsidy" else 0.78
            ax.text(
                0.98,
                y,
                f"Mean ({label}) = £{mean_value:,.0f}",
                transform=ax.transAxes,
                fontsize=8,
                ha="right",
                va="top",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    fig.suptitle("Monte Carlo distributions: with subsidy vs no subsidy", fontsize=16, y=1.02)
    fig.tight_layout()
    return fig


def plot_independent_tco_boxplots_by_scenario(df):
    if "subsidy_scenario" not in df.columns:
        return plot_independent_tco_boxplots(df)
    for scenario, sub_df in df.groupby("subsidy_scenario"):
        print(f"Plotting independent TCO boxplots: {scenario}")
        plot_independent_tco_boxplots(sub_df)


def plot_independent_gap_boxplots_by_scenario(df):
    if "subsidy_scenario" not in df.columns:
        return plot_independent_gap_boxplots(df)
    for scenario, sub_df in df.groupby("subsidy_scenario"):
        print(f"Plotting independent gap boxplots: {scenario}")
        plot_independent_gap_boxplots(sub_df)


def plot_independent_bets_vs_diesel_boxplot_by_scenario(df):
    if "subsidy_scenario" not in df.columns:
        return plot_independent_bets_vs_diesel_boxplot(df)
    for scenario, sub_df in df.groupby("subsidy_scenario"):
        print(f"Plotting BET-S vs Diesel independent boxplot: {scenario}")
        plot_independent_bets_vs_diesel_boxplot(sub_df)

# =========================================================
# Compare discounted TCO per km under station utilisation
# 30%, 40%, 50%
# =========================================================

def plot_utilisation_comparison_tco_per_km():
    utilisation_values = [0.30, 0.40, 0.50]

    results_data = {
        "Diesel": [],
        "BET-C": [],
        "BET-S": [],
    }

    for utilisation in utilisation_values:
        shared = SharedInputs()

        diesel_inp = DieselInputs()

        betc_inp = BETCInputs(
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
        )

        bets_inp = BETSInputs(
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio,
            expected_station_utilisation=utilisation
        )

        results = run_model(
            shared=shared,
            diesel_inp=diesel_inp,
            betc_inp=betc_inp,
            bets_inp=bets_inp,
        )

        results_data["Diesel"].append(
            results["diesel"]["tco_per_km_discounted"]
        )

        results_data["BET-C"].append(
            results["bet_c"]["tco_per_km_discounted"]
        )

        results_data["BET-S"].append(
            results["bet_s"]["tco_per_km_discounted"]
        )

    # ================= Plot =================

    x = np.arange(len(utilisation_values))
    width = 0.24

    plt.figure(figsize=(9, 6))

    bars1 = plt.bar(
        x - width,
        results_data["Diesel"],
        width,
        label="Diesel"
    )

    bars2 = plt.bar(
        x,
        results_data["BET-C"],
        width,
        label="BET-C"
    )

    bars3 = plt.bar(
        x + width,
        results_data["BET-S"],
        width,
        label="BET-S"
    )

    plt.xticks(
        x,
        [f"{u:.0%}" for u in utilisation_values]
    )

    plt.xlabel("Expected Station Utilisation")
    plt.ylabel("Discounted TCO per km (£/km)")
    plt.title("Discounted TCO per km under Different Station Utilisation Levels")

    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()

    # value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    return plt.gcf()




# =========================================================
# Compare discounted TCO per km under shift = 1 and 2
# =========================================================

def plot_shift_comparison_tco_per_km():
    shift_values = [1, 2]

    truck_types = ["Diesel", "BET-C", "BET-S"]

    results_data = {
        "Diesel": [],
        "BET-C": [],
        "BET-S": [],
    }

    for shift in shift_values:

        shared = SharedInputs(
            shift_per_day=shift
        )

        diesel_inp = DieselInputs()

        betc_inp = BETCInputs(
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
        )

        bets_inp = BETSInputs(
            battery_recycle_value_ratio=shared.battery_recycle_value_ratio
        )

        results = run_model(
            shared=shared,
            diesel_inp=diesel_inp,
            betc_inp=betc_inp,
            bets_inp=bets_inp,
        )

        results_data["Diesel"].append(
            results["diesel"]["tco_per_km_discounted"]
        )

        results_data["BET-C"].append(
            results["bet_c"]["tco_per_km_discounted"]
        )

        results_data["BET-S"].append(
            results["bet_s"]["tco_per_km_discounted"]
        )

    # ================= Plot =================

    x = np.arange(len(shift_values))
    width = 0.24

    plt.figure(figsize=(9, 6))

    bars1 = plt.bar(
        x - width,
        results_data["Diesel"],
        width,
        label="Diesel"
    )

    bars2 = plt.bar(
        x,
        results_data["BET-C"],
        width,
        label="BET-C"
    )

    bars3 = plt.bar(
        x + width,
        results_data["BET-S"],
        width,
        label="BET-S"
    )

    plt.xticks(x, [f"Shift = {s}" for s in shift_values])

    plt.ylabel("Discounted TCO per km (£/km)")

    plt.title("Discounted TCO per km under Different Shift Levels")

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.legend()

    # value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    return plt.gcf()



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
    bet_c = compute_bet_c(shared, BETCInputs(battery_recycle_value_ratio=shared.battery_recycle_value_ratio))
    bet_s = compute_bet_s(shared, BETSInputs(battery_recycle_value_ratio=shared.battery_recycle_value_ratio))

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
    lines.append(f"BET-C - Diesel: £{bet_c['tco_discounted_recycle'] - diesel['tco_discounted']:,.2f}")
    #lines.append(f"BET-S EO1L - Diesel: £{bet_s['tco_discounted_eol'] - diesel['tco_discounted']:,.2f}")
    lines.append(f"BET-S - Diesel: £{bet_s['tco_discounted_recycle'] - diesel['tco_discounted']:,.2f}")
    lines.append(f"BET-S - BET-C: £{bet_s['tco_discounted_recycle'] - bet_c['tco_discounted_recycle']:,.2f}")
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
    





