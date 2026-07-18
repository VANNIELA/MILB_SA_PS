r"""
Customer-Arrival Scaling Validation Framework for Airport Lounge MILP
=====================================================================

Purpose
-------
This script validates the MILP roster model by scaling CUSTOMER ARRIVALS first,
not by scaling the final staff demand table directly.

Correct validation flow:
    1. Read the 48-slice customer arrival baseline.
    2. Scale expected_arrivals_30min by scenario factor.
    3. Run the 28-day event-driven SimPy operational simulation.
    4. Convert simulated workload into role/shift staffing demand.
    5. Run the MILP roster model using that scenario-specific demand.
    6. Audit feasibility, coverage, role capacity, contract capacity, and output status.

This is more realistic than scaling required staff directly because the workload
is regenerated from operational drivers: arrivals, departures, buffet visits,
table resets, dishes, stock refills, and scheduled food-safety checks.

Required packages
-----------------
pip install pandas numpy openpyxl xlsxwriter simpy pulp

PowerShell example
------------------
python "E:\NĂM 4\Capstone\Validation\milp_customer_arrival_scaling_validation.py" ^
  --sim-script "E:\NĂM 4\Capstone\Data_input\[2] airport_lounge_28day_event_simulation.py" ^
  --model-script "CapstoneE:\NĂM 4\Capstone\MILP_Model\milp_roster_model.py" ^
  --arrival-file "E:\NĂM 4\Capstone\Data_input\arrival_baseline_48_slices.xlsx" ^
  --employee-dir "E:\NĂM 4\Capstone\Data_input\employee_milp_inputs" ^
  --output-dir "E:\NĂM 4\Capstone\Validation\customer_arrival_scaling_outputs" ^
  --scales 0.90 1.00 1.10 1.20 1.30 1.40 1.50 ^
  --demand-source staffing_pivot ^
  --time-limit 300 ^
  --mip-gap 0.02

Notes
-----
- demand-source=staffing_pivot repeats the simulation p95 staffing recommendation
  across the 28-day MILP horizon. This is stable and matches a planning roster.
- demand-source=daily_workload uses day-specific simulated workload demand. This
  is more granular but more stochastic.
- The script intentionally does NOT copy the base staffing_demand_daily.csv into
  scenario folders. Each scenario receives its own generated staffing_demand_daily.csv.
"""

from __future__ import annotations

import argparse
from html import parser
import importlib.util
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd


# -----------------------------
# Constants used by both model stages
# -----------------------------

ROLES = ["RS", "DAS", "BLO", "FSTC", "DS", "SLS"]
SHIFT_ID_TO_SHIFT = {1: "S1", 2: "S2", 3: "S3", "1": "S1", "2": "S2", "3": "S3"}
SHIFT_TIME = {"S1": "00:00-08:00", "S2": "08:00-16:00", "S3": "16:00-24:00"}
SHIFTS = ["S1", "S2", "S3"]
BAD_AVAILABILITY_STATUS = {
    "unavailable", "leave", "paid_leave", "unpaid_leave", "off", "day_off",
    "holiday", "recovery", "vacation", "not_available", "blocked"
}


# -----------------------------
# Generic helpers
# -----------------------------


def import_module_from_path(module_name: str, script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def scenario_name(scale: float) -> str:
    return f"scale_{scale:.2f}".replace(".", "p")


def read_csv_utf8(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def copy_employee_inputs_without_base_demand(src: Path, dst: Path) -> None:
    """Copy employee CSVs but do not copy old/base staffing_demand_daily.csv."""
    if not src.exists():
        raise FileNotFoundError(f"Employee input directory not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for csv_file in src.glob("*.csv"):
        if csv_file.name.lower() == "staffing_demand_daily.csv":
            continue
        shutil.copy2(csv_file, dst / csv_file.name)


def normalize_shift(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text in SHIFT_ID_TO_SHIFT:
        return SHIFT_ID_TO_SHIFT[text]
    if text in SHIFTS:
        return text
    try:
        return SHIFT_ID_TO_SHIFT.get(int(float(text)), text)
    except Exception:
        return text


def safe_number(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


# -----------------------------
# Arrival baseline scaling
# -----------------------------


def load_arrival_baseline(arrival_file: Path) -> pd.DataFrame:
    """Read the 48-slice arrival baseline without using the simulation script's hardcoded path."""
    if not arrival_file.exists():
        raise FileNotFoundError(f"Arrival baseline file not found: {arrival_file}")

    xl = pd.ExcelFile(arrival_file)
    sheet = None
    for candidate in ["demand_shape_48", "generated_arrivals"]:
        if candidate in xl.sheet_names:
            sheet = candidate
            break
    if sheet is None:
        raise ValueError(
            f"Could not find demand_shape_48 or generated_arrivals in {arrival_file}. "
            f"Available sheets: {xl.sheet_names}"
        )

    df = pd.read_excel(arrival_file, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    if "slice_id" not in df.columns or "expected_arrivals_30min" not in df.columns:
        raise ValueError("Arrival baseline must contain slice_id and expected_arrivals_30min.")

    df = df.copy().sort_values("slice_id").reset_index(drop=True)
    if len(df) != 48:
        raise ValueError(f"Expected 48 slices, found {len(df)}.")

    # Reconstruct missing columns so it is compatible with the event simulation functions.
    if "start_minute" not in df.columns:
        df["start_minute"] = (df["slice_id"].astype(int) - 1) * 30
    if "end_minute" not in df.columns:
        df["end_minute"] = df["slice_id"].astype(int) * 30
    if "start_time" not in df.columns:
        df["start_time"] = df["start_minute"].apply(minute_to_clock)
    if "end_time" not in df.columns:
        df["end_time"] = df["end_minute"].apply(minute_to_clock)
    if "shift_id" not in df.columns:
        df["shift_id"] = ((df["slice_id"].astype(int) - 1) // 16) + 1
    if "slice_in_shift" not in df.columns:
        df["slice_in_shift"] = ((df["slice_id"].astype(int) - 1) % 16) + 1
    if "demand_level" not in df.columns:
        df["demand_level"] = "Unknown"
    if "lambda_per_hour" not in df.columns:
        df["lambda_per_hour"] = df["expected_arrivals_30min"] / 0.5
    return df


def minute_to_clock(minute: float) -> str:
    minute_int = int(round(minute)) % (24 * 60)
    return f"{minute_int // 60:02d}:{minute_int % 60:02d}"


def scale_arrival_baseline(base: pd.DataFrame, scale: float) -> pd.DataFrame:
    out = base.copy()
    out["base_expected_arrivals_30min_before_scale"] = pd.to_numeric(
        out["expected_arrivals_30min"], errors="coerce"
    ).fillna(0.0)
    out["customer_arrival_scale"] = scale
    out["expected_arrivals_30min"] = out["base_expected_arrivals_30min_before_scale"] * scale
    out["lambda_per_hour"] = out["expected_arrivals_30min"] / 0.5
    return out


def write_scaled_arrival_baseline(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="demand_shape_48_scaled", index=False)
        summary = pd.DataFrame([
            {"metric": "customer_arrival_scale", "value": df["customer_arrival_scale"].iloc[0]},
            {"metric": "base_expected_daily_customers", "value": df["base_expected_arrivals_30min_before_scale"].sum()},
            {"metric": "scaled_expected_daily_customers", "value": df["expected_arrivals_30min"].sum()},
        ])
        summary.to_excel(writer, sheet_name="scaling_summary", index=False)


# -----------------------------
# Simulation stage
# -----------------------------


def build_sim_config(sim, args: argparse.Namespace, scale: float, sim_output: Path, scenario_index: int):
    """Create simulation Config while keeping default operational assumptions from the user's simulation module."""
    return sim.Config(
        input_file=str(args.arrival_file),
        output_file=str(sim_output),
        days=args.days,
        warmup_days=args.warmup_days,
        start_date=args.start_date,
        holidays=args.holidays,
        base_seed=args.base_seed + scenario_index * args.seed_step,
        weekday_multiplier=args.weekday_multiplier,
        weekend_multiplier=args.weekend_multiplier,
        holiday_multiplier=args.holiday_multiplier,
        daily_cv=args.daily_cv,
        staff_rs=args.staff_rs,
        staff_das=args.staff_das,
        staff_blo=args.staff_blo,
        staff_fstc=args.staff_fstc,
        staff_ds=args.staff_ds,
        staff_sls=args.staff_sls,
        min_staff=args.min_staff,
        target_utilization=args.target_utilization,
        effective_minutes_override=args.effective_minutes,
        kpi_rs=args.kpi_rs,
        kpi_das=args.kpi_das,
        kpi_blo=args.kpi_blo,
        kpi_fstc=args.kpi_fstc,
        kpi_ds=args.kpi_ds,
        kpi_sls=args.kpi_sls,
        lounge_capacity=args.lounge_capacity,
        max_admission_wait=args.max_admission_wait,
        fstc_check_interval=args.fstc_check_interval,
        das_effective_mean=args.das_effective_mean,
        das_use_effective_mean=not args.use_das_tria,
        table_reset_rate=args.table_reset_rate,
        buffet_visit_rate=args.buffet_visit_rate,
        buffet_visit_min=args.buffet_visit_min,
        blo_intervention_rate=args.blo_intervention_rate,
        stock_refill_qty=args.stock_refill_qty,
        stock_unit_per_visit=args.stock_unit_per_visit,
        runout_hours=args.runout_hours,
    )


def run_simulation_scenario(
    sim,
    args: argparse.Namespace,
    scale: float,
    scenario_dir: Path,
    scenario_index: int,
) -> Tuple[Path, Dict[str, float]]:
    """Scale arrivals, run the event simulation, and export scenario simulation workbook."""
    sim_dir = scenario_dir / "simulation"
    sim_dir.mkdir(parents=True, exist_ok=True)
    sim_output = sim_dir / f"Final_data_input_milp_customer_scale_{scale:.2f}.xlsx"

    base_baseline = load_arrival_baseline(args.arrival_file)
    scaled_baseline = scale_arrival_baseline(base_baseline, scale)
    write_scaled_arrival_baseline(sim_dir / f"arrival_baseline_scaled_{scale:.2f}.xlsx", scaled_baseline)

    cfg = build_sim_config(sim, args, scale, sim_output, scenario_index)

    day_plan = sim.create_day_wave_plan(cfg)
    slice_plan, arrival_events = sim.generate_28day_arrivals(scaled_baseline, day_plan, cfg)

    model = sim.LoungeSimulation(cfg, arrival_events)
    logs = model.run()
    outputs = sim.build_summaries(cfg, day_plan, slice_plan, arrival_events, logs)

    # Add scenario traceability sheets before exporting.
    outputs["scenario_metadata"] = pd.DataFrame([
        {"metric": "customer_arrival_scale", "value": scale},
        {"metric": "base_expected_daily_customers", "value": base_baseline["expected_arrivals_30min"].sum()},
        {"metric": "scaled_expected_daily_customers", "value": scaled_baseline["expected_arrivals_30min"].sum()},
        {"metric": "days", "value": args.days},
        {"metric": "warmup_days", "value": args.warmup_days},
        {"metric": "start_date", "value": args.start_date},
        {"metric": "weekday_multiplier", "value": args.weekday_multiplier},
        {"metric": "weekend_multiplier", "value": args.weekend_multiplier},
        {"metric": "holiday_multiplier", "value": args.holiday_multiplier},
        {"metric": "daily_cv", "value": args.daily_cv},
    ])

    exported_path = sim.export_outputs(outputs, str(sim_output))

    customer_daily = outputs.get("customer_daily_summary", pd.DataFrame())
    if customer_daily is not None and not customer_daily.empty:
        measured = customer_daily[customer_daily.get("is_warmup_day", False) == False]
        total_arrivals = float(measured.get("total_arrivals", pd.Series(dtype=float)).sum())
        admitted = float(measured.get("admitted_customers", pd.Series(dtype=float)).sum())
        rejected = float(measured.get("rejected_customers", pd.Series(dtype=float)).sum())
    else:
        total_arrivals = float(len(arrival_events))
        admitted = rejected = math.nan

    metrics = {
        "base_expected_daily_customers": float(base_baseline["expected_arrivals_30min"].sum()),
        "scaled_expected_daily_customers": float(scaled_baseline["expected_arrivals_30min"].sum()),
        "generated_arrival_events_all_days": float(len(arrival_events)),
        "measured_total_arrivals_after_warmup": total_arrivals,
        "measured_admitted_customers_after_warmup": admitted,
        "measured_rejected_customers_after_warmup": rejected,
    }
    return Path(exported_path), metrics


# -----------------------------
# Demand extraction from simulation output
# -----------------------------


def demand_from_staffing_pivot(sim_output: Path, days: int, min_staff: int) -> pd.DataFrame:
    pivot = pd.read_excel(sim_output, sheet_name="staffing_pivot")
    pivot.columns = [str(c).strip() for c in pivot.columns]
    if "shift_id" not in pivot.columns:
        raise ValueError("staffing_pivot sheet must contain shift_id.")
    pivot["shift"] = pivot["shift_id"].map(normalize_shift)

    rows = []
    for d in range(1, days + 1):
        for _, row in pivot.iterrows():
            shift = normalize_shift(row["shift"])
            out = {"day": d, "shift": shift, "shift_time": row.get("shift_time", SHIFT_TIME.get(shift, ""))}
            for role in ROLES:
                value = safe_number(row.get(role, min_staff), min_staff)
                out[role] = max(min_staff, int(math.ceil(value)))
            out["total_staff"] = sum(out[r] for r in ROLES)
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["day", "shift"]).reset_index(drop=True)


def demand_from_daily_workload(sim_output: Path, days: int, min_staff: int) -> pd.DataFrame:
    raw = pd.read_excel(sim_output, sheet_name="role_daily_shift_workload")
    raw.columns = [str(c).strip() for c in raw.columns]
    if raw.empty:
        raise ValueError("role_daily_shift_workload is empty; cannot build demand.")

    required_col = "recommended_staff_initial" if "recommended_staff_initial" in raw.columns else "required_staff_by_workload"
    if required_col not in raw.columns:
        raise ValueError(
            "role_daily_shift_workload must contain required_staff_by_workload or recommended_staff_initial."
        )
    raw = raw.copy()
    raw["day"] = pd.to_numeric(raw["day_index"], errors="coerce").astype("Int64")
    raw["shift"] = raw["shift_id"].map(normalize_shift)
    raw["role"] = raw["role"].astype(str).str.strip()
    raw = raw[(raw["day"].notna()) & (raw["day"].astype(int).between(1, days)) & raw["role"].isin(ROLES)]

    pivot = raw.pivot_table(
        index=["day", "shift"],
        columns="role",
        values=required_col,
        aggfunc="max",
        fill_value=0,
    ).reset_index()

    # Make a complete day x shift grid and enforce minimum staffing.
    complete = pd.MultiIndex.from_product([range(1, days + 1), SHIFTS], names=["day", "shift"]).to_frame(index=False)
    pivot = complete.merge(pivot, on=["day", "shift"], how="left")
    for role in ROLES:
        if role not in pivot.columns:
            pivot[role] = 0
        pivot[role] = pd.to_numeric(pivot[role], errors="coerce").fillna(0).apply(
            lambda v: max(min_staff, int(math.ceil(float(v))))
        )
    pivot["shift_time"] = pivot["shift"].map(SHIFT_TIME)
    pivot["total_staff"] = pivot[ROLES].sum(axis=1)
    return pivot[["day", "shift", "shift_time"] + ROLES + ["total_staff"]].sort_values(["day", "shift"]).reset_index(drop=True)


def extract_scenario_demand(sim_output: Path, days: int, min_staff: int, demand_source: str) -> pd.DataFrame:
    xl = pd.ExcelFile(sim_output)
    if demand_source == "daily_workload":
        if "role_daily_shift_workload" not in xl.sheet_names:
            raise ValueError("Simulation output does not contain role_daily_shift_workload.")
        return demand_from_daily_workload(sim_output, days, min_staff)
    if demand_source == "staffing_pivot":
        if "staffing_pivot" not in xl.sheet_names:
            raise ValueError("Simulation output does not contain staffing_pivot.")
        return demand_from_staffing_pivot(sim_output, days, min_staff)
    raise ValueError(f"Unknown demand_source: {demand_source}")


# -----------------------------
# Feasibility prechecks before MILP
# -----------------------------


def load_employee_tables(employee_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "master": "employee_master.csv",
        "skills": "employee_skills.csv",
        "availability": "employee_availability.csv",
        "preferences": "employee_preferences.csv",
        "history": "employee_history.csv",
        "incompatibility": "employee_incompatibility.csv",
        "standard_role_shift_preferences": "standard_role_shift_preferences.csv",
    }
    out = {}
    for key, filename in files.items():
        path = employee_dir / filename
        out[key] = read_csv_utf8(path) if path.exists() else pd.DataFrame()
    return out


def unavailable_pairs(availability: pd.DataFrame) -> set[Tuple[str, int]]:
    if availability is None or availability.empty:
        return set()
    df = availability.copy()
    df["employee_id"] = df["employee_id"].astype(str)
    df["day"] = pd.to_numeric(df["day"], errors="coerce")
    df["status"] = df["status"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    pairs = set()
    for _, row in df.dropna(subset=["day"]).iterrows():
        if str(row["status"]) in BAD_AVAILABILITY_STATUS:
            pairs.add((str(row["employee_id"]), int(row["day"])))
    return pairs


def precheck_capacity(employee_dir: Path, demand_df: pd.DataFrame, days: int) -> Dict[str, pd.DataFrame]:
    tables = load_employee_tables(employee_dir)
    master = tables["master"].copy()
    skills = tables["skills"].copy()
    availability = tables["availability"].copy()

    if master.empty or skills.empty:
        raise ValueError("employee_master.csv and employee_skills.csv are required for precheck.")

    master["employee_id"] = master["employee_id"].astype(str)
    skills["employee_id"] = skills["employee_id"].astype(str)
    for c in ["min_work_days", "max_work_days", "min_hours", "max_hours", "max_night_shifts"]:
        if c not in master.columns:
            master[c] = 0
        master[c] = pd.to_numeric(master[c], errors="coerce").fillna(0)
    for r in ROLES:
        if r not in skills.columns:
            skills[r] = 0
        skills[r] = pd.to_numeric(skills[r], errors="coerce").fillna(0).astype(int)

    unavail = unavailable_pairs(availability)
    employees = master["employee_id"].tolist()
    available_days_by_e = {
        e: sum(1 for d in range(1, days + 1) if (e, d) not in unavail)
        for e in employees
    }

    total_slots = int(demand_df[ROLES].sum().sum())
    total_min_days = int(master["min_work_days"].sum())
    total_max_days_policy = int(master["max_work_days"].sum())
    total_max_days_avail_adjusted = int(sum(
        min(float(row["max_work_days"]), available_days_by_e.get(str(row["employee_id"]), days))
        for _, row in master.iterrows()
    ))
    total_min_hours = int(master["min_hours"].sum())
    total_max_hours_policy = int(master["max_hours"].sum())
    total_night_slots = int(demand_df.loc[demand_df["shift"].astype(str).eq("S1"), ROLES].sum().sum())
    total_night_capacity = int(master["max_night_shifts"].sum())

    total_available_employee_days = sum(available_days_by_e.values())
    precheck_summary = pd.DataFrame([
        {"check": "total_slots_vs_min_contract_days", "lhs": total_slots, "operator": ">=", "rhs": total_min_days, "pass": total_slots >= total_min_days,
         "interpretation": "If total demand is below total minimum workdays, employees cannot all reach minimum days."},
        {"check": "total_slots_vs_max_contract_days_policy", "lhs": total_slots, "operator": "<=", "rhs": total_max_days_policy, "pass": total_slots <= total_max_days_policy,
         "interpretation": "If total demand exceeds total maximum workdays, there is not enough monthly labor capacity."},
        {"check": "total_slots_vs_max_contract_days_availability_adjusted", "lhs": total_slots, "operator": "<=", "rhs": total_max_days_avail_adjusted, "pass": total_slots <= total_max_days_avail_adjusted,
         "interpretation": "Same as above, but reduced by fixed leave/unavailable days."},
        {"check": "total_slots_vs_available_employee_days", "lhs": total_slots, "operator": "<=", "rhs": total_available_employee_days, "pass": total_slots <= total_available_employee_days,
         "interpretation": "One employee can work at most one shift per day."},
        {"check": "total_hours_vs_min_contract_hours", "lhs": total_slots * 8, "operator": ">=", "rhs": total_min_hours, "pass": total_slots * 8 >= total_min_hours,
         "interpretation": "If demand hours are below minimum contract hours, the model may need overstaffing or become infeasible."},
        {"check": "total_hours_vs_max_contract_hours", "lhs": total_slots * 8, "operator": "<=", "rhs": total_max_hours_policy, "pass": total_slots * 8 <= total_max_hours_policy,
         "interpretation": "If demand hours exceed maximum contract hours, infeasibility is expected."},
        {"check": "night_slots_vs_night_capacity", "lhs": total_night_slots, "operator": "<=", "rhs": total_night_capacity, "pass": total_night_slots <= total_night_capacity,
         "interpretation": "If S1 demand exceeds monthly max night-shift capacity, infeasibility is expected."},
    ])

    role_rows = []
    for role in ROLES:
        qualified = int(skills[role].sum())
        role_slots = int(demand_df[role].sum())
        max_capacity = int(sum(
            min(float(master.loc[master["employee_id"] == e, "max_work_days"].iloc[0]), available_days_by_e.get(e, days))
            for e in skills.loc[skills[role] == 1, "employee_id"].astype(str)
        ))
        peak_shift = int(demand_df[role].max())
        role_rows.append({
            "role": role,
            "qualified_employees": qualified,
            "role_required_slots": role_slots,
            "qualified_max_day_capacity_availability_adjusted": max_capacity,
            "peak_single_shift_requirement": peak_shift,
            "pass_role_monthly_capacity": role_slots <= max_capacity,
            "pass_peak_shift_pool": peak_shift <= qualified,
        })
    role_capacity = pd.DataFrame(role_rows)

    daily_rows = []
    skill_index = skills.set_index("employee_id")
    for d in range(1, days + 1):
        available_employees = [e for e in employees if (e, d) not in unavail]
        total_day_required = int(demand_df.loc[demand_df["day"] == d, ROLES].sum().sum())
        daily_rows.append({
            "day": d,
            "role": "ALL",
            "required_role_slots_day": total_day_required,
            "available_qualified_employees": len(available_employees),
            "pass_daily_capacity": total_day_required <= len(available_employees),
        })
        for role in ROLES:
            required_role_day = int(demand_df.loc[demand_df["day"] == d, role].sum())
            peak_shift_req = int(demand_df.loc[demand_df["day"] == d, role].max())
            available_qualified = sum(
                1 for e in available_employees
                if e in skill_index.index and int(skill_index.loc[e, role]) == 1
            )
            daily_rows.append({
                "day": d,
                "role": role,
                "required_role_slots_day": required_role_day,
                "peak_shift_requirement": peak_shift_req,
                "available_qualified_employees": available_qualified,
                "pass_daily_capacity": required_role_day <= available_qualified,
                "pass_peak_shift_pool": peak_shift_req <= available_qualified,
            })
    daily_availability = pd.DataFrame(daily_rows)

    employee_contract_issues = []
    for _, row in master.iterrows():
        e = str(row["employee_id"])
        avail_days = available_days_by_e.get(e, days)
        if avail_days < float(row["min_work_days"]):
            employee_contract_issues.append({
                "employee_id": e,
                "available_days": avail_days,
                "min_work_days": row["min_work_days"],
                "issue": "available days below min_work_days",
            })
    employee_contract_issues = pd.DataFrame(employee_contract_issues)

    return {
        "precheck_summary": precheck_summary,
        "role_capacity": role_capacity,
        "daily_availability": daily_availability,
        "employee_contract_issues": employee_contract_issues,
    }



# -----------------------------
# Convert scenario demand to coverage bands for the current MILP script
# -----------------------------


def write_coverage_bands_long_from_demand(
    demand_df: pd.DataFrame,
    scenario_employee_dir: Path,
    min_staff: int = 1,
    cmin_ratio: float = 1.00,
    cmax_ratio: float = 1.50,
    cmax_buffer: int = 2,
) -> Path:
    """
    The current milp_roster_model.py reads staffing_coverage_bands_long.csv,
    not only staffing_demand_daily.csv. This function converts the scenario
    demand table produced by DES into the long C_min/C_opt/C_max format needed
    by the MILP model.

    Default validation logic:
      C_opt = scenario required staff from DES
      C_min = C_opt by default, so the simulation-based requirement is protected
      C_max = wider soft/hard tolerance so low-scale cases can still satisfy
              employee minimum-workday rules without artificial infeasibility
    """
    rows = []
    df = demand_df.copy()
    df["shift"] = df["shift"].map(normalize_shift)
    if "shift_time" not in df.columns:
        df["shift_time"] = df["shift"].map(SHIFT_TIME)

    for _, row in df.iterrows():
        d = int(row["day"])
        s = normalize_shift(row["shift"])
        st = row.get("shift_time", SHIFT_TIME.get(s, ""))
        for role in ROLES:
            copt = max(min_staff, int(math.ceil(safe_number(row.get(role, min_staff), min_staff))))
            cmin = max(min_staff, int(math.ceil(copt * cmin_ratio)))
            cmax = max(copt, int(math.ceil(copt * cmax_ratio)) + cmax_buffer)
            rows.append({
                "day": d,
                "shift": s,
                "shift_time": st,
                "role": role,
                "C_min": cmin,
                "C_opt": copt,
                "C_max": cmax,
            })

    out = pd.DataFrame(rows).sort_values(["day", "shift", "role"])
    out_path = scenario_employee_dir / "staffing_coverage_bands_long.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path

# -----------------------------
# MILP stage
# -----------------------------


def run_milp_scenario(
    milp,
    scenario_employee_dir: Path,
    demand_df: pd.DataFrame,
    milp_output_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Optional[Dict[str, pd.DataFrame]]]:
    """
    Run MILP using the CURRENT milp_roster_model.py interface.

    This replaces the older interface that expected functions named:
    load_employee_inputs(), build_parameter_dicts(), build_and_solve_model(),
    extract_solution(), and write_outputs().

    The current MILP script uses:
    load_inputs(), build_model(), solve_model(), build_outputs(), export_outputs().
    """
    milp_output_dir.mkdir(parents=True, exist_ok=True)

    # The current MILP model reads coverage bands from this file.
    coverage_path = write_coverage_bands_long_from_demand(
        demand_df=demand_df,
        scenario_employee_dir=scenario_employee_dir,
        min_staff=args.min_staff,
        cmin_ratio=1.00,
        cmax_ratio=1.50,
        cmax_buffer=2,
    )

    status_row = {
        "solver_status": "not_started",
        "objective_value": None,
        "total_required_assignments": int(demand_df[ROLES].sum().sum()),
        "actual_assignments": None,
        "total_shortage": None,
        "total_overstaff": None,
        "milp_output_workbook": "",
        "coverage_bands_used": str(coverage_path),
    }

    required_api = ["load_inputs", "build_model", "solve_model", "build_outputs", "export_outputs"]
    missing = [name for name in required_api if not hasattr(milp, name)]
    if missing:
        raise AttributeError(
            "The selected MILP script does not match this validation runner. "
            f"Missing functions: {missing}. Use the current milp_roster_model.py or update the runner."
        )

    data = milp.load_inputs(scenario_employee_dir)
    prob, x, meta = milp.build_model(
        data=data,
        coverage_lower="C_opt",
        cmax_hard=False,          # validation should diagnose excess staffing, not fail early on Cmax
        fairness_weight=1.0,
        soft_start_history=True,  # avoids Day-1 generated-history anomaly
        soft_coverage=True,       # shortage becomes reported penalty, not solver failure
    )
    solve_info = milp.solve_model(
        prob,
        time_limit=args.time_limit,
        gap=args.mip_gap,
        msg=not args.quiet,
    )

    status = solve_info.get("solver_status", "Unknown")
    objective = solve_info.get("objective_value")
    status_row["solver_status"] = status
    status_row["objective_value"] = objective

    if status not in {"Optimal", "Feasible"}:
        pd.DataFrame([status_row]).to_csv(milp_output_dir / "scenario_status.csv", index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(milp_output_dir / "infeasible_or_unsolved_diagnostic.xlsx", engine="openpyxl") as writer:
            pd.DataFrame([status_row]).to_excel(writer, sheet_name="solver_status", index=False)
            demand_df.to_excel(writer, sheet_name="scenario_demand", index=False)
            data["coverage"].to_excel(writer, sheet_name="coverage_bands", index=False)
        return status_row, None

    dfs = milp.build_outputs(data, x, meta, solve_info)
    workbook = milp.export_outputs(dfs, milp_output_dir)

    coverage = dfs.get("coverage_summary", pd.DataFrame())
    assignment_long = dfs.get("assignment_long", pd.DataFrame())
    if not coverage.empty:
        shortage = int(coverage.get("shortage_to_Copt", pd.Series(dtype=float)).sum())
        overstaff = int(coverage.get("overstaff_above_Copt", pd.Series(dtype=float)).sum())
    else:
        shortage = None
        overstaff = None

    status_row.update({
        "actual_assignments": int(len(assignment_long)) if assignment_long is not None else None,
        "total_shortage": shortage,
        "total_overstaff": overstaff,
        "milp_output_workbook": str(workbook),
    })
    pd.DataFrame([status_row]).to_csv(milp_output_dir / "scenario_status.csv", index=False, encoding="utf-8-sig")
    return status_row, dfs


# -----------------------------
# Output helpers
# -----------------------------


def write_scenario_precheck(scenario_dir: Path, prechecks: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(scenario_dir / "scenario_precheck.xlsx", engine="openpyxl") as writer:
        for name, df in prechecks.items():
            sheet = name[:31]
            if df is None or df.empty:
                pd.DataFrame({"message": ["No records"]}).to_excel(writer, sheet_name=sheet, index=False)
            else:
                df.to_excel(writer, sheet_name=sheet, index=False)


def write_final_summary(output_dir: Path, scenario_rows: List[Dict[str, object]], all_precheck_rows: List[pd.DataFrame], all_role_rows: List[pd.DataFrame], all_daily_rows: List[pd.DataFrame]) -> None:
    summary = pd.DataFrame(scenario_rows)
    summary_csv = output_dir / "customer_arrival_scaling_validation_summary.csv"
    summary_xlsx = output_dir / "customer_arrival_scaling_validation_summary.xlsx"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="scenario_summary", index=False)
        if all_precheck_rows:
            pd.concat(all_precheck_rows, ignore_index=True).to_excel(writer, sheet_name="precheck_summary", index=False)
        if all_role_rows:
            pd.concat(all_role_rows, ignore_index=True).to_excel(writer, sheet_name="role_capacity", index=False)
        if all_daily_rows:
            daily_all = pd.concat(all_daily_rows, ignore_index=True)
            daily_all.to_excel(writer, sheet_name="daily_availability", index=False)
            failures = daily_all[(daily_all.get("pass_daily_capacity") == False) | (daily_all.get("pass_peak_shift_pool") == False)]
            if not failures.empty:
                failures.to_excel(writer, sheet_name="daily_failures_only", index=False)


def bool_all(series: pd.Series) -> bool:
    if series is None or len(series) == 0:
        return True
    return bool(series.fillna(True).astype(bool).all())


# -----------------------------
# Main orchestration
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MILP by scaling customer arrivals, rerunning simulation, then solving MILP.")

  

    parser.add_argument(
    "--sim-script",
    type=Path,
    default=Path(r"E:\NĂM 4\Capstone\Data_input\[2] airport_lounge_28day_event_simulation.py"),
    help="Path to [2] airport_lounge_28day_event_simulation.py"
    )

    parser.add_argument(
    "--model-script",
    type=Path,
    default=Path(r"E:\NĂM 4\Capstone\MILP_Model\milp_roster_model.py"),
    help="Path to milp_roster_model.py"
    )

    parser.add_argument(
    "--arrival-file",
    type=Path,
    default=Path(r"E:\NĂM 4\Capstone\Data_input\arrival_baseline_48_slices.xlsx"),
    help="Path to arrival_baseline_48_slices.xlsx"
    )

    parser.add_argument(
    "--employee-dir",
    type=Path,
    default=Path(r"E:\NĂM 4\Capstone\Data_input\employee_milp_inputs"),
    help="Directory containing employee MILP input CSV files"
    )

    parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(r"E:\NĂM 4\Capstone\Validation\customer_arrival_scaling_outputs"),
    help="Output directory for scenario validation results"
    )
    parser.add_argument("--scales", type=float, nargs="+", default=[0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50])
    parser.add_argument("--demand-source", choices=["staffing_pivot", "daily_workload"], default="staffing_pivot")

    # Simulation controls.
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--warmup-days", type=int, default=1)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--holidays", default="")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed-step", type=int, default=0, help="0 uses same random seed structure for all scales; positive values vary seeds by scenario.")
    parser.add_argument("--weekday-multiplier", type=float, default=1.00)
    parser.add_argument("--weekend-multiplier", type=float, default=1.15)
    parser.add_argument("--holiday-multiplier", type=float, default=1.35)
    parser.add_argument("--daily-cv", type=float, default=0.05)

    # Simulation staffing/resource assumptions.
    parser.add_argument("--staff-rs", type=int, default=2)
    parser.add_argument("--staff-das", type=int, default=10)
    parser.add_argument("--staff-blo", type=int, default=6)
    parser.add_argument("--staff-fstc", type=int, default=2)
    parser.add_argument("--staff-ds", type=int, default=2)
    parser.add_argument("--staff-sls", type=int, default=2)
    parser.add_argument("--min-staff", type=int, default=1)
    parser.add_argument("--target-utilization", type=float, default=0.85)
    parser.add_argument("--effective-minutes", type=float, default=None)

    # KPI and operational behavior assumptions.
    parser.add_argument("--kpi-rs", type=float, default=5.0)
    parser.add_argument("--kpi-das", type=float, default=4.0)
    parser.add_argument("--kpi-blo", type=float, default=6.0)
    parser.add_argument("--kpi-fstc", type=float, default=5.0)
    parser.add_argument("--kpi-ds", type=float, default=10.0)
    parser.add_argument("--kpi-sls", type=float, default=5.0)
    parser.add_argument("--lounge-capacity", type=int, default=200)
    parser.add_argument("--max-admission-wait", type=float, default=30.0)
    parser.add_argument("--fstc-check-interval", type=float, default=60.0)
    parser.add_argument("--das-effective-mean", type=float, default=7.1)
    parser.add_argument("--use-das-tria", action="store_true")
    parser.add_argument("--table-reset-rate", type=float, default=0.65)
    parser.add_argument("--buffet-visit-rate", type=float, default=1.5)
    parser.add_argument("--buffet-visit-min", type=int, default=0)
    parser.add_argument("--blo-intervention-rate", type=float, default=0.20)
    parser.add_argument("--stock-refill-qty", type=float, default=220.0)
    parser.add_argument("--stock-unit-per-visit", type=float, default=1.0)
    parser.add_argument("--runout-hours", type=float, default=4.0)

    # MILP solver controls.
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--mip-gap", type=float, default=0.02)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[0] Importing simulation and MILP modules...")
    sim = import_module_from_path("airport_lounge_event_simulation", args.sim_script)
    milp = import_module_from_path("milp_roster_model", args.model_script)

    scenario_rows: List[Dict[str, object]] = []
    all_precheck_rows: List[pd.DataFrame] = []
    all_role_rows: List[pd.DataFrame] = []
    all_daily_rows: List[pd.DataFrame] = []

    for idx, scale in enumerate(args.scales, start=1):
        name = scenario_name(scale)
        scenario_dir = args.output_dir / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_employee_dir = scenario_dir / "employee_inputs"
        milp_output_dir = scenario_dir / "milp_solution"

        print(f"\n========== Scenario {name}: customer arrival scale = {scale:.2f} ==========")
        row: Dict[str, object] = {
            "scenario": name,
            "customer_arrival_scale": scale,
            "demand_source": args.demand_source,
            "simulation_status": "not_started",
            "milp_solver_status": "not_started",
        }

        try:
            copy_employee_inputs_without_base_demand(args.employee_dir, scenario_employee_dir)

            print("[1/5] Running customer-arrival-scaled simulation...")
            sim_output, sim_metrics = run_simulation_scenario(sim, args, scale, scenario_dir, idx - 1)
            row.update(sim_metrics)
            row["simulation_status"] = "OK"
            row["simulation_output"] = str(sim_output)

            print("[2/5] Extracting scenario-specific staffing demand from simulation output...")
            demand_df = extract_scenario_demand(
                sim_output=sim_output,
                days=args.days,
                min_staff=args.min_staff,
                demand_source=args.demand_source,
            )
            demand_df["customer_arrival_scale"] = scale
            demand_csv = scenario_employee_dir / "staffing_demand_daily.csv"
            demand_df.to_csv(demand_csv, index=False, encoding="utf-8-sig")
            demand_df.to_excel(scenario_dir / "scenario_staffing_demand_daily.xlsx", index=False)
            row["total_required_assignments_from_sim"] = int(demand_df[ROLES].sum().sum())

            print("[3/5] Running pre-MILP feasibility checks...")
            prechecks = precheck_capacity(scenario_employee_dir, demand_df, args.days)
            write_scenario_precheck(scenario_dir, prechecks)
            pc = prechecks["precheck_summary"].copy()
            rc = prechecks["role_capacity"].copy()
            da = prechecks["daily_availability"].copy()
            for df in [pc, rc, da]:
                df.insert(0, "scenario", name)
                df.insert(1, "customer_arrival_scale", scale)
            all_precheck_rows.append(pc)
            all_role_rows.append(rc)
            all_daily_rows.append(da)
            row["precheck_pass_all_summary"] = bool_all(pc["pass"])
            row["precheck_pass_role_capacity"] = bool_all(rc["pass_role_monthly_capacity"]) and bool_all(rc["pass_peak_shift_pool"])
            row["precheck_pass_daily_availability"] = bool_all(da["pass_daily_capacity"]) and bool_all(da.get("pass_peak_shift_pool", pd.Series([True])))

            print("[4/5] Solving MILP for this scenario...")
            milp_status, dfs = run_milp_scenario(milp, scenario_employee_dir, demand_df, milp_output_dir, args)
            row["milp_solver_status"] = milp_status.get("solver_status")
            row["milp_objective_value"] = milp_status.get("objective_value")
            row["milp_actual_assignments"] = milp_status.get("actual_assignments")
            row["milp_total_shortage"] = milp_status.get("total_shortage")
            row["milp_total_overstaff"] = milp_status.get("total_overstaff")
            row["milp_output_workbook"] = milp_status.get("milp_output_workbook", "")

            print("[5/5] Scenario done.")

        except Exception as exc:
            row["error"] = str(exc)
            row["traceback"] = traceback.format_exc()
            print("ERROR in scenario", name)
            print(row["traceback"])
            pd.DataFrame([row]).to_csv(scenario_dir / "scenario_error.csv", index=False, encoding="utf-8-sig")

        scenario_rows.append(row)
        write_final_summary(args.output_dir, scenario_rows, all_precheck_rows, all_role_rows, all_daily_rows)

    print("\nDONE")
    print(f"Summary CSV : {args.output_dir / 'customer_arrival_scaling_validation_summary.csv'}")
    print(f"Summary XLSX: {args.output_dir / 'customer_arrival_scaling_validation_summary.xlsx'}")


if __name__ == "__main__":
    main()
