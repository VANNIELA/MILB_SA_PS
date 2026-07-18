r"""
MILP roster model for the SASCO airport lounge capstone.

Default input folder for MEDIUM scale:
    E:\NĂM 4\Capstone\SA_Improvement\employee_milp_inputs_medium
Default output folder for MEDIUM scale:
    E:\NĂM 4\Capstone\SA_Improvement\MILP_Result_medium

Install dependencies:
    python -m pip install pandas openpyxl pulp

Run:
    python milp_roster_model_medium.py

Optional examples:
    python milp_roster_model_medium.py --coverage-lower C_opt --cmax-hard
    python milp_roster_model_medium.py --coverage-lower C_min --soft-cmax
    python milp_roster_model_medium.py --soft-start-history
    python milp_roster_model_medium.py --soft-coverage
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

import pandas as pd

try:
    import pulp
except ImportError as exc:
    raise SystemExit(
        "PuLP is required for this MILP script. Install it with:\n"
        "    python -m pip install pulp\n"
        "Also install Excel support if needed:\n"
        "    python -m pip install pandas openpyxl"
    ) from exc


# ============================================================
# 1. DEFAULT PATHS
# ============================================================
DEFAULT_INPUT_DIR = Path(r"E:\NĂM 4\Capstone\SA_Improvement\employee_milp_inputs_medium")
DEFAULT_OUTPUT_DIR = Path(r"E:\NĂM 4\Capstone\SA_Improvement\MILP_Result_medium")

ROLE_COLUMNS = ["RS", "DAS", "BLO", "FSTC", "DS", "SLS"]
HORIZON_DAYS = 56  # MEDIUM scale default; actual days are read from coverage input if available


# ============================================================
# 2. SMALL UTILITIES
# ============================================================
def read_csv_required(input_dir: Path, filename: str) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    return pd.read_csv(path)


def read_csv_optional(input_dir: Path, filename: str, columns: List[str]) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df[columns]


def clean_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value))


def as_int(value: Any, default: int = 0) -> int:
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def lp_sum(items: Iterable[Any]) -> Any:
    items = list(items)
    return pulp.lpSum(items) if items else 0


# ============================================================
# 3. LOAD AND STANDARDIZE INPUT DATA
# ============================================================
def load_inputs(input_dir: Path) -> Dict[str, pd.DataFrame]:
    data = {
        "employees": read_csv_required(input_dir, "employee_master.csv"),
        "skills": read_csv_required(input_dir, "employee_skills.csv"),
        "coverage": read_csv_required(input_dir, "staffing_coverage_bands_long.csv"),
        "shifts": read_csv_required(input_dir, "shift_structure.csv"),
        "rest": read_csv_required(input_dir, "shift_transition_rest.csv"),
        "penalties": read_csv_required(input_dir, "penalty_config.csv"),
        "availability": read_csv_optional(input_dir, "employee_availability.csv", ["employee_id", "day", "status"]),
        "day_requests": read_csv_optional(input_dir, "employee_day_requests.csv", ["employee_id", "day", "request_type", "penalty", "is_fixed"]),
        "shift_preferences": read_csv_optional(input_dir, "employee_preferences.csv", ["employee_id", "day", "shift", "preference_type", "penalty"]),
        "assignment_preferences": read_csv_optional(input_dir, "employee_assignment_preferences.csv", ["employee_id", "day", "shift", "role", "preference_type", "penalty"]),
        "fixed_assignments": read_csv_optional(input_dir, "employee_fixed_assignments.csv", ["employee_id", "day", "shift", "role"]),
        "history": read_csv_optional(input_dir, "employee_history.csv", [
            "employee_id", "previous_last_shift", "previous_last_status",
            "previous_consecutive_work_days", "previous_consecutive_days_off",
            "previous_consecutive_night_shifts",
        ]),
        "standard_profile": read_csv_optional(input_dir, "employee_standard_profile.csv", ["employee_id", "standard_role", "standard_shift"]),
        "incompatibility": read_csv_optional(input_dir, "employee_incompatibility.csv", ["employee_i", "employee_j", "incompatibility_type", "priority", "penalty"]),
        "pairing": read_csv_optional(input_dir, "employee_pairing.csv", ["employee_i", "employee_j", "pair_type", "priority", "penalty"]),
        "holiday": read_csv_optional(input_dir, "holiday_calendar.csv", ["day", "weekday_index", "is_saturday", "is_sunday", "is_weekend", "is_public_holiday"]),
        "weekend_policy": read_csv_optional(input_dir, "weekend_policy.csv", ["policy_name", "max_working_weekends", "complete_weekend_required", "saturday_sunday_should_match"]),
        "role_shift_pref": read_csv_optional(input_dir, "standard_role_shift_preferences.csv", ["role", "shift", "preference_type", "penalty", "note"]),
    }

    # Standard column types.
    for key in ["coverage", "availability", "day_requests", "shift_preferences", "assignment_preferences", "fixed_assignments", "holiday"]:
        if "day" in data[key].columns and not data[key].empty:
            data[key]["day"] = data[key]["day"].apply(as_int)

    # Keep only known roles in coverage.
    data["coverage"] = data["coverage"][data["coverage"]["role"].isin(ROLE_COLUMNS)].copy()

    return data


def penalty_weights(penalty_df: pd.DataFrame) -> Dict[str, float]:
    if penalty_df.empty:
        return {}
    return {
        str(row["penalty_component"]): as_float(row["weight"], 0.0)
        for _, row in penalty_df.iterrows()
    }


# ============================================================
# 4. BUILD MILP MODEL
# ============================================================
def build_model(
    data: Dict[str, pd.DataFrame],
    coverage_lower: str = "C_opt",
    cmax_hard: bool = True,
    fairness_weight: float = 1.0,
    soft_start_history: bool = True,
    soft_coverage: bool = True,
) -> Tuple[pulp.LpProblem, Dict[Tuple[str, int, str, str], pulp.LpVariable], Dict[str, Any]]:
    if coverage_lower not in {"C_min", "C_opt"}:
        raise ValueError("coverage_lower must be 'C_min' or 'C_opt'")

    employees_df = data["employees"].copy()
    skills_df = data["skills"].copy()
    coverage_df = data["coverage"].copy()
    shifts_df = data["shifts"].copy()
    rest_df = data["rest"].copy()
    weights = penalty_weights(data["penalties"])

    employees = employees_df["employee_id"].astype(str).tolist()
    days = sorted(coverage_df["day"].unique().tolist())
    if not days:
        days = list(range(1, HORIZON_DAYS + 1))
    shifts = shifts_df["shift"].astype(str).tolist()
    roles = ROLE_COLUMNS

    shift_time = dict(zip(shifts_df["shift"].astype(str), shifts_df["shift_time"].astype(str)))
    shift_hours = dict(zip(shifts_df["shift"].astype(str), shifts_df["duration_hours"].apply(as_float)))
    is_night_shift = dict(zip(shifts_df["shift"].astype(str), shifts_df["is_night_shift"].apply(as_int)))
    night_shifts = [s for s in shifts if is_night_shift.get(s, 0) == 1]

    # Employee dictionaries.
    emp_row = employees_df.set_index("employee_id").to_dict("index")
    skill_row = skills_df.set_index("employee_id").to_dict("index")

    # Coverage dictionaries.
    cov_key = coverage_df.set_index(["day", "shift", "role"]).to_dict("index")
    def cov_value(d: int, s: str, r: str, col: str) -> int:
        return as_int(cov_key.get((d, s, r), {}).get(col, 0), 0)

    # Availability: any listed non-working status blocks assignment.
    blocked_status = set()
    for _, row in data["availability"].iterrows():
        status = str(row.get("status", "")).strip().lower()
        if status and status not in {"available", "work", "working", "ok"}:
            blocked_status.add((str(row["employee_id"]), as_int(row["day"])))

    # Fixed day-off requests also block assignment.
    for _, row in data["day_requests"].iterrows():
        if as_int(row.get("is_fixed", 0)) == 1 and str(row.get("request_type", "")).lower() in {
            "desired_day_off", "day_off", "leave", "fixed_day_off"
        }:
            blocked_status.add((str(row["employee_id"]), as_int(row["day"])))

    # Create feasible assignment variables only.
    prob = pulp.LpProblem("SASCO_Airport_Lounge_MILP_Rostering", pulp.LpMinimize)
    x: Dict[Tuple[str, int, str, str], pulp.LpVariable] = {}

    for e in employees:
        for d in days:
            if (e, d) in blocked_status:
                continue
            for s in shifts:
                for r in roles:
                    if as_int(skill_row.get(e, {}).get(r, 0)) == 1:
                        x[(e, d, s, r)] = pulp.LpVariable(
                            f"x_{clean_id(e)}_D{d}_{clean_id(s)}_{clean_id(r)}",
                            lowBound=0,
                            upBound=1,
                            cat="Binary",
                        )

    def assign_expr(e: str, d: int, s: str, r: str) -> Any:
        return x.get((e, d, s, r), 0)

    def shift_expr(e: str, d: int, s: str) -> Any:
        return lp_sum(assign_expr(e, d, s, r) for r in roles)

    def work_expr(e: str, d: int) -> Any:
        return lp_sum(assign_expr(e, d, s, r) for s in shifts for r in roles)

    def cov_expr(d: int, s: str, r: str) -> Any:
        return lp_sum(assign_expr(e, d, s, r) for e in employees)

    objective_terms = []
    meta: Dict[str, Any] = {
        "coverage_over_opt": {},
        "coverage_over_max": {},
        "target_shortage": {},
        "soft_vars": [],
        "shift_time": shift_time,
        "shift_hours": shift_hours,
        "is_night_shift": is_night_shift,
        "coverage_lower": coverage_lower,
        "cmax_hard": cmax_hard,
        "soft_start_history": soft_start_history,
        "soft_coverage": soft_coverage,
        "coverage_short_min": {},
        "coverage_short_lower": {},
        "employees": employees,
        "days": days,
        "shifts": shifts,
        "roles": roles,
        "blocked_status": blocked_status,
    }

    # ------------------------------------------------------------
    # A1/P1/A2 Coverage bands by day-shift-role.
    # Default implementation: C_opt is the hard lower bound and C_max is the hard upper bound.
    # Change --coverage-lower C_min to use paper-style minimum feasibility with target shortage penalty.
    # ------------------------------------------------------------
    w_coverage_short = weights.get("coverage_shortage", 1000.0)
    w_target_short = weights.get("target_shortage", 300.0)
    w_over = weights.get("overstaff", 50.0)

    for d in days:
        for s in shifts:
            for r in roles:
                c_min = cov_value(d, s, r, "C_min")
                c_opt = cov_value(d, s, r, "C_opt")
                c_max = cov_value(d, s, r, "C_max")
                cov = cov_expr(d, s, r)
                lower_value = c_opt if coverage_lower == "C_opt" else c_min

                if soft_coverage:
                    # Penalized shortage instead of hard infeasibility.
                    # This lets the solver return the best legal roster when a cell has
                    # no qualified/available employee, e.g., Day 1-S1-RS.
                    short_lower = pulp.LpVariable(
                        f"short_{coverage_lower}_D{d}_{clean_id(s)}_{clean_id(r)}",
                        lowBound=0, cat="Integer"
                    )
                    prob += cov + short_lower >= lower_value, f"soft_coverage_lower_{d}_{s}_{r}"
                    objective_terms.append(w_coverage_short * short_lower)
                    meta["coverage_short_lower"][(d, s, r)] = short_lower

                    # Extra diagnostic/penalty for falling below C_min.
                    # If coverage_lower is C_opt, a below-min shortage receives both
                    # the lower-bound penalty and the stronger minimum-service signal.
                    short_min = pulp.LpVariable(
                        f"short_Cmin_D{d}_{clean_id(s)}_{clean_id(r)}",
                        lowBound=0, cat="Integer"
                    )
                    prob += short_min >= c_min - cov, f"short_Cmin_def_{d}_{s}_{r}"
                    objective_terms.append(w_coverage_short * short_min)
                    meta["coverage_short_min"][(d, s, r)] = short_min

                    # Target/optimal coverage remains a soft service-quality goal.
                    target_short = pulp.LpVariable(
                        f"short_Copt_D{d}_{clean_id(s)}_{clean_id(r)}",
                        lowBound=0, cat="Integer"
                    )
                    prob += target_short >= c_opt - cov, f"target_short_def_{d}_{s}_{r}"
                    objective_terms.append(w_target_short * target_short)
                    meta["target_shortage"][(d, s, r)] = target_short
                else:
                    prob += cov >= lower_value, f"coverage_lower_{d}_{s}_{r}"
                    # If lower bound is only C_min, target shortage to C_opt is a soft penalty.
                    if coverage_lower == "C_min":
                        target_short = pulp.LpVariable(f"short_Copt_D{d}_{clean_id(s)}_{clean_id(r)}", lowBound=0, cat="Integer")
                        prob += target_short >= c_opt - cov, f"target_short_def_{d}_{s}_{r}"
                        objective_terms.append(w_target_short * target_short)
                        meta["target_shortage"][(d, s, r)] = target_short

                if cmax_hard:
                    prob += cov <= c_max, f"coverage_Cmax_{d}_{s}_{r}"
                else:
                    over_max = pulp.LpVariable(f"over_Cmax_D{d}_{clean_id(s)}_{clean_id(r)}", lowBound=0, cat="Integer")
                    prob += over_max >= cov - c_max, f"soft_over_Cmax_def_{d}_{s}_{r}"
                    objective_terms.append(w_over * 2 * over_max)
                    meta["coverage_over_max"][(d, s, r)] = over_max

                # Soft labor-efficiency penalty above the target coverage C_opt.
                over_opt = pulp.LpVariable(f"over_Copt_D{d}_{clean_id(s)}_{clean_id(r)}", lowBound=0, cat="Integer")
                prob += over_opt >= cov - c_opt, f"over_Copt_def_{d}_{s}_{r}"
                objective_terms.append(w_over * over_opt)
                meta["coverage_over_opt"][(d, s, r)] = over_opt

    # ------------------------------------------------------------
    # A4 Single assignment per employee per day.
    # ------------------------------------------------------------
    for e in employees:
        for d in days:
            prob += work_expr(e, d) <= 1, f"one_shift_per_day_{clean_id(e)}_D{d}"

    # ------------------------------------------------------------
    # A5/A6 Contract day and hour bounds, days off, night limits.
    # ------------------------------------------------------------
    for e in employees:
        row = emp_row.get(e, {})
        total_work_days = lp_sum(work_expr(e, d) for d in days)
        total_hours = lp_sum(shift_hours.get(s, 8.0) * shift_expr(e, d, s) for d in days for s in shifts)
        total_nights = lp_sum(shift_expr(e, d, s) for d in days for s in night_shifts)
        total_days_off = len(days) - total_work_days

        min_work_days = as_int(row.get("min_work_days", 0), 0)
        max_work_days = as_int(row.get("max_work_days", len(days)), len(days))
        min_hours = as_float(row.get("min_hours", 0), 0.0)
        max_hours = as_float(row.get("max_hours", 24 * 8), 24 * 8.0)
        min_days_off = as_int(row.get("min_days_off", 0), 0)
        max_days_off = as_int(row.get("max_days_off", len(days)), len(days))
        max_nights = as_int(row.get("max_night_shifts", len(days)), len(days))

        prob += total_work_days >= min_work_days, f"min_work_days_{clean_id(e)}"
        prob += total_work_days <= max_work_days, f"max_work_days_{clean_id(e)}"
        prob += total_hours >= min_hours, f"min_hours_{clean_id(e)}"
        prob += total_hours <= max_hours, f"max_hours_{clean_id(e)}"
        prob += total_days_off >= min_days_off, f"min_days_off_{clean_id(e)}"
        prob += total_days_off <= max_days_off, f"max_days_off_{clean_id(e)}"
        prob += total_nights <= max_nights, f"max_night_shifts_{clean_id(e)}"

        # Soft fairness around target workdays; intentionally low by default.
        target_days = row.get("target_work_days", pd.NA)
        if not pd.isna(target_days) and fairness_weight > 0:
            target_days = as_float(target_days, 0.0)
            dev_plus = pulp.LpVariable(f"workday_dev_plus_{clean_id(e)}", lowBound=0, cat="Continuous")
            dev_minus = pulp.LpVariable(f"workday_dev_minus_{clean_id(e)}", lowBound=0, cat="Continuous")
            prob += total_work_days - target_days == dev_plus - dev_minus, f"target_workday_dev_{clean_id(e)}"
            objective_terms.append(float(fairness_weight) * (dev_plus + dev_minus))
            meta["soft_vars"].extend([dev_plus, dev_minus])

    # ------------------------------------------------------------
    # N1 Fatigue control: forbidden shift successions and consecutive limits.
    # ------------------------------------------------------------
    # Normal in-horizon rest and consecutiveness rules stay HARD.
    # Only the cross-horizon START HISTORY rules can be softened by default,
    # because generated/old history can otherwise make Day 1 infeasible even
    # when the monthly roster itself has enough staff.
    w_rest_violation = weights.get("rest_violation", 800.0)
    w_consec_work_violation = weights.get("workday_or_hour_violation", 200.0)
    w_consec_night_violation = weights.get("consecutive_night_violation", 600.0)

    forbidden_transitions = []
    for _, row in rest_df.iterrows():
        if as_int(row.get("allowed", 1), 1) == 0 or as_int(row.get("is_forbidden_successession", 0), 0) == 1:
            forbidden_transitions.append((str(row["from_shift"]), str(row["to_shift_next_day"])))

    for e in employees:
        for d in days[:-1]:
            next_d = d + 1
            if next_d not in days:
                continue
            for s_from, s_to in forbidden_transitions:
                prob += shift_expr(e, d, s_from) + shift_expr(e, next_d, s_to) <= 1, \
                    f"forbidden_rest_{clean_id(e)}_D{d}_{clean_id(s_from)}_to_{clean_id(s_to)}"

    hist = data["history"].set_index("employee_id").to_dict("index") if not data["history"].empty else {}
    first_day = min(days)
    for e in employees:
        h = hist.get(e, {})
        prev_shift = str(h.get("previous_last_shift", "OFF"))
        prev_status = str(h.get("previous_last_status", "OFF")).upper()
        if prev_shift in shifts and prev_status == "WORK":
            for s_from, s_to in forbidden_transitions:
                if prev_shift == s_from:
                    if soft_start_history:
                        # Penalized relaxation for the anomaly: previous-horizon
                        # history says this Day-1 transition should be blocked,
                        # but coverage may require one qualified employee.
                        v = pulp.LpVariable(
                            f"soft_history_rest_{clean_id(e)}_{clean_id(prev_shift)}_to_{clean_id(s_to)}",
                            lowBound=0, upBound=1, cat="Binary"
                        )
                        prob += shift_expr(e, first_day, s_to) <= v, \
                            f"soft_history_forbidden_rest_{clean_id(e)}_{clean_id(prev_shift)}_to_{clean_id(s_to)}"
                        objective_terms.append(w_rest_violation * v)
                        meta["soft_vars"].append(v)
                    else:
                        prob += shift_expr(e, first_day, s_to) <= 0, \
                            f"history_forbidden_rest_{clean_id(e)}_{clean_id(prev_shift)}_to_{clean_id(s_to)}"

    for e in employees:
        row = emp_row.get(e, {})
        max_consec_work = as_int(row.get("max_consecutive_work_days", len(days)), len(days))
        max_consec_night = as_int(row.get("max_consecutive_night_shifts", len(days)), len(days))

        if 0 < max_consec_work < len(days):
            window = max_consec_work + 1
            for i in range(0, len(days) - window + 1):
                w_days = days[i:i + window]
                prob += lp_sum(work_expr(e, d) for d in w_days) <= max_consec_work, \
                    f"max_consec_work_{clean_id(e)}_startD{w_days[0]}"

            prev_work = as_int(hist.get(e, {}).get("previous_consecutive_work_days", 0), 0)
            if prev_work > 0:
                allowed_prefix_work = max(0, max_consec_work - prev_work)
                prefix_len = min(len(days), allowed_prefix_work + 1)
                if prefix_len > 0:
                    lhs = lp_sum(work_expr(e, d) for d in days[:prefix_len])
                    if soft_start_history:
                        max_excess = max(1, prefix_len - allowed_prefix_work)
                        v = pulp.LpVariable(
                            f"soft_history_consec_work_{clean_id(e)}",
                            lowBound=0, upBound=max_excess, cat="Integer"
                        )
                        prob += lhs <= allowed_prefix_work + v, f"soft_history_max_consec_work_{clean_id(e)}"
                        objective_terms.append(w_consec_work_violation * v)
                        meta["soft_vars"].append(v)
                    else:
                        prob += lhs <= allowed_prefix_work, f"history_max_consec_work_{clean_id(e)}"

        if night_shifts and 0 < max_consec_night < len(days):
            window = max_consec_night + 1
            for i in range(0, len(days) - window + 1):
                w_days = days[i:i + window]
                prob += lp_sum(shift_expr(e, d, s) for d in w_days for s in night_shifts) <= max_consec_night, \
                    f"max_consec_night_{clean_id(e)}_startD{w_days[0]}"

            prev_night = as_int(hist.get(e, {}).get("previous_consecutive_night_shifts", 0), 0)
            if prev_night > 0:
                allowed_prefix_night = max(0, max_consec_night - prev_night)
                prefix_len = min(len(days), allowed_prefix_night + 1)
                if prefix_len > 0:
                    lhs = lp_sum(shift_expr(e, d, s) for d in days[:prefix_len] for s in night_shifts)
                    if soft_start_history:
                        max_excess = max(1, prefix_len - allowed_prefix_night)
                        v = pulp.LpVariable(
                            f"soft_history_consec_night_{clean_id(e)}",
                            lowBound=0, upBound=max_excess, cat="Integer"
                        )
                        prob += lhs <= allowed_prefix_night + v, f"soft_history_max_consec_night_{clean_id(e)}"
                        objective_terms.append(w_consec_night_violation * v)
                        meta["soft_vars"].append(v)
                    else:
                        prob += lhs <= allowed_prefix_night, f"history_max_consec_night_{clean_id(e)}"

    # ------------------------------------------------------------
    # Weekend and public holiday rules.
    # ------------------------------------------------------------
    holiday_df = data["holiday"].copy()
    if not holiday_df.empty:
        public_days = holiday_df.loc[holiday_df["is_public_holiday"].apply(as_int) == 1, "day"].apply(as_int).tolist()
        for e in employees:
            max_ph = as_int(emp_row.get(e, {}).get("max_working_public_holidays", len(public_days)), len(public_days))
            if public_days:
                prob += lp_sum(work_expr(e, d) for d in public_days if d in days) <= max_ph, \
                    f"max_public_holidays_{clean_id(e)}"

        # Count weekend blocks as pairs of Saturday + following Sunday where possible.
        saturday_days = holiday_df.loc[holiday_df["is_saturday"].apply(as_int) == 1, "day"].apply(as_int).tolist()
        weekend_blocks = []
        for sat in saturday_days:
            sun = sat + 1
            if sun in days:
                weekend_blocks.append((sat, sun))
            else:
                weekend_blocks.append((sat,))

        for e in employees:
            max_weekends = as_int(emp_row.get(e, {}).get("max_working_weekends", len(weekend_blocks)), len(weekend_blocks))
            weekend_work_vars = []
            for idx, block in enumerate(weekend_blocks, start=1):
                z = pulp.LpVariable(f"weekend_work_{clean_id(e)}_{idx}", lowBound=0, upBound=1, cat="Binary")
                for d in block:
                    if d in days:
                        prob += z >= work_expr(e, d), f"weekend_link_{clean_id(e)}_{idx}_D{d}"
                weekend_work_vars.append(z)
            if weekend_work_vars:
                prob += lp_sum(weekend_work_vars) <= max_weekends, f"max_working_weekends_{clean_id(e)}"

    # ------------------------------------------------------------
    # N3 Fixed assignments and fixed day requests.
    # ------------------------------------------------------------
    for _, row in data["fixed_assignments"].iterrows():
        if pd.isna(row.get("employee_id")):
            continue
        e = str(row["employee_id"])
        d = as_int(row["day"])
        s = str(row["shift"])
        r = str(row["role"])
        if (e, d, s, r) not in x:
            raise ValueError(
                f"Fixed assignment is infeasible because variable does not exist: "
                f"employee={e}, day={d}, shift={s}, role={r}. Check skill or availability."
            )
        prob += x[(e, d, s, r)] == 1, f"fixed_assignment_{clean_id(e)}_D{d}_{clean_id(s)}_{clean_id(r)}"

    for _, row in data["day_requests"].iterrows():
        e = str(row.get("employee_id"))
        d = as_int(row.get("day"))
        if e not in employees or d not in days:
            continue
        req = str(row.get("request_type", "")).strip().lower()
        is_fixed = as_int(row.get("is_fixed", 0), 0)
        if is_fixed == 1:
            if req in {"desired_day_off", "day_off", "leave", "fixed_day_off"}:
                prob += work_expr(e, d) == 0, f"fixed_day_off_{clean_id(e)}_D{d}"
            elif req in {"preferred_working_day", "must_work", "fixed_working_day"}:
                prob += work_expr(e, d) == 1, f"fixed_working_day_{clean_id(e)}_D{d}"

    # ------------------------------------------------------------
    # N7 Soft employee requests and preferences.
    # ------------------------------------------------------------
    w_pref_default = weights.get("preference_violation", 10.0)

    for idx, row in data["day_requests"].iterrows():
        e = str(row.get("employee_id"))
        d = as_int(row.get("day"))
        if e not in employees or d not in days or as_int(row.get("is_fixed", 0), 0) == 1:
            continue
        req = str(row.get("request_type", "")).strip().lower()
        penalty = as_float(row.get("penalty", w_pref_default), w_pref_default)
        v = pulp.LpVariable(f"day_request_violation_{idx}_{clean_id(e)}_D{d}", lowBound=0, upBound=1, cat="Binary")
        if req in {"desired_day_off", "day_off"}:
            prob += v >= work_expr(e, d), f"soft_day_off_req_{idx}_{clean_id(e)}_D{d}"
            objective_terms.append(penalty * v)
        elif req in {"preferred_working_day", "prefer_work"}:
            prob += v >= 1 - work_expr(e, d), f"soft_working_day_req_{idx}_{clean_id(e)}_D{d}"
            objective_terms.append(penalty * v)

    for idx, row in data["shift_preferences"].iterrows():
        e = str(row.get("employee_id"))
        d = as_int(row.get("day"))
        s = str(row.get("shift"))
        if e not in employees or d not in days or s not in shifts:
            continue
        pref = str(row.get("preference_type", "")).strip().lower()
        penalty = as_float(row.get("penalty", w_pref_default), w_pref_default)
        v = pulp.LpVariable(f"shift_pref_violation_{idx}_{clean_id(e)}_D{d}_{clean_id(s)}", lowBound=0, upBound=1, cat="Binary")
        if pref == "avoid":
            prob += v >= shift_expr(e, d, s), f"avoid_shift_pref_{idx}_{clean_id(e)}_D{d}_{clean_id(s)}"
            objective_terms.append(penalty * v)
        elif pref == "prefer":
            prob += v >= 1 - shift_expr(e, d, s), f"prefer_shift_pref_{idx}_{clean_id(e)}_D{d}_{clean_id(s)}"
            objective_terms.append(penalty * v)

    for idx, row in data["assignment_preferences"].iterrows():
        e = str(row.get("employee_id"))
        d = as_int(row.get("day"))
        s = str(row.get("shift"))
        r = str(row.get("role"))
        if e not in employees or d not in days or s not in shifts or r not in roles:
            continue
        pref = str(row.get("preference_type", "")).strip().lower()
        penalty = as_float(row.get("penalty", w_pref_default), w_pref_default)
        v = pulp.LpVariable(f"assign_pref_violation_{idx}_{clean_id(e)}_D{d}_{clean_id(s)}_{clean_id(r)}", lowBound=0, upBound=1, cat="Binary")
        if pref == "avoid":
            prob += v >= assign_expr(e, d, s, r), f"avoid_assign_pref_{idx}_{clean_id(e)}_D{d}_{clean_id(s)}_{clean_id(r)}"
            objective_terms.append(penalty * v)
        elif pref == "prefer":
            prob += v >= 1 - assign_expr(e, d, s, r), f"prefer_assign_pref_{idx}_{clean_id(e)}_D{d}_{clean_id(s)}_{clean_id(r)}"
            objective_terms.append(penalty * v)

    # ------------------------------------------------------------
    # N8 Soft incompatibility in same day-shift.
    # ------------------------------------------------------------
    w_incompat_default = weights.get("incompatibility", 100.0)
    incompatible_rows = []
    if not data["incompatibility"].empty:
        tmp = data["incompatibility"].copy()
        tmp = tmp.rename(columns={"employee_i": "employee_i", "employee_j": "employee_j"})
        incompatible_rows.append(tmp[["employee_i", "employee_j", "penalty"]])
    if not data["pairing"].empty:
        tmp = data["pairing"].copy()
        tmp = tmp[tmp["pair_type"].astype(str).str.lower().eq("incompatible")]
        if not tmp.empty:
            incompatible_rows.append(tmp[["employee_i", "employee_j", "penalty"]])
    if incompatible_rows:
        inc_df = pd.concat(incompatible_rows, ignore_index=True).drop_duplicates(subset=["employee_i", "employee_j"])
    else:
        inc_df = pd.DataFrame(columns=["employee_i", "employee_j", "penalty"])

    for idx, row in inc_df.iterrows():
        e1 = str(row.get("employee_i"))
        e2 = str(row.get("employee_j"))
        if e1 not in employees or e2 not in employees:
            continue
        penalty = as_float(row.get("penalty", w_incompat_default), w_incompat_default)
        for d in days:
            for s in shifts:
                v = pulp.LpVariable(f"incompat_{idx}_{clean_id(e1)}_{clean_id(e2)}_D{d}_{clean_id(s)}", lowBound=0, upBound=1, cat="Binary")
                prob += v >= shift_expr(e1, d, s) + shift_expr(e2, d, s) - 1, \
                    f"soft_incompat_def_{idx}_{clean_id(e1)}_{clean_id(e2)}_D{d}_{clean_id(s)}"
                objective_terms.append(penalty * v)

    # ------------------------------------------------------------
    # N5 Standard profile stability.
    # Penalize deviation from employee standard role and standard shift.
    # ------------------------------------------------------------
    w_standard = weights.get("standard_profile_change", 3.0)
    std_profile = data["standard_profile"].set_index("employee_id").to_dict("index") if not data["standard_profile"].empty else {}
    for e in employees:
        std_role = str(std_profile.get(e, {}).get("standard_role", emp_row.get(e, {}).get("standard_role", "")))
        std_shift = str(std_profile.get(e, {}).get("standard_shift", emp_row.get(e, {}).get("standard_shift", "")))
        for d in days:
            for s in shifts:
                for r in roles:
                    var = assign_expr(e, d, s, r)
                    if isinstance(var, (int, float)) and var == 0:
                        continue
                    if std_role and std_role in roles and r != std_role:
                        objective_terms.append(w_standard * var)
                    if std_shift and std_shift in shifts and s != std_shift:
                        objective_terms.append(w_standard * var)

    # Role-shift standard preference, e.g. avoid RS at S1.
    for _, row in data["role_shift_pref"].iterrows():
        role = str(row.get("role"))
        shift = str(row.get("shift"))
        pref = str(row.get("preference_type", "")).strip().lower()
        penalty = as_float(row.get("penalty", 0.0), 0.0)
        if role not in roles or shift not in shifts or penalty <= 0:
            continue
        if pref == "avoid":
            for e in employees:
                for d in days:
                    var = assign_expr(e, d, shift, role)
                    if not (isinstance(var, (int, float)) and var == 0):
                        objective_terms.append(penalty * var)

    prob += lp_sum(objective_terms), "total_weighted_penalty"
    return prob, x, meta


# ============================================================
# 5. SOLVE AND EXPORT RESULTS
# ============================================================
def solve_model(prob: pulp.LpProblem, time_limit: int = 300, gap: float = 0.01, msg: bool = True) -> Dict[str, Any]:
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, gapRel=gap)
    status_code = prob.solve(solver)
    return {
        "status_code": status_code,
        "solver_status": pulp.LpStatus.get(prob.status, str(prob.status)),
        "objective_value": pulp.value(prob.objective),
    }


def value_of(expr: Any) -> float:
    try:
        v = pulp.value(expr)
        return 0.0 if v is None else float(v)
    except Exception:
        return 0.0


def build_outputs(
    data: Dict[str, pd.DataFrame],
    x: Dict[Tuple[str, int, str, str], pulp.LpVariable],
    meta: Dict[str, Any],
    solve_info: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    employees_df = data["employees"].copy()
    coverage_df = data["coverage"].copy()
    employees = meta["employees"]
    days = meta["days"]
    shifts = meta["shifts"]
    roles = meta["roles"]
    shift_time = meta["shift_time"]
    shift_hours = meta["shift_hours"]
    is_night_shift = meta["is_night_shift"]
    blocked_status = meta["blocked_status"]

    # Assignment long table.
    assignment_rows = []
    for (e, d, s, r), var in x.items():
        if value_of(var) > 0.5:
            assignment_rows.append({
                "employee_id": e,
                "day": d,
                "shift": s,
                "shift_time": shift_time.get(s, ""),
                "role": r,
                "assigned": 1,
            })
    assignment_df = pd.DataFrame(assignment_rows)
    if not assignment_df.empty:
        assignment_df = assignment_df.sort_values(["day", "shift", "role", "employee_id"]).reset_index(drop=True)
    else:
        assignment_df = pd.DataFrame(columns=["employee_id", "day", "shift", "shift_time", "role", "assigned"])

    # Roster matrix by employee and day.
    matrix_rows = []
    assign_lookup = {(row.employee_id, row.day): f"{row.shift}-{row.role}" for row in assignment_df.itertuples(index=False)}
    name_map = dict(zip(employees_df["employee_id"], employees_df.get("employee_name", employees_df["employee_id"])))
    for e in employees:
        row = {"employee_id": e, "employee_name": name_map.get(e, e)}
        for d in days:
            if (e, d) in assign_lookup:
                row[f"D{d}"] = assign_lookup[(e, d)]
            elif (e, d) in blocked_status:
                row[f"D{d}"] = "UNAVAILABLE"
            else:
                row[f"D{d}"] = "OFF"
        matrix_rows.append(row)
    roster_matrix_df = pd.DataFrame(matrix_rows)

    # Coverage summary.
    cov_rows = []
    cov_actual = assignment_df.groupby(["day", "shift", "role"]).size().to_dict() if not assignment_df.empty else {}
    for row in coverage_df.itertuples(index=False):
        d = as_int(getattr(row, "day"))
        s = str(getattr(row, "shift"))
        r = str(getattr(row, "role"))
        actual = int(cov_actual.get((d, s, r), 0))
        c_min = as_int(getattr(row, "C_min"))
        c_opt = as_int(getattr(row, "C_opt"))
        c_max = as_int(getattr(row, "C_max"))
        cov_rows.append({
            "day": d,
            "shift": s,
            "shift_time": getattr(row, "shift_time", shift_time.get(s, "")),
            "role": r,
            "C_min": c_min,
            "C_opt": c_opt,
            "C_max": c_max,
            "actual_staff": actual,
            "gap_to_Copt": actual - c_opt,
            "shortage_to_Cmin": max(0, c_min - actual),
            "shortage_to_Copt": max(0, c_opt - actual),
            "overstaff_above_Copt": max(0, actual - c_opt),
            "overstaff_above_Cmax": max(0, actual - c_max),
            "coverage_status": (
                "UNDER_CMIN" if actual < c_min else
                "UNDER_COPT" if actual < c_opt else
                "OVER_CMAX" if actual > c_max else
                "OK"
            ),
        })
    coverage_summary_df = pd.DataFrame(cov_rows).sort_values(["day", "shift", "role"])

    # Employee summary.
    emp_rows = []
    emp_contract = employees_df.set_index("employee_id").to_dict("index")
    for e in employees:
        e_assign = assignment_df[assignment_df["employee_id"] == e]
        work_days = int(e_assign["day"].nunique()) if not e_assign.empty else 0
        hours = 0.0
        night_count = 0
        shift_counts = {f"count_{s}": 0 for s in shifts}
        role_counts = {f"count_{r}": 0 for r in roles}
        for a in e_assign.itertuples(index=False):
            hours += shift_hours.get(a.shift, 8.0)
            night_count += as_int(is_night_shift.get(a.shift, 0))
            shift_counts[f"count_{a.shift}"] = shift_counts.get(f"count_{a.shift}", 0) + 1
            role_counts[f"count_{a.role}"] = role_counts.get(f"count_{a.role}", 0) + 1
        c = emp_contract.get(e, {})
        emp_rows.append({
            "employee_id": e,
            "employee_name": name_map.get(e, e),
            "work_days": work_days,
            "hours": hours,
            "night_shifts": night_count,
            "min_work_days": c.get("min_work_days", pd.NA),
            "max_work_days": c.get("max_work_days", pd.NA),
            "target_work_days": c.get("target_work_days", pd.NA),
            "min_hours": c.get("min_hours", pd.NA),
            "max_hours": c.get("max_hours", pd.NA),
            "target_hours": c.get("target_hours", pd.NA),
            "standard_role": c.get("standard_role", pd.NA),
            "standard_shift": c.get("standard_shift", pd.NA),
            **shift_counts,
            **role_counts,
        })
    employee_summary_df = pd.DataFrame(emp_rows)

    # Status summary.
    total_required_opt = int(coverage_df["C_opt"].sum()) if "C_opt" in coverage_df.columns else 0
    total_required_min = int(coverage_df["C_min"].sum()) if "C_min" in coverage_df.columns else 0
    total_required_max = int(coverage_df["C_max"].sum()) if "C_max" in coverage_df.columns else 0
    actual_assignments = int(len(assignment_df))
    status_rows = [
        {"metric": "solver_status", "value": solve_info.get("solver_status")},
        {"metric": "objective_value", "value": solve_info.get("objective_value")},
        {"metric": "coverage_lower_used", "value": meta.get("coverage_lower")},
        {"metric": "cmax_hard", "value": meta.get("cmax_hard")},
        {"metric": "soft_start_history", "value": meta.get("soft_start_history")},
        {"metric": "soft_coverage", "value": meta.get("soft_coverage")},
        {"metric": "total_C_min", "value": total_required_min},
        {"metric": "total_C_opt_required", "value": total_required_opt},
        {"metric": "total_C_max", "value": total_required_max},
        {"metric": "actual_assignments", "value": actual_assignments},
        {"metric": "total_overstaff_above_Copt", "value": int(coverage_summary_df["overstaff_above_Copt"].sum())},
        {"metric": "total_shortage_to_Copt", "value": int(coverage_summary_df["shortage_to_Copt"].sum())},
        {"metric": "coverage_under_Cmin_cells", "value": int((coverage_summary_df["coverage_status"] == "UNDER_CMIN").sum())},
        {"metric": "coverage_under_Copt_cells", "value": int((coverage_summary_df["shortage_to_Copt"] > 0).sum())},
        {"metric": "coverage_over_Cmax_cells", "value": int((coverage_summary_df["overstaff_above_Cmax"] > 0).sum())},
    ]
    model_status_df = pd.DataFrame(status_rows)

    return {
        "assignment_long": assignment_df,
        "roster_matrix": roster_matrix_df,
        "coverage_summary": coverage_summary_df,
        "employee_summary": employee_summary_df,
        "model_status": model_status_df,
        "demand_input": coverage_df,
    }


def export_outputs(outputs: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    xlsx_path = output_dir / "milp_roster_output.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for sheet_name, df in outputs.items():
            safe_sheet = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)
    return xlsx_path


# ============================================================
# 6. MAIN
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve SASCO airport lounge MILP roster model.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder containing employee_milp_inputs CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for MILP solution outputs.")
    parser.add_argument("--coverage-lower", choices=["C_min", "C_opt"], default="C_opt", help="Hard lower coverage level. Use C_opt for main staffing requirement; use C_min for paper-style minimum with soft target shortage.")
    parser.add_argument("--cmax-hard", dest="cmax_hard", action="store_true", default=True, help="Treat C_max as a hard upper bound.")
    parser.add_argument("--soft-cmax", dest="cmax_hard", action="store_false", help="Treat C_max as soft with overstaff penalty.")
    parser.add_argument("--time-limit", type=int, default=300, help="CBC solver time limit in seconds. Default is 300 for MEDIUM scale.")
    parser.add_argument("--gap", type=float, default=0.01, help="Relative MIP gap for CBC.")
    parser.add_argument("--fairness-weight", type=float, default=1.0, help="Small penalty for deviation from target workdays.")
    parser.add_argument("--soft-start-history", dest="soft_start_history", action="store_true", default=True, help="Treat previous-horizon Day-1 history conflicts as penalized soft violations instead of hard infeasibility.")
    parser.add_argument("--hard-start-history", dest="soft_start_history", action="store_false", help="Keep previous-horizon Day-1 history conflicts as hard constraints.")
    parser.add_argument("--soft-coverage", dest="soft_coverage", action="store_true", default=True, help="Treat coverage lower-bound shortages as penalized soft violations instead of hard infeasibility.")
    parser.add_argument("--hard-coverage", dest="soft_coverage", action="store_false", help="Keep coverage lower bound as a hard constraint.")
    parser.add_argument("--solver-msg", action="store_true", help="Show CBC solver log.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=== MILP MEDIUM SCALE RUN ===")
    print("Expected input folder: employee_milp_inputs_medium")
    print("Expected output folder: MILP_Result_medium")
    print(f"Reading input data from: {args.input_dir}")
    data = load_inputs(args.input_dir)

    print("Building MILP model...")
    prob, x, meta = build_model(
        data=data,
        coverage_lower=args.coverage_lower,
        cmax_hard=args.cmax_hard,
        fairness_weight=args.fairness_weight,
        soft_start_history=args.soft_start_history,
        soft_coverage=args.soft_coverage,
    )
    print(f"Variables: {len(prob.variables()):,}")
    print(f"Constraints: {len(prob.constraints):,}")

    print("Solving MILP...")
    solve_info = solve_model(prob, time_limit=args.time_limit, gap=args.gap, msg=args.solver_msg)
    print(f"Solver status: {solve_info['solver_status']}")
    print(f"Objective value: {solve_info['objective_value']}")

    print("Building output tables...")
    outputs = build_outputs(data, x, meta, solve_info)
    xlsx_path = export_outputs(outputs, args.output_dir)
    print(f"Done. Main Excel output: {xlsx_path}")
    print(f"CSV outputs are also saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
