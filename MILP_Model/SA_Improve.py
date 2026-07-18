r"""
SA Phase 2 - Small Scale Roster Improvement
===========================================

Purpose
-------
This script applies Simulated Annealing (SA) as a Phase-2 improvement method
for the SASCO Airport Lounge roster model.

It does NOT replace the MILP. It uses the MILP assignment_long output as the
initial solution S0, then improves the roster by local neighborhood search:

1) Reassign same shift-role coverage to another qualified employee
2) Swap assignments between employees
3) Move one assignment from an overstaffed cell to an under-target cell
4) Drop assignment from an overstaffed cell if contract feasibility allows
5) Add assignment to an under-target cell if qualified resources exist

Main improvement targets
------------------------
- Keep hard violations at zero or as low as possible
- Reduce shortage below C_min / C_opt
- Reduce overstaffing above C_opt and C_max
- Reduce employee preference violations
- Improve fairness around target workdays
- Improve standard role / standard shift stability

Expected folders
----------------
Input folder should contain the same CSV files used by the MILP:
    employee_master.csv
    employee_skills.csv
    staffing_coverage_bands_long.csv
    shift_structure.csv
    shift_transition_rest.csv
    penalty_config.csv
    employee_availability.csv                       optional
    employee_day_requests.csv                       optional
    employee_preferences.csv                        optional
    employee_assignment_preferences.csv             optional
    employee_fixed_assignments.csv                  optional
    employee_history.csv                            optional
    employee_standard_profile.csv                   optional
    employee_incompatibility.csv                    optional
    employee_pairing.csv                            optional
    standard_role_shift_preferences.csv             optional
    holiday_calendar.csv                            optional

MILP output folder should contain:
    assignment_long.csv
or:
    milp_roster_output.xlsx with sheet assignment_long

Example small-scale run
-----------------------
python sa_phase2_small_scale.py ^
  --input-dir "E:\\NĂM 4\\Capstone\\Data_input\\employee_milp_inputs" ^
  --milp-output-dir "E:\\NĂM 4\\Capstone\\MILP_Model\\MILP_Result" ^
  --output-dir "E:\\NĂM 4\\Capstone\\MILP_Model\\SA_Result_small" ^
  --initial-temp 500 ^
  --min-temp 0.1 ^
  --cooling-rate 0.95 ^
  --cycles-per-temp 700 ^
  --seed 42

Notes
-----
- For small scale, the search is intentionally conservative. It treats C_min
  and C_max as hard service feasibility boundaries by applying a very large
  penalty to violations.
- C_opt is treated as the main service-quality target.
- The score reported by this script is a Common Evaluation Score, not the raw
  MILP objective. Use it to evaluate both MILP baseline and SA output fairly.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 1. CONSTANTS
# ============================================================

ROLE_COLUMNS = ["RS", "DAS", "BLO", "FSTC", "DS", "SLS"]
DEFAULT_HARD_PENALTY = 100_000.0

BAD_AVAILABILITY_STATUS = {
    "unavailable", "leave", "paid_leave", "unpaid_leave", "holiday",
    "vacation", "recovery", "not_available", "blocked", "off",
}


# ============================================================
# 2. BASIC HELPERS
# ============================================================

def read_csv_required(input_dir: Path, filename: str) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def read_csv_optional(input_dir: Path, filename: str, columns: Sequence[str]) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    df = pd.read_csv(path, encoding="utf-8-sig")
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df[list(columns)]


def as_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize_shift(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().upper()
    s = s.replace("SHIFT", "").replace("_", "").replace("-", "").strip()
    if s in {"1", "01"}:
        return "S1"
    if s in {"2", "02"}:
        return "S2"
    if s in {"3", "03"}:
        return "S3"
    return s


def safe_value(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


# ============================================================
# 3. DATA STRUCTURES
# ============================================================

AssignmentValue = Optional[Tuple[str, str]]  # None or (shift, role)
State = Dict[Tuple[str, int], AssignmentValue]


@dataclass
class ProblemData:
    input_dir: Path
    milp_output_dir: Path

    employees_df: pd.DataFrame
    skills_df: pd.DataFrame
    coverage_df: pd.DataFrame
    shifts_df: pd.DataFrame
    rest_df: pd.DataFrame
    penalties_df: pd.DataFrame
    availability_df: pd.DataFrame
    day_requests_df: pd.DataFrame
    shift_preferences_df: pd.DataFrame
    assignment_preferences_df: pd.DataFrame
    fixed_assignments_df: pd.DataFrame
    history_df: pd.DataFrame
    standard_profile_df: pd.DataFrame
    incompatibility_df: pd.DataFrame
    pairing_df: pd.DataFrame
    holiday_df: pd.DataFrame
    role_shift_pref_df: pd.DataFrame

    employees: List[str]
    days: List[int]
    shifts: List[str]
    roles: List[str]

    shift_time: Dict[str, str]
    shift_hours: Dict[str, float]
    is_night_shift: Dict[str, int]
    night_shifts: List[str]

    skill: Dict[Tuple[str, str], int]
    coverage: Dict[Tuple[int, str, str], Dict[str, int]]
    weights: Dict[str, float]

    blocked_days: set = field(default_factory=set)
    fixed_assignments: Dict[Tuple[str, int], Tuple[str, str]] = field(default_factory=dict)
    fixed_day_off: set = field(default_factory=set)
    forbidden_transitions: set = field(default_factory=set)

    employee_rule: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    standard_profile: Dict[str, Dict[str, str]] = field(default_factory=dict)

    day_request_map: Dict[Tuple[str, int], List[Dict[str, Any]]] = field(default_factory=dict)
    shift_pref_map: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = field(default_factory=dict)
    assign_pref_map: Dict[Tuple[str, int, str, str], List[Dict[str, Any]]] = field(default_factory=dict)
    role_shift_avoid: Dict[Tuple[str, str], float] = field(default_factory=dict)
    incompatible_pairs: List[Tuple[str, str, float]] = field(default_factory=list)


@dataclass
class EvaluationResult:
    score: float
    metrics: Dict[str, Any]
    coverage_counts: Dict[Tuple[int, str, str], int]
    workdays: Dict[str, int]


# ============================================================
# 4. LOAD INPUTS
# ============================================================

def penalty_weights(penalty_df: pd.DataFrame) -> Dict[str, float]:
    if penalty_df.empty:
        return {}
    if "penalty_component" not in penalty_df.columns or "weight" not in penalty_df.columns:
        return {}
    return {
        str(row["penalty_component"]): as_float(row["weight"], 0.0)
        for _, row in penalty_df.iterrows()
    }


def load_initial_assignment(milp_output_dir: Path) -> pd.DataFrame:
    csv_path = milp_output_dir / "assignment_long.csv"
    xlsx_path = milp_output_dir / "milp_roster_output.xlsx"

    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    elif xlsx_path.exists():
        df = pd.read_excel(xlsx_path, sheet_name="assignment_long")
    else:
        raise FileNotFoundError(
            "Could not find MILP assignment output. Expected assignment_long.csv "
            f"or milp_roster_output.xlsx in: {milp_output_dir}"
        )

    required = {"employee_id", "day", "shift", "role"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MILP assignment_long is missing columns: {sorted(missing)}")

    df = df.copy()
    df["employee_id"] = df["employee_id"].astype(str)
    df["day"] = df["day"].apply(as_int)
    df["shift"] = df["shift"].apply(normalize_shift)
    df["role"] = df["role"].astype(str).str.strip().str.upper()
    return df[df["role"].isin(ROLE_COLUMNS)].copy()


def load_problem(input_dir: Path, milp_output_dir: Path) -> Tuple[ProblemData, pd.DataFrame]:
    employees_df = read_csv_required(input_dir, "employee_master.csv")
    skills_df = read_csv_required(input_dir, "employee_skills.csv")
    coverage_df = read_csv_required(input_dir, "staffing_coverage_bands_long.csv")
    shifts_df = read_csv_required(input_dir, "shift_structure.csv")
    rest_df = read_csv_required(input_dir, "shift_transition_rest.csv")
    penalties_df = read_csv_required(input_dir, "penalty_config.csv")

    availability_df = read_csv_optional(input_dir, "employee_availability.csv", ["employee_id", "day", "status"])
    day_requests_df = read_csv_optional(input_dir, "employee_day_requests.csv", ["employee_id", "day", "request_type", "penalty", "is_fixed"])
    shift_preferences_df = read_csv_optional(input_dir, "employee_preferences.csv", ["employee_id", "day", "shift", "preference_type", "penalty"])
    assignment_preferences_df = read_csv_optional(input_dir, "employee_assignment_preferences.csv", ["employee_id", "day", "shift", "role", "preference_type", "penalty"])
    fixed_assignments_df = read_csv_optional(input_dir, "employee_fixed_assignments.csv", ["employee_id", "day", "shift", "role"])
    history_df = read_csv_optional(input_dir, "employee_history.csv", [
        "employee_id", "previous_last_shift", "previous_last_status",
        "previous_consecutive_work_days", "previous_consecutive_days_off",
        "previous_consecutive_night_shifts",
    ])
    standard_profile_df = read_csv_optional(input_dir, "employee_standard_profile.csv", ["employee_id", "standard_role", "standard_shift"])
    incompatibility_df = read_csv_optional(input_dir, "employee_incompatibility.csv", ["employee_i", "employee_j", "incompatibility_type", "priority", "penalty"])
    pairing_df = read_csv_optional(input_dir, "employee_pairing.csv", ["employee_i", "employee_j", "pair_type", "priority", "penalty"])
    holiday_df = read_csv_optional(input_dir, "holiday_calendar.csv", ["day", "weekday_index", "is_saturday", "is_sunday", "is_weekend", "is_public_holiday"])
    role_shift_pref_df = read_csv_optional(input_dir, "standard_role_shift_preferences.csv", ["role", "shift", "preference_type", "penalty", "note"])

    # Standardize main tables.
    employees_df = employees_df.copy()
    employees_df["employee_id"] = employees_df["employee_id"].astype(str)
    employees = employees_df["employee_id"].tolist()

    skills_df = skills_df.copy()
    skills_df["employee_id"] = skills_df["employee_id"].astype(str)
    for role in ROLE_COLUMNS:
        if role not in skills_df.columns:
            skills_df[role] = 0
        skills_df[role] = skills_df[role].apply(as_int)

    coverage_df = coverage_df.copy()
    coverage_df["day"] = coverage_df["day"].apply(as_int)
    coverage_df["shift"] = coverage_df["shift"].apply(normalize_shift)
    coverage_df["role"] = coverage_df["role"].astype(str).str.strip().str.upper()
    for col in ["C_min", "C_opt", "C_max"]:
        if col not in coverage_df.columns:
            raise ValueError(f"coverage file missing {col}")
        coverage_df[col] = coverage_df[col].apply(as_int)
    coverage_df = coverage_df[coverage_df["role"].isin(ROLE_COLUMNS)].copy()

    shifts_df = shifts_df.copy()
    shifts_df["shift"] = shifts_df["shift"].apply(normalize_shift)
    if "duration_hours" not in shifts_df.columns:
        shifts_df["duration_hours"] = 8
    if "is_night_shift" not in shifts_df.columns:
        shifts_df["is_night_shift"] = (shifts_df["shift"] == "S1").astype(int)
    if "shift_time" not in shifts_df.columns:
        shifts_df["shift_time"] = shifts_df["shift"].map({"S1": "00:00-08:00", "S2": "08:00-16:00", "S3": "16:00-24:00"}).fillna("")

    days = sorted(coverage_df["day"].dropna().astype(int).unique().tolist())
    shifts = shifts_df["shift"].astype(str).tolist()
    roles = ROLE_COLUMNS[:]

    shift_time = dict(zip(shifts_df["shift"].astype(str), shifts_df["shift_time"].astype(str)))
    shift_hours = dict(zip(shifts_df["shift"].astype(str), shifts_df["duration_hours"].apply(as_float)))
    is_night_shift = dict(zip(shifts_df["shift"].astype(str), shifts_df["is_night_shift"].apply(as_int)))
    night_shifts = [s for s in shifts if is_night_shift.get(s, 0) == 1]

    skill = {}
    for _, row in skills_df.iterrows():
        e = str(row["employee_id"])
        for role in roles:
            skill[(e, role)] = as_int(row.get(role, 0), 0)

    coverage = {}
    for _, row in coverage_df.iterrows():
        coverage[(int(row["day"]), str(row["shift"]), str(row["role"]))] = {
            "C_min": as_int(row["C_min"]),
            "C_opt": as_int(row["C_opt"]),
            "C_max": as_int(row["C_max"]),
        }

    weights = penalty_weights(penalties_df)

    # Blocked availability.
    blocked_days = set()
    if not availability_df.empty:
        availability_df = availability_df.copy()
        availability_df["employee_id"] = availability_df["employee_id"].astype(str)
        availability_df["day"] = availability_df["day"].apply(as_int)
        for _, row in availability_df.iterrows():
            status = str(row.get("status", "")).strip().lower().replace(" ", "_")
            if status in BAD_AVAILABILITY_STATUS:
                blocked_days.add((str(row["employee_id"]), int(row["day"])))

    fixed_day_off = set()
    fixed_assignments = {}
    if not day_requests_df.empty:
        day_requests_df = day_requests_df.copy()
        day_requests_df["employee_id"] = day_requests_df["employee_id"].astype(str)
        day_requests_df["day"] = day_requests_df["day"].apply(as_int)
        for _, row in day_requests_df.iterrows():
            req = str(row.get("request_type", "")).strip().lower()
            is_fixed = as_int(row.get("is_fixed", 0), 0)
            if is_fixed == 1 and req in {"desired_day_off", "day_off", "leave", "fixed_day_off"}:
                fixed_day_off.add((str(row["employee_id"]), int(row["day"])))
                blocked_days.add((str(row["employee_id"]), int(row["day"])))

    if not fixed_assignments_df.empty:
        fixed_assignments_df = fixed_assignments_df.copy()
        fixed_assignments_df["employee_id"] = fixed_assignments_df["employee_id"].astype(str)
        fixed_assignments_df["day"] = fixed_assignments_df["day"].apply(as_int)
        fixed_assignments_df["shift"] = fixed_assignments_df["shift"].apply(normalize_shift)
        fixed_assignments_df["role"] = fixed_assignments_df["role"].astype(str).str.strip().str.upper()
        for _, row in fixed_assignments_df.iterrows():
            fixed_assignments[(str(row["employee_id"]), int(row["day"]))] = (str(row["shift"]), str(row["role"]))

    # Rest transition rules.
    forbidden_transitions = set()
    if not rest_df.empty:
        for _, row in rest_df.iterrows():
            allowed = as_int(row.get("allowed", 1), 1)
            forbidden_flag = as_int(row.get("is_forbidden_successession", 0), 0)
            if allowed == 0 or forbidden_flag == 1:
                s_from = normalize_shift(row.get("from_shift"))
                s_to = normalize_shift(row.get("to_shift_next_day"))
                if s_from and s_to:
                    forbidden_transitions.add((s_from, s_to))

    # Employee rules.
    employee_rule = employees_df.set_index("employee_id").to_dict("index")

    # Standard profile.
    standard_profile = {}
    if not standard_profile_df.empty:
        standard_profile_df = standard_profile_df.copy()
        standard_profile_df["employee_id"] = standard_profile_df["employee_id"].astype(str)
        for _, row in standard_profile_df.iterrows():
            standard_profile[str(row["employee_id"])] = {
                "standard_role": str(row.get("standard_role", "")),
                "standard_shift": normalize_shift(row.get("standard_shift", "")),
            }

    # Preference maps.
    day_request_map: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for _, row in day_requests_df.iterrows():
        e = str(row.get("employee_id"))
        d = as_int(row.get("day"))
        if e and d:
            day_request_map.setdefault((e, d), []).append(row.to_dict())

    shift_pref_map: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    if not shift_preferences_df.empty:
        shift_preferences_df = shift_preferences_df.copy()
        shift_preferences_df["employee_id"] = shift_preferences_df["employee_id"].astype(str)
        shift_preferences_df["day"] = shift_preferences_df["day"].apply(as_int)
        shift_preferences_df["shift"] = shift_preferences_df["shift"].apply(normalize_shift)
        for _, row in shift_preferences_df.iterrows():
            key = (str(row["employee_id"]), int(row["day"]), str(row["shift"]))
            shift_pref_map.setdefault(key, []).append(row.to_dict())

    assign_pref_map: Dict[Tuple[str, int, str, str], List[Dict[str, Any]]] = {}
    if not assignment_preferences_df.empty:
        assignment_preferences_df = assignment_preferences_df.copy()
        assignment_preferences_df["employee_id"] = assignment_preferences_df["employee_id"].astype(str)
        assignment_preferences_df["day"] = assignment_preferences_df["day"].apply(as_int)
        assignment_preferences_df["shift"] = assignment_preferences_df["shift"].apply(normalize_shift)
        assignment_preferences_df["role"] = assignment_preferences_df["role"].astype(str).str.strip().str.upper()
        for _, row in assignment_preferences_df.iterrows():
            key = (str(row["employee_id"]), int(row["day"]), str(row["shift"]), str(row["role"]))
            assign_pref_map.setdefault(key, []).append(row.to_dict())

    role_shift_avoid = {}
    if not role_shift_pref_df.empty:
        role_shift_pref_df = role_shift_pref_df.copy()
        role_shift_pref_df["role"] = role_shift_pref_df["role"].astype(str).str.strip().str.upper()
        role_shift_pref_df["shift"] = role_shift_pref_df["shift"].apply(normalize_shift)
        for _, row in role_shift_pref_df.iterrows():
            pref = str(row.get("preference_type", "")).strip().lower()
            if pref == "avoid":
                role_shift_avoid[(str(row["role"]), str(row["shift"]))] = as_float(row.get("penalty", 0.0), 0.0)

    incompatible_pairs: List[Tuple[str, str, float]] = []
    if not incompatibility_df.empty:
        for _, row in incompatibility_df.iterrows():
            e1 = safe_value(row.get("employee_i"))
            e2 = safe_value(row.get("employee_j"))
            if e1 and e2:
                incompatible_pairs.append((e1, e2, as_float(row.get("penalty", weights.get("incompatibility", 100.0)), weights.get("incompatibility", 100.0))))
    if not pairing_df.empty:
        for _, row in pairing_df.iterrows():
            pair_type = str(row.get("pair_type", "")).strip().lower()
            if pair_type == "incompatible":
                e1 = safe_value(row.get("employee_i"))
                e2 = safe_value(row.get("employee_j"))
                if e1 and e2:
                    incompatible_pairs.append((e1, e2, as_float(row.get("penalty", weights.get("incompatibility", 100.0)), weights.get("incompatibility", 100.0))))

    problem = ProblemData(
        input_dir=input_dir,
        milp_output_dir=milp_output_dir,
        employees_df=employees_df,
        skills_df=skills_df,
        coverage_df=coverage_df,
        shifts_df=shifts_df,
        rest_df=rest_df,
        penalties_df=penalties_df,
        availability_df=availability_df,
        day_requests_df=day_requests_df,
        shift_preferences_df=shift_preferences_df,
        assignment_preferences_df=assignment_preferences_df,
        fixed_assignments_df=fixed_assignments_df,
        history_df=history_df,
        standard_profile_df=standard_profile_df,
        incompatibility_df=incompatibility_df,
        pairing_df=pairing_df,
        holiday_df=holiday_df,
        role_shift_pref_df=role_shift_pref_df,
        employees=employees,
        days=days,
        shifts=shifts,
        roles=roles,
        shift_time=shift_time,
        shift_hours=shift_hours,
        is_night_shift=is_night_shift,
        night_shifts=night_shifts,
        skill=skill,
        coverage=coverage,
        weights=weights,
        blocked_days=blocked_days,
        fixed_assignments=fixed_assignments,
        fixed_day_off=fixed_day_off,
        forbidden_transitions=forbidden_transitions,
        employee_rule=employee_rule,
        standard_profile=standard_profile,
        day_request_map=day_request_map,
        shift_pref_map=shift_pref_map,
        assign_pref_map=assign_pref_map,
        role_shift_avoid=role_shift_avoid,
        incompatible_pairs=incompatible_pairs,
    )

    initial_assignment_df = load_initial_assignment(milp_output_dir)
    return problem, initial_assignment_df


# ============================================================
# 5. STATE BUILDING AND CHECKS
# ============================================================

def build_initial_state(problem: ProblemData, assignment_df: pd.DataFrame) -> State:
    state: State = {(e, d): None for e in problem.employees for d in problem.days}

    duplicated = assignment_df.duplicated(subset=["employee_id", "day"], keep=False)
    if duplicated.any():
        bad = assignment_df.loc[duplicated, ["employee_id", "day", "shift", "role"]].head(20)
        raise ValueError(
            "MILP initial solution has more than one assignment per employee-day. "
            f"Examples:\n{bad.to_string(index=False)}"
        )

    for _, row in assignment_df.iterrows():
        e = str(row["employee_id"])
        d = int(row["day"])
        s = normalize_shift(row["shift"])
        r = str(row["role"]).strip().upper()
        if e in problem.employees and d in problem.days and s in problem.shifts and r in problem.roles:
            state[(e, d)] = (s, r)

    return state


def is_fixed_employee_day(problem: ProblemData, e: str, d: int) -> bool:
    return (e, d) in problem.fixed_assignments or (e, d) in problem.fixed_day_off


def can_assign(problem: ProblemData, e: str, d: int, s: str, r: str, respect_fixed: bool = True) -> bool:
    if e not in problem.employees or d not in problem.days or s not in problem.shifts or r not in problem.roles:
        return False
    if (e, d) in problem.blocked_days:
        return False
    if problem.skill.get((e, r), 0) != 1:
        return False
    if respect_fixed:
        fixed = problem.fixed_assignments.get((e, d))
        if fixed is not None and fixed != (s, r):
            return False
        if (e, d) in problem.fixed_day_off:
            return False
    return True


def copy_state(state: State) -> State:
    return dict(state)


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def count_consecutive_violations(values: List[int], max_allowed: int) -> int:
    if max_allowed <= 0 or max_allowed >= len(values):
        return 0
    violations = 0
    current = 0
    for v in values:
        if v:
            current += 1
            if current > max_allowed:
                violations += 1
        else:
            current = 0
    return violations


def evaluate_state(problem: ProblemData, state: State, hard_penalty: float = DEFAULT_HARD_PENALTY) -> EvaluationResult:
    w = problem.weights
    w_shortage = w.get("coverage_shortage", 1000.0)
    w_target_short = w.get("target_shortage", 300.0)
    w_over = w.get("overstaff", 50.0)
    w_pref = w.get("preference_violation", 10.0)
    w_incompat = w.get("incompatibility", 100.0)
    w_standard = w.get("standard_profile_change", 3.0)
    w_rest = w.get("rest_violation", 800.0)
    w_consec = w.get("workday_or_hour_violation", 200.0)
    w_consec_night = w.get("consecutive_night_violation", 600.0)
    fairness_weight = w.get("fairness", 1.0) if "fairness" in w else 1.0

    score = 0.0

    coverage_counts = {key: 0 for key in problem.coverage.keys()}
    workdays = {e: 0 for e in problem.employees}
    hours = {e: 0.0 for e in problem.employees}
    night_counts = {e: 0 for e in problem.employees}
    employee_day_shift: Dict[Tuple[str, int], Optional[str]] = {(e, d): None for e in problem.employees for d in problem.days}
    same_day_shift_employees: Dict[Tuple[int, str], set] = {(d, s): set() for d in problem.days for s in problem.shifts}

    hard_violations = 0
    skill_violations = 0
    availability_violations = 0
    fixed_assignment_violations = 0
    rest_violations = 0
    contract_violations = 0
    consecutive_work_violations = 0
    consecutive_night_violations = 0
    preference_penalty = 0.0
    standard_penalty = 0.0
    incompatibility_penalty = 0.0
    fairness_penalty = 0.0

    # Assignment-level checks and soft preferences.
    for (e, d), val in state.items():
        if val is None:
            # Soft preferred working day request.
            for req in problem.day_request_map.get((e, d), []):
                if as_int(req.get("is_fixed", 0), 0) == 1:
                    continue
                typ = str(req.get("request_type", "")).strip().lower()
                p = as_float(req.get("penalty", w_pref), w_pref)
                if typ in {"preferred_working_day", "prefer_work"}:
                    preference_penalty += p
            continue

        s, r = val
        workdays[e] += 1
        hours[e] += problem.shift_hours.get(s, 8.0)
        night_counts[e] += int(problem.is_night_shift.get(s, 0))
        employee_day_shift[(e, d)] = s
        same_day_shift_employees.setdefault((d, s), set()).add(e)

        if (d, s, r) in coverage_counts:
            coverage_counts[(d, s, r)] += 1
        else:
            hard_violations += 1

        if problem.skill.get((e, r), 0) != 1:
            skill_violations += 1
            hard_violations += 1

        if (e, d) in problem.blocked_days:
            availability_violations += 1
            hard_violations += 1

        fixed = problem.fixed_assignments.get((e, d))
        if fixed is not None and fixed != (s, r):
            fixed_assignment_violations += 1
            hard_violations += 1
        if (e, d) in problem.fixed_day_off:
            fixed_assignment_violations += 1
            hard_violations += 1

        # Day request soft violation.
        for req in problem.day_request_map.get((e, d), []):
            if as_int(req.get("is_fixed", 0), 0) == 1:
                continue
            typ = str(req.get("request_type", "")).strip().lower()
            p = as_float(req.get("penalty", w_pref), w_pref)
            if typ in {"desired_day_off", "day_off"}:
                preference_penalty += p

        # Shift preference.
        for pref in problem.shift_pref_map.get((e, d, s), []):
            ptype = str(pref.get("preference_type", "")).strip().lower()
            p = as_float(pref.get("penalty", w_pref), w_pref)
            if ptype == "avoid":
                preference_penalty += p
        for s2 in problem.shifts:
            if s2 == s:
                continue
            for pref in problem.shift_pref_map.get((e, d, s2), []):
                ptype = str(pref.get("preference_type", "")).strip().lower()
                p = as_float(pref.get("penalty", w_pref), w_pref)
                if ptype == "prefer":
                    preference_penalty += p

        # Assignment preference.
        for pref in problem.assign_pref_map.get((e, d, s, r), []):
            ptype = str(pref.get("preference_type", "")).strip().lower()
            p = as_float(pref.get("penalty", w_pref), w_pref)
            if ptype == "avoid":
                preference_penalty += p

        # Standard role/shift stability.
        std = problem.standard_profile.get(e, {})
        std_role = str(std.get("standard_role", ""))
        std_shift = normalize_shift(std.get("standard_shift", ""))
        if std_role in problem.roles and r != std_role:
            standard_penalty += w_standard
        if std_shift in problem.shifts and s != std_shift:
            standard_penalty += w_standard

        # Role-shift avoid.
        standard_penalty += problem.role_shift_avoid.get((r, s), 0.0)

    # Coverage score.
    shortage_to_cmin = 0
    shortage_to_copt = 0
    overstaff_above_copt = 0
    overstaff_above_cmax = 0
    under_cmin_cells = 0
    under_copt_cells = 0
    over_cmax_cells = 0

    for key, band in problem.coverage.items():
        actual = int(coverage_counts.get(key, 0))
        c_min = int(band["C_min"])
        c_opt = int(band["C_opt"])
        c_max = int(band["C_max"])

        below_min = max(0, c_min - actual)
        below_opt = max(0, c_opt - actual)
        above_opt = max(0, actual - c_opt)
        above_max = max(0, actual - c_max)

        shortage_to_cmin += below_min
        shortage_to_copt += below_opt
        overstaff_above_copt += above_opt
        overstaff_above_cmax += above_max

        if below_min > 0:
            under_cmin_cells += 1
            hard_violations += below_min
        if below_opt > 0:
            under_copt_cells += 1
        if above_max > 0:
            over_cmax_cells += 1
            hard_violations += above_max

        # C_min and C_max are hard feasibility boundaries.
        score += hard_penalty * below_min
        score += hard_penalty * above_max
        # C_opt is target service quality.
        score += w_target_short * below_opt
        score += w_over * above_opt

    # Contract, fairness, and fatigue.
    for e in problem.employees:
        rule = problem.employee_rule.get(e, {})
        wd = int(workdays[e])
        hr = float(hours[e])
        nights = int(night_counts[e])

        min_work_days = as_int(rule.get("min_work_days", 0), 0)
        max_work_days = as_int(rule.get("max_work_days", len(problem.days)), len(problem.days))
        min_hours = as_float(rule.get("min_hours", min_work_days * 8), min_work_days * 8.0)
        max_hours = as_float(rule.get("max_hours", max_work_days * 8), max_work_days * 8.0)
        target_work_days = as_float(rule.get("target_work_days", (min_work_days + max_work_days) / 2), (min_work_days + max_work_days) / 2)
        max_night_shifts = as_int(rule.get("max_night_shifts", len(problem.days)), len(problem.days))
        max_consec_work = as_int(rule.get("max_consecutive_work_days", len(problem.days)), len(problem.days))
        max_consec_night = as_int(rule.get("max_consecutive_night_shifts", len(problem.days)), len(problem.days))

        if wd < min_work_days:
            deficit = min_work_days - wd
            contract_violations += deficit
            hard_violations += deficit
            score += hard_penalty * deficit
        if wd > max_work_days:
            excess = wd - max_work_days
            contract_violations += excess
            hard_violations += excess
            score += hard_penalty * excess
        if hr < min_hours:
            deficit_h = int(math.ceil((min_hours - hr) / max(1.0, min(problem.shift_hours.values()) if problem.shift_hours else 8.0)))
            contract_violations += deficit_h
            hard_violations += deficit_h
            score += hard_penalty * deficit_h
        if hr > max_hours:
            excess_h = int(math.ceil((hr - max_hours) / max(1.0, min(problem.shift_hours.values()) if problem.shift_hours else 8.0)))
            contract_violations += excess_h
            hard_violations += excess_h
            score += hard_penalty * excess_h
        if nights > max_night_shifts:
            excess = nights - max_night_shifts
            hard_violations += excess
            score += hard_penalty * excess

        fairness_penalty += fairness_weight * abs(wd - target_work_days)

        work_seq = [1 if state.get((e, d)) is not None else 0 for d in problem.days]
        night_seq = []
        for d in problem.days:
            val = state.get((e, d))
            if val is None:
                night_seq.append(0)
            else:
                night_seq.append(1 if val[0] in problem.night_shifts else 0)

        consec_v = count_consecutive_violations(work_seq, max_consec_work)
        if consec_v:
            consecutive_work_violations += consec_v
            hard_violations += consec_v
            score += w_consec * consec_v

        consec_n_v = count_consecutive_violations(night_seq, max_consec_night)
        if consec_n_v:
            consecutive_night_violations += consec_n_v
            hard_violations += consec_n_v
            score += w_consec_night * consec_n_v

        # Forbidden shift transitions.
        for d1, d2 in zip(problem.days[:-1], problem.days[1:]):
            s1 = employee_day_shift.get((e, d1))
            s2 = employee_day_shift.get((e, d2))
            if s1 is not None and s2 is not None and (s1, s2) in problem.forbidden_transitions:
                rest_violations += 1
                hard_violations += 1
                score += w_rest

    # Incompatibility in same day-shift.
    for e1, e2, p in problem.incompatible_pairs:
        for d in problem.days:
            for s in problem.shifts:
                emp_set = same_day_shift_employees.get((d, s), set())
                if e1 in emp_set and e2 in emp_set:
                    incompatibility_penalty += p if p else w_incompat

    score += preference_penalty
    score += standard_penalty
    score += incompatibility_penalty
    score += fairness_penalty

    metrics = {
        "score": score,
        "hard_violations": int(hard_violations),
        "skill_violations": int(skill_violations),
        "availability_violations": int(availability_violations),
        "fixed_assignment_violations": int(fixed_assignment_violations),
        "rest_violations": int(rest_violations),
        "contract_violations": int(contract_violations),
        "consecutive_work_violations": int(consecutive_work_violations),
        "consecutive_night_violations": int(consecutive_night_violations),
        "shortage_to_Cmin": int(shortage_to_cmin),
        "shortage_to_Copt": int(shortage_to_copt),
        "overstaff_above_Copt": int(overstaff_above_copt),
        "overstaff_above_Cmax": int(overstaff_above_cmax),
        "coverage_under_Cmin_cells": int(under_cmin_cells),
        "coverage_under_Copt_cells": int(under_copt_cells),
        "coverage_over_Cmax_cells": int(over_cmax_cells),
        "preference_penalty": float(preference_penalty),
        "standard_profile_penalty": float(standard_penalty),
        "incompatibility_penalty": float(incompatibility_penalty),
        "fairness_penalty": float(fairness_penalty),
        "actual_assignments": int(sum(1 for v in state.values() if v is not None)),
    }
    return EvaluationResult(score=score, metrics=metrics, coverage_counts=coverage_counts, workdays=workdays)


# ============================================================
# 7. NEIGHBORHOOD STRUCTURES
# ============================================================

def assigned_keys(state: State) -> List[Tuple[str, int]]:
    return [key for key, val in state.items() if val is not None]


def off_keys(problem: ProblemData, state: State) -> List[Tuple[str, int]]:
    return [key for key, val in state.items() if val is None and key[0] in problem.employees and key[1] in problem.days]


def coverage_cells_by_status(problem: ProblemData, eval_result: EvaluationResult) -> Tuple[List[Tuple[int, str, str]], List[Tuple[int, str, str]], List[Tuple[int, str, str]]]:
    over_opt = []
    under_opt = []
    over_min_or_opt = []
    for key, band in problem.coverage.items():
        actual = eval_result.coverage_counts.get(key, 0)
        if actual > band["C_opt"]:
            over_opt.append(key)
        if actual < band["C_opt"]:
            under_opt.append(key)
        if actual > band["C_min"]:
            over_min_or_opt.append(key)
    return over_opt, under_opt, over_min_or_opt


def propose_reassign_same_cell(problem: ProblemData, state: State, rng: random.Random) -> Optional[State]:
    assigned = assigned_keys(state)
    if not assigned:
        return None
    rng.shuffle(assigned)
    for e1, d in assigned[:50]:
        if is_fixed_employee_day(problem, e1, d):
            continue
        val = state[(e1, d)]
        if val is None:
            continue
        s, r = val
        candidates = [e2 for e2 in problem.employees if e2 != e1 and state.get((e2, d)) is None]
        rng.shuffle(candidates)
        for e2 in candidates[:50]:
            if can_assign(problem, e2, d, s, r):
                new_state = copy_state(state)
                new_state[(e1, d)] = None
                new_state[(e2, d)] = (s, r)
                return new_state
    return None


def propose_swap_assignments(problem: ProblemData, state: State, rng: random.Random) -> Optional[State]:
    assigned = assigned_keys(state)
    if len(assigned) < 2:
        return None
    for _ in range(80):
        k1, k2 = rng.sample(assigned, 2)
        e1, d1 = k1
        e2, d2 = k2
        if k1 == k2:
            continue
        if is_fixed_employee_day(problem, e1, d1) or is_fixed_employee_day(problem, e2, d2):
            continue
        v1 = state[k1]
        v2 = state[k2]
        if v1 is None or v2 is None:
            continue
        s1, r1 = v1
        s2, r2 = v2
        if can_assign(problem, e1, d1, s2, r2) and can_assign(problem, e2, d2, s1, r1):
            new_state = copy_state(state)
            new_state[k1] = (s2, r2)
            new_state[k2] = (s1, r1)
            return new_state
    return None


def propose_move_over_to_short(problem: ProblemData, state: State, eval_result: EvaluationResult, rng: random.Random) -> Optional[State]:
    over_opt, under_opt, _ = coverage_cells_by_status(problem, eval_result)
    if not over_opt or not under_opt:
        return None

    rng.shuffle(over_opt)
    rng.shuffle(under_opt)

    for source_cell in over_opt[:30]:
        sd, ss, sr = source_cell
        source_emps = [e for e in problem.employees if state.get((e, sd)) == (ss, sr) and not is_fixed_employee_day(problem, e, sd)]
        rng.shuffle(source_emps)
        for e in source_emps[:20]:
            for target_cell in under_opt[:40]:
                td, ts, tr = target_cell
                # Conservative small-scale move: keep same day so employee workday count is unchanged.
                if td != sd:
                    continue
                if can_assign(problem, e, td, ts, tr):
                    new_state = copy_state(state)
                    new_state[(e, sd)] = (ts, tr)
                    return new_state
    return None


def propose_drop_overstaff(problem: ProblemData, state: State, eval_result: EvaluationResult, rng: random.Random) -> Optional[State]:
    over_opt, _, _ = coverage_cells_by_status(problem, eval_result)
    if not over_opt:
        return None
    rng.shuffle(over_opt)
    for d, s, r in over_opt[:50]:
        employees_in_cell = [e for e in problem.employees if state.get((e, d)) == (s, r)]
        rng.shuffle(employees_in_cell)
        for e in employees_in_cell:
            if is_fixed_employee_day(problem, e, d):
                continue
            min_work_days = as_int(problem.employee_rule.get(e, {}).get("min_work_days", 0), 0)
            if eval_result.workdays.get(e, 0) <= min_work_days:
                continue
            # Do not drop if it goes below C_opt; by construction actual > C_opt, so safe for C_opt.
            new_state = copy_state(state)
            new_state[(e, d)] = None
            return new_state
    return None


def propose_add_to_shortage(problem: ProblemData, state: State, eval_result: EvaluationResult, rng: random.Random) -> Optional[State]:
    _, under_opt, _ = coverage_cells_by_status(problem, eval_result)
    if not under_opt:
        return None
    rng.shuffle(under_opt)
    for d, s, r in under_opt[:50]:
        # Do not exceed C_max.
        if eval_result.coverage_counts.get((d, s, r), 0) >= problem.coverage[(d, s, r)]["C_max"]:
            continue
        candidates = [e for e in problem.employees if state.get((e, d)) is None]
        rng.shuffle(candidates)
        for e in candidates[:80]:
            if not can_assign(problem, e, d, s, r):
                continue
            max_work_days = as_int(problem.employee_rule.get(e, {}).get("max_work_days", len(problem.days)), len(problem.days))
            if eval_result.workdays.get(e, 0) >= max_work_days:
                continue
            new_state = copy_state(state)
            new_state[(e, d)] = (s, r)
            return new_state
    return None


def propose_neighbor(problem: ProblemData, state: State, eval_result: EvaluationResult, rng: random.Random) -> Optional[State]:
    # Weighted neighborhood selection. Conservative settings for small scale.
    moves = [
        ("drop_overstaff", 0.40),
        ("move_over_to_short", 0.25),
        ("reassign_same_cell", 0.15),
        ("swap_assignments", 0.15),
        ("add_to_shortage", 0.05),
    ]
    names = [m[0] for m in moves]
    probs = [m[1] for m in moves]

    # Try selected move first, then fallback to others.
    selected = rng.choices(names, weights=probs, k=1)[0]
    ordered = [selected] + [n for n in names if n != selected]

    for name in ordered:
        if name == "reassign_same_cell":
            cand = propose_reassign_same_cell(problem, state, rng)
        elif name == "swap_assignments":
            cand = propose_swap_assignments(problem, state, rng)
        elif name == "move_over_to_short":
            cand = propose_move_over_to_short(problem, state, eval_result, rng)
        elif name == "drop_overstaff":
            cand = propose_drop_overstaff(problem, state, eval_result, rng)
        elif name == "add_to_shortage":
            cand = propose_add_to_shortage(problem, state, eval_result, rng)
        else:
            cand = None
        if cand is not None:
            return cand
    return None


# ============================================================
# 8. SIMULATED ANNEALING ENGINE
# ============================================================

def run_sa(
    problem: ProblemData,
    initial_state: State,
    initial_temp: float,
    min_temp: float,
    cooling_rate: float,
    cycles_per_temp: int,
    seed: int,
    hard_penalty: float,
    max_no_improve: int,
    log_every: int,
) -> Tuple[State, EvaluationResult, EvaluationResult, pd.DataFrame]:
    rng = random.Random(seed)

    current_state = copy_state(initial_state)
    current_eval = evaluate_state(problem, current_state, hard_penalty=hard_penalty)
    best_state = copy_state(current_state)
    best_eval = current_eval
    initial_eval = current_eval

    history = []
    T = float(initial_temp)
    iteration = 0
    accepted = 0
    rejected = 0
    no_improve = 0

    while T > min_temp:
        for _ in range(cycles_per_temp):
            iteration += 1
            neighbor = propose_neighbor(problem, current_state, current_eval, rng)
            if neighbor is None:
                rejected += 1
                continue

            neighbor_eval = evaluate_state(problem, neighbor, hard_penalty=hard_penalty)
            delta = neighbor_eval.score - current_eval.score

            accept = False
            if delta <= 0:
                accept = True
            else:
                prob_accept = math.exp(-delta / max(T, 1e-9))
                accept = rng.random() < prob_accept

            if accept:
                current_state = neighbor
                current_eval = neighbor_eval
                accepted += 1

                if current_eval.score < best_eval.score:
                    best_state = copy_state(current_state)
                    best_eval = current_eval
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                rejected += 1
                no_improve += 1

            if log_every > 0 and iteration % log_every == 0:
                row = {
                    "iteration": iteration,
                    "temperature": T,
                    "current_score": current_eval.score,
                    "best_score": best_eval.score,
                    "accepted": accepted,
                    "rejected": rejected,
                    **{f"current_{k}": v for k, v in current_eval.metrics.items() if k != "score"},
                    **{f"best_{k}": v for k, v in best_eval.metrics.items() if k != "score"},
                }
                history.append(row)

            if max_no_improve > 0 and no_improve >= max_no_improve:
                history.append({
                    "iteration": iteration,
                    "temperature": T,
                    "current_score": current_eval.score,
                    "best_score": best_eval.score,
                    "accepted": accepted,
                    "rejected": rejected,
                    "stop_reason": "max_no_improve",
                })
                return best_state, initial_eval, best_eval, pd.DataFrame(history)

        T *= cooling_rate

    history.append({
        "iteration": iteration,
        "temperature": T,
        "current_score": current_eval.score,
        "best_score": best_eval.score,
        "accepted": accepted,
        "rejected": rejected,
        "stop_reason": "temperature_below_min",
    })
    return best_state, initial_eval, best_eval, pd.DataFrame(history)


# ============================================================
# 9. OUTPUT BUILDERS
# ============================================================

def state_to_assignment_long(problem: ProblemData, state: State) -> pd.DataFrame:
    rows = []
    for (e, d), val in state.items():
        if val is None:
            continue
        s, r = val
        rows.append({
            "employee_id": e,
            "day": d,
            "shift": s,
            "shift_time": problem.shift_time.get(s, ""),
            "role": r,
            "assigned": 1,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["employee_id", "day", "shift", "shift_time", "role", "assigned"])
    return df.sort_values(["day", "shift", "role", "employee_id"]).reset_index(drop=True)


def build_roster_matrix(problem: ProblemData, state: State) -> pd.DataFrame:
    name_col = "employee_name" if "employee_name" in problem.employees_df.columns else "employee_id"
    name_map = dict(zip(problem.employees_df["employee_id"].astype(str), problem.employees_df[name_col].astype(str)))
    rows = []
    for e in problem.employees:
        row = {"employee_id": e, "employee_name": name_map.get(e, e)}
        for d in problem.days:
            val = state.get((e, d))
            if val is None:
                row[f"D{d}"] = "UNAVAILABLE" if (e, d) in problem.blocked_days else "OFF"
            else:
                s, r = val
                row[f"D{d}"] = f"{s}-{r}"
        rows.append(row)
    return pd.DataFrame(rows)


def build_coverage_summary(problem: ProblemData, eval_result: EvaluationResult) -> pd.DataFrame:
    rows = []
    for (d, s, r), band in sorted(problem.coverage.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        actual = int(eval_result.coverage_counts.get((d, s, r), 0))
        c_min = int(band["C_min"])
        c_opt = int(band["C_opt"])
        c_max = int(band["C_max"])
        rows.append({
            "day": d,
            "shift": s,
            "shift_time": problem.shift_time.get(s, ""),
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
    return pd.DataFrame(rows)


def build_employee_summary(problem: ProblemData, state: State) -> pd.DataFrame:
    rows = []
    name_col = "employee_name" if "employee_name" in problem.employees_df.columns else "employee_id"
    name_map = dict(zip(problem.employees_df["employee_id"].astype(str), problem.employees_df[name_col].astype(str)))
    for e in problem.employees:
        row = problem.employee_rule.get(e, {})
        shift_counts = {f"count_{s}": 0 for s in problem.shifts}
        role_counts = {f"count_{r}": 0 for r in problem.roles}
        work_days = 0
        hours = 0.0
        night_shifts = 0
        for d in problem.days:
            val = state.get((e, d))
            if val is None:
                continue
            s, r = val
            work_days += 1
            hours += problem.shift_hours.get(s, 8.0)
            night_shifts += int(problem.is_night_shift.get(s, 0))
            shift_counts[f"count_{s}"] += 1
            role_counts[f"count_{r}"] += 1
        rows.append({
            "employee_id": e,
            "employee_name": name_map.get(e, e),
            "work_days": work_days,
            "hours": hours,
            "night_shifts": night_shifts,
            "min_work_days": row.get("min_work_days", pd.NA),
            "max_work_days": row.get("max_work_days", pd.NA),
            "target_work_days": row.get("target_work_days", pd.NA),
            "min_hours": row.get("min_hours", pd.NA),
            "max_hours": row.get("max_hours", pd.NA),
            **shift_counts,
            **role_counts,
        })
    return pd.DataFrame(rows)


def build_status(initial_eval: EvaluationResult, best_eval: EvaluationResult, args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        {"metric": "algorithm", "value": "Simulated Annealing Phase 2"},
        {"metric": "scale", "value": "small"},
        {"metric": "initial_temp", "value": args.initial_temp},
        {"metric": "min_temp", "value": args.min_temp},
        {"metric": "cooling_rate", "value": args.cooling_rate},
        {"metric": "cycles_per_temp", "value": args.cycles_per_temp},
        {"metric": "seed", "value": args.seed},
        {"metric": "hard_penalty", "value": args.hard_penalty},
        {"metric": "initial_score", "value": initial_eval.score},
        {"metric": "final_score", "value": best_eval.score},
        {"metric": "score_improvement", "value": initial_eval.score - best_eval.score},
        {"metric": "score_improvement_percent", "value": 0.0 if initial_eval.score == 0 else (initial_eval.score - best_eval.score) / initial_eval.score * 100.0},
    ]
    for k, v in initial_eval.metrics.items():
        rows.append({"metric": f"initial_{k}", "value": v})
    for k, v in best_eval.metrics.items():
        rows.append({"metric": f"final_{k}", "value": v})
    return pd.DataFrame(rows)


def export_outputs(
    problem: ProblemData,
    best_state: State,
    initial_eval: EvaluationResult,
    best_eval: EvaluationResult,
    history_df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "sa_assignment_long": state_to_assignment_long(problem, best_state),
        "sa_roster_matrix": build_roster_matrix(problem, best_state),
        "sa_coverage_summary": build_coverage_summary(problem, best_eval),
        "sa_employee_summary": build_employee_summary(problem, best_state),
        "sa_model_status": build_status(initial_eval, best_eval, args),
        "sa_iteration_history": history_df,
        "sa_demand_input": problem.coverage_df,
    }

    for name, df in outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    xlsx_path = output_dir / "sa_small_scale_output.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for sheet, df in outputs.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    return xlsx_path


# ============================================================
# 10. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SA Phase 2 small-scale roster improvement from MILP baseline.")
    # Default folders are set for the current Small Scale capstone project.
    # You can still override them from PowerShell by passing --input-dir,
    # --milp-output-dir, and --output-dir manually.
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"E:\NĂM 4\Capstone\Data_input\employee_milp_inputs"),
        help="Folder containing employee_milp_inputs CSV files."
    )
    parser.add_argument(
        "--milp-output-dir",
        type=Path,
        default=Path(r"E:\NĂM 4\Capstone\MILP_Model\MILP_Result"),
        help="Folder containing MILP assignment_long.csv or milp_roster_output.xlsx."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\NĂM 4\Capstone\MILP_Model\SA_Result_small"),
        help="Folder for SA outputs."
    )

    parser.add_argument("--initial-temp", type=float, default=500.0, help="Initial SA temperature T0.")
    parser.add_argument("--min-temp", type=float, default=0.1, help="Minimum stopping temperature Tmin.")
    parser.add_argument("--cooling-rate", type=float, default=0.95, help="Cooling rate beta; new T = beta*T.")
    parser.add_argument("--cycles-per-temp", type=int, default=700, help="Number of neighbor trials at each temperature.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hard-penalty", type=float, default=DEFAULT_HARD_PENALTY, help="Penalty for hard feasibility violations.")
    parser.add_argument("--max-no-improve", type=int, default=0, help="Optional early stop after this many non-improving iterations; 0 disables.")
    parser.add_argument("--log-every", type=int, default=200, help="Save iteration history every N iterations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("SA Phase 2 - Small Scale Roster Improvement")
    print(f"Input dir:       {args.input_dir}")
    print(f"MILP output dir: {args.milp_output_dir}")
    print(f"SA output dir:   {args.output_dir}")
    print()

    print("[1/5] Loading MILP-compatible input data...")
    problem, initial_assignment_df = load_problem(args.input_dir, args.milp_output_dir)
    print(f"      Employees: {len(problem.employees)}")
    print(f"      Days: {len(problem.days)}")
    print(f"      Shifts: {len(problem.shifts)}")
    print(f"      Roles: {len(problem.roles)}")
    print(f"      Initial MILP assignments: {len(initial_assignment_df)}")

    print("[2/5] Building initial state S0 from MILP roster...")
    initial_state = build_initial_state(problem, initial_assignment_df)
    initial_eval = evaluate_state(problem, initial_state, hard_penalty=args.hard_penalty)
    print(f"      Initial common score: {initial_eval.score:.2f}")
    print(f"      Initial hard violations: {initial_eval.metrics['hard_violations']}")
    print(f"      Initial overstaff above Copt: {initial_eval.metrics['overstaff_above_Copt']}")
    print(f"      Initial preference penalty: {initial_eval.metrics['preference_penalty']:.2f}")

    print("[3/5] Running Simulated Annealing improvement...")
    best_state, initial_eval, best_eval, history_df = run_sa(
        problem=problem,
        initial_state=initial_state,
        initial_temp=args.initial_temp,
        min_temp=args.min_temp,
        cooling_rate=args.cooling_rate,
        cycles_per_temp=args.cycles_per_temp,
        seed=args.seed,
        hard_penalty=args.hard_penalty,
        max_no_improve=args.max_no_improve,
        log_every=args.log_every,
    )

    print("[4/5] Exporting outputs...")
    xlsx_path = export_outputs(problem, best_state, initial_eval, best_eval, history_df, args.output_dir, args)

    print("[5/5] DONE")
    print(f"      Initial score: {initial_eval.score:.2f}")
    print(f"      Final score:   {best_eval.score:.2f}")
    print(f"      Improvement:   {initial_eval.score - best_eval.score:.2f}")
    pct = 0.0 if initial_eval.score == 0 else (initial_eval.score - best_eval.score) / initial_eval.score * 100.0
    print(f"      Improvement %: {pct:.2f}%")
    print(f"      Final hard violations: {best_eval.metrics['hard_violations']}")
    print(f"      Final overstaff above Copt: {best_eval.metrics['overstaff_above_Copt']}")
    print(f"      Final preference penalty: {best_eval.metrics['preference_penalty']:.2f}")
    print(f"      Excel output: {xlsx_path}")
    print(f"      CSV outputs:  {args.output_dir}")


if __name__ == "__main__":
    main()
