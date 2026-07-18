#!/usr/bin/env python3
r"""
sa_phase2_medium.py

Simulated Annealing Phase 2 for MEDIUM scale airport lounge rostering.

Research logic:
  MILP roster = baseline solution S0
  SA = post-optimization improvement/refinement model

This script reads:
  - Scaled MILP input folder, e.g. employee_milp_inputs_medium / large
  - MILP output folder containing assignment_long.csv

It outputs:
  - sa_assignment_long.csv
  - sa_roster_matrix.csv
  - sa_coverage_summary.csv
  - sa_employee_summary.csv
  - sa_model_status.csv
  - sa_iteration_history.csv
  - sa_phase2_output.xlsx

Example:
python sa_phase2_medium.py ^
  --input-dir "E:\NĂM 4\Capstone\SA_Improvement\employee_milp_inputs_medium" ^
  --milp-output-dir "E:\NĂM 4\Capstone\SA_Improvement\MILP_Result_medium" ^
  --output-dir "E:\NĂM 4\Capstone\SA_Improvement\SA_Result_medium" ^
  --max-seconds 300
"""
from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

ROLES = ["RS", "DAS", "BLO", "FSTC", "DS", "SLS"]
SHIFTS = ["S1", "S2", "S3"]
SHIFT_TIME = {"S1": "00:00-08:00", "S2": "08:00-16:00", "S3": "16:00-24:00"}

# Fixed MEDIUM default paths. You can still override them from PowerShell if needed.
DEFAULT_INPUT_DIR = Path(r"E:\NĂM 4\Capstone\SA_Improvement\employee_milp_inputs_medium")
DEFAULT_MILP_OUTPUT_DIR = Path(r"E:\NĂM 4\Capstone\SA_Improvement\MILP_Result_medium")
DEFAULT_OUTPUT_DIR = Path(r"E:\NĂM 4\Capstone\SA_Improvement\SA_Result_medium")

Assignment = Tuple[str, int, str, str]  # employee_id, day, shift, role
Cell = Tuple[int, str, str]             # day, shift, role


@dataclass
class SAParams:
    initial_temp: float
    min_temp: float
    cooling_rate: float
    cycles_per_temp: int
    max_iterations: int
    max_seconds: float
    seed: int
    hard_penalty: float


PRESET_PARAMS = {
    "medium": SAParams(
        initial_temp=750.0,
        min_temp=0.25,
        cooling_rate=0.95,
        cycles_per_temp=1200,
        max_iterations=120000,
        max_seconds=300.0,
        seed=42,
        hard_penalty=100000.0,
    ),
    "large": SAParams(
        initial_temp=1200.0,
        min_temp=0.25,
        cooling_rate=0.97,
        cycles_per_temp=2000,
        max_iterations=250000,
        max_seconds=0.0,
        seed=42,
        hard_penalty=100000.0,
    ),
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def read_csv_required(folder: Path, filename: str) -> pd.DataFrame:
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def read_csv_optional(folder: Path, filename: str, columns: Sequence[str]) -> pd.DataFrame:
    path = folder / filename
    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    df = pd.read_csv(path, encoding="utf-8-sig")
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df[list(columns)]


def as_int(v, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return default
        return int(float(v))
    except Exception:
        return default


def as_float(v, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def normalize_shift(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().upper().replace("SHIFT", "").replace("_", "").replace("-", "")
    if s in {"1", "01"}: return "S1"
    if s in {"2", "02"}: return "S2"
    if s in {"3", "03"}: return "S3"
    return s


def normalize_pref_type(v) -> str:
    return str(v).strip().lower().replace(" ", "_")


def safe_exp(x: float) -> float:
    if x < -700:
        return 0.0
    if x > 700:
        return float("inf")
    return math.exp(x)


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------

class InputData:
    def __init__(self, input_dir: Path, milp_output_dir: Path):
        self.input_dir = input_dir
        self.milp_output_dir = milp_output_dir

        self.employees_df = read_csv_required(input_dir, "employee_master.csv")
        self.skills_df = read_csv_required(input_dir, "employee_skills.csv")
        self.coverage_df = read_csv_required(input_dir, "staffing_coverage_bands_long.csv")
        self.shifts_df = read_csv_required(input_dir, "shift_structure.csv")
        self.rest_df = read_csv_required(input_dir, "shift_transition_rest.csv")
        self.penalty_df = read_csv_optional(input_dir, "penalty_config.csv", ["penalty_component", "weight"])

        self.availability_df = read_csv_optional(input_dir, "employee_availability.csv", ["employee_id", "day", "status"])
        self.day_requests_df = read_csv_optional(input_dir, "employee_day_requests.csv", ["employee_id", "day", "request_type", "penalty", "is_fixed"])
        self.shift_preferences_df = read_csv_optional(input_dir, "employee_preferences.csv", ["employee_id", "day", "shift", "preference_type", "penalty"])
        self.assignment_preferences_df = read_csv_optional(input_dir, "employee_assignment_preferences.csv", ["employee_id", "day", "shift", "role", "preference_type", "penalty"])
        self.fixed_assignments_df = read_csv_optional(input_dir, "employee_fixed_assignments.csv", ["employee_id", "day", "shift", "role"])
        self.standard_profile_df = read_csv_optional(input_dir, "employee_standard_profile.csv", ["employee_id", "standard_role", "standard_shift"])
        self.incompatibility_df = read_csv_optional(input_dir, "employee_incompatibility.csv", ["employee_i", "employee_j", "incompatibility_type", "priority", "penalty"])

        self.assignment_initial_df = read_csv_required(milp_output_dir, "assignment_long.csv")

        self._standardize()
        self._build_dictionaries()

    def _standardize(self) -> None:
        self.employees_df["employee_id"] = self.employees_df["employee_id"].astype(str)
        self.skills_df["employee_id"] = self.skills_df["employee_id"].astype(str)
        self.coverage_df["day"] = self.coverage_df["day"].apply(as_int)
        self.coverage_df["shift"] = self.coverage_df["shift"].apply(normalize_shift)
        self.coverage_df["role"] = self.coverage_df["role"].astype(str).str.strip().str.upper()
        self.coverage_df = self.coverage_df[self.coverage_df["role"].isin(ROLES)].copy()
        if "shift_time" not in self.coverage_df.columns:
            self.coverage_df["shift_time"] = self.coverage_df["shift"].map(SHIFT_TIME)
        for c in ["C_min", "C_opt", "C_max"]:
            self.coverage_df[c] = self.coverage_df[c].apply(as_int)

        self.shifts_df["shift"] = self.shifts_df["shift"].apply(normalize_shift)
        if "duration_hours" not in self.shifts_df.columns:
            self.shifts_df["duration_hours"] = 8.0
        if "is_night_shift" not in self.shifts_df.columns:
            self.shifts_df["is_night_shift"] = self.shifts_df["shift"].apply(lambda s: 1 if s == "S1" else 0)

        for df in [self.availability_df, self.day_requests_df, self.shift_preferences_df, self.assignment_preferences_df, self.fixed_assignments_df]:
            if "employee_id" in df.columns:
                df["employee_id"] = df["employee_id"].astype(str)
            if "day" in df.columns:
                df["day"] = df["day"].apply(as_int)
            if "shift" in df.columns:
                df["shift"] = df["shift"].apply(normalize_shift)
            if "role" in df.columns:
                df["role"] = df["role"].astype(str).str.strip().str.upper()

        for c in ["employee_i", "employee_j"]:
            if c in self.incompatibility_df.columns:
                self.incompatibility_df[c] = self.incompatibility_df[c].astype(str)

        if not self.standard_profile_df.empty:
            self.standard_profile_df["employee_id"] = self.standard_profile_df["employee_id"].astype(str)
            self.standard_profile_df["standard_role"] = self.standard_profile_df["standard_role"].astype(str).str.strip().str.upper()
            self.standard_profile_df["standard_shift"] = self.standard_profile_df["standard_shift"].apply(normalize_shift)

        # Initial MILP assignment format.
        required = {"employee_id", "day", "shift", "role"}
        missing = required - set(self.assignment_initial_df.columns)
        if missing:
            raise ValueError(f"assignment_long.csv missing columns: {missing}")
        self.assignment_initial_df["employee_id"] = self.assignment_initial_df["employee_id"].astype(str)
        self.assignment_initial_df["day"] = self.assignment_initial_df["day"].apply(as_int)
        self.assignment_initial_df["shift"] = self.assignment_initial_df["shift"].apply(normalize_shift)
        self.assignment_initial_df["role"] = self.assignment_initial_df["role"].astype(str).str.strip().str.upper()
        self.assignment_initial_df = self.assignment_initial_df[
            self.assignment_initial_df["role"].isin(ROLES)
        ][["employee_id", "day", "shift", "role"]].drop_duplicates().reset_index(drop=True)

    def _build_dictionaries(self) -> None:
        self.employees: List[str] = self.employees_df["employee_id"].astype(str).tolist()
        self.employee_set: Set[str] = set(self.employees)
        self.days: List[int] = sorted(self.coverage_df["day"].unique().tolist())
        self.shifts: List[str] = [s for s in self.shifts_df["shift"].tolist() if s]
        if not self.shifts:
            self.shifts = SHIFTS
        self.roles: List[str] = ROLES

        self.shift_time = dict(zip(self.shifts_df["shift"], self.shifts_df.get("shift_time", self.shifts_df["shift"].map(SHIFT_TIME))))
        self.shift_hours = dict(zip(self.shifts_df["shift"], self.shifts_df["duration_hours"].apply(as_float)))
        self.is_night_shift = dict(zip(self.shifts_df["shift"], self.shifts_df["is_night_shift"].apply(as_int)))

        self.emp_row = self.employees_df.set_index("employee_id").to_dict("index")
        self.skill_row = self.skills_df.set_index("employee_id").to_dict("index")
        self.coverage = self.coverage_df.set_index(["day", "shift", "role"]).to_dict("index")
        self.cells: List[Cell] = sorted(list(self.coverage.keys()))

        self.blocked_day: Set[Tuple[str, int]] = set()
        for _, row in self.availability_df.iterrows():
            status = normalize_pref_type(row.get("status", ""))
            if status and status not in {"available", "work", "working", "ok", "nan"}:
                self.blocked_day.add((str(row["employee_id"]), as_int(row["day"])))
        for _, row in self.day_requests_df.iterrows():
            req = normalize_pref_type(row.get("request_type", ""))
            if as_int(row.get("is_fixed", 0)) == 1 and req in {"desired_day_off", "day_off", "leave", "fixed_day_off", "unavailable"}:
                self.blocked_day.add((str(row["employee_id"]), as_int(row["day"])))

        self.fixed_assignments: Set[Assignment] = set()
        for _, row in self.fixed_assignments_df.iterrows():
            e, d, s, r = str(row["employee_id"]), as_int(row["day"]), normalize_shift(row["shift"]), str(row["role"]).upper()
            if e in self.employee_set and r in ROLES:
                self.fixed_assignments.add((e, d, s, r))

        # Preference penalty dictionaries.
        self.shift_avoid_penalty: Dict[Tuple[str, int, str], float] = defaultdict(float)
        self.shift_prefer_bonus: Dict[Tuple[str, int, str], float] = defaultdict(float)
        for _, row in self.shift_preferences_df.iterrows():
            key = (str(row["employee_id"]), as_int(row["day"]), normalize_shift(row["shift"]))
            typ = normalize_pref_type(row.get("preference_type", ""))
            penalty = as_float(row.get("penalty"), 10.0)
            if typ in {"avoid", "undesired", "not_preferred", "prefer_not", "day_off"}:
                self.shift_avoid_penalty[key] += penalty
            elif typ in {"prefer", "preferred", "desired"}:
                self.shift_prefer_bonus[key] += min(penalty, 10.0)

        self.assignment_avoid_penalty: Dict[Assignment, float] = defaultdict(float)
        for _, row in self.assignment_preferences_df.iterrows():
            key = (str(row["employee_id"]), as_int(row["day"]), normalize_shift(row["shift"]), str(row["role"]).upper())
            typ = normalize_pref_type(row.get("preference_type", ""))
            penalty = as_float(row.get("penalty"), 10.0)
            if typ in {"avoid", "undesired", "not_preferred", "prefer_not"}:
                self.assignment_avoid_penalty[key] += penalty

        self.day_avoid_penalty: Dict[Tuple[str, int], float] = defaultdict(float)
        for _, row in self.day_requests_df.iterrows():
            key = (str(row["employee_id"]), as_int(row["day"]))
            typ = normalize_pref_type(row.get("request_type", ""))
            penalty = as_float(row.get("penalty"), 10.0)
            if typ in {"desired_day_off", "day_off", "leave", "fixed_day_off", "unavailable"}:
                self.day_avoid_penalty[key] += penalty

        self.standard_role: Dict[str, str] = {}
        self.standard_shift: Dict[str, str] = {}
        if not self.standard_profile_df.empty:
            for _, row in self.standard_profile_df.iterrows():
                e = str(row["employee_id"])
                if row.get("standard_role") in ROLES:
                    self.standard_role[e] = row.get("standard_role")
                sh = normalize_shift(row.get("standard_shift"))
                if sh:
                    self.standard_shift[e] = sh
        # Fallback to employee_master standard columns if available.
        for e, row in self.emp_row.items():
            if e not in self.standard_role and str(row.get("standard_role", "")).upper() in ROLES:
                self.standard_role[e] = str(row.get("standard_role")).upper()
            if e not in self.standard_shift and normalize_shift(row.get("standard_shift", "")):
                self.standard_shift[e] = normalize_shift(row.get("standard_shift"))

        self.incompat_pairs: Dict[Tuple[str, str], float] = {}
        for _, row in self.incompatibility_df.iterrows():
            a, b = str(row.get("employee_i")), str(row.get("employee_j"))
            if not a or not b or a == "nan" or b == "nan":
                continue
            pair = tuple(sorted([a, b]))
            self.incompat_pairs[pair] = max(self.incompat_pairs.get(pair, 0.0), as_float(row.get("penalty"), 50.0))

        self.forbidden_transitions: Set[Tuple[str, str]] = set()
        for _, row in self.rest_df.iterrows():
            s_from = normalize_shift(row.get("from_shift"))
            s_to = normalize_shift(row.get("to_shift_next_day"))
            if not s_from or not s_to:
                continue
            if as_int(row.get("allowed", 1), 1) == 0 or as_int(row.get("is_forbidden_successession", 0), 0) == 1:
                self.forbidden_transitions.add((s_from, s_to))

    def c_value(self, cell: Cell, col: str) -> int:
        return as_int(self.coverage.get(cell, {}).get(col, 0), 0)

    def is_qualified(self, e: str, role: str) -> bool:
        return as_int(self.skill_row.get(e, {}).get(role, 0), 0) == 1

    def is_blocked(self, e: str, day: int) -> bool:
        return (e, day) in self.blocked_day


# -----------------------------------------------------------------------------
# State and scoring
# -----------------------------------------------------------------------------

class SAState:
    def __init__(self, assignments: Iterable[Assignment]):
        self.assignments: Set[Assignment] = set(assignments)

    def copy(self) -> "SAState":
        return SAState(set(self.assignments))

    def key(self) -> frozenset:
        return frozenset(self.assignments)


@dataclass
class ScoreResult:
    score: float
    hard_violations: float
    total_shortage_cmin: int
    total_shortage_copt: int
    total_over_copt: int
    total_over_cmax: int
    preference_penalty: float
    fairness_penalty: float
    stability_penalty: float
    rest_violations: int
    skill_violations: int
    availability_violations: int
    one_per_day_violations: int
    contract_violations: float
    fixed_assignment_violations: int
    incompatibility_penalty: float


def build_counts(state: SAState):
    cov = Counter()
    emp_day = Counter()
    emp_workdays = Counter()
    emp_shift_counts = Counter()
    cell_emp = defaultdict(list)
    for e, d, s, r in state.assignments:
        cov[(d, s, r)] += 1
        emp_day[(e, d)] += 1
        emp_workdays[e] += 1
        emp_shift_counts[(e, s)] += 1
        cell_emp[(d, s, r)].append(e)
    return cov, emp_day, emp_workdays, emp_shift_counts, cell_emp


def evaluate(state: SAState, data: InputData, hard_penalty: float = 100000.0) -> ScoreResult:
    cov, emp_day, emp_workdays, _, cell_emp = build_counts(state)

    total_shortage_cmin = total_shortage_copt = total_over_copt = total_over_cmax = 0
    for cell in data.cells:
        actual = cov[cell]
        cmin = data.c_value(cell, "C_min")
        copt = data.c_value(cell, "C_opt")
        cmax = data.c_value(cell, "C_max")
        total_shortage_cmin += max(0, cmin - actual)
        total_shortage_copt += max(0, copt - actual)
        total_over_copt += max(0, actual - copt)
        total_over_cmax += max(0, actual - cmax)

    skill_violations = 0
    availability_violations = 0
    preference_penalty = 0.0
    stability_penalty = 0.0
    fixed_assignment_violations = 0

    for a in state.assignments:
        e, d, s, r = a
        if e not in data.employee_set or not data.is_qualified(e, r):
            skill_violations += 1
        if data.is_blocked(e, d):
            availability_violations += 1
        preference_penalty += data.day_avoid_penalty.get((e, d), 0.0)
        preference_penalty += data.shift_avoid_penalty.get((e, d, s), 0.0)
        preference_penalty += data.assignment_avoid_penalty.get(a, 0.0)
        # small reward for satisfying positive shift preference, clipped by zero score floor later indirectly
        preference_penalty -= data.shift_prefer_bonus.get((e, d, s), 0.0) * 0.25
        if data.standard_role.get(e) and data.standard_role[e] != r:
            stability_penalty += 1.0
        if data.standard_shift.get(e) and data.standard_shift[e] != s:
            stability_penalty += 0.5

    for fixed in data.fixed_assignments:
        if fixed not in state.assignments:
            fixed_assignment_violations += 1

    one_per_day_violations = sum(max(0, cnt - 1) for cnt in emp_day.values())

    # Contract violations and fairness.
    contract_violations = 0.0
    fairness_penalty = 0.0
    workday_values = []
    night_values = []
    for e in data.employees:
        row = data.emp_row.get(e, {})
        wd = int(emp_workdays.get(e, 0))
        workday_values.append(wd)
        min_wd = as_int(row.get("min_work_days", 0), 0)
        max_wd = as_int(row.get("max_work_days", max(data.days)), max(data.days))
        target_wd = as_float(row.get("target_work_days", (min_wd + max_wd) / 2), (min_wd + max_wd) / 2)
        if wd < min_wd:
            contract_violations += (min_wd - wd)
        if wd > max_wd:
            contract_violations += (wd - max_wd)
        fairness_penalty += abs(wd - target_wd)
        night_count = sum(1 for a in state.assignments if a[0] == e and data.is_night_shift.get(a[2], 0) == 1)
        night_values.append(night_count)
        max_night = as_int(row.get("max_night_shifts", 9999), 9999)
        if night_count > max_night:
            contract_violations += (night_count - max_night)

    if workday_values:
        fairness_penalty += float(np.std(workday_values)) * len(workday_values) * 0.10
    if night_values:
        fairness_penalty += float(np.std(night_values)) * len(night_values) * 0.10

    # Rest violations.
    by_emp_day_shift: Dict[Tuple[str, int], str] = {}
    for e, d, s, r in state.assignments:
        if (e, d) not in by_emp_day_shift:
            by_emp_day_shift[(e, d)] = s
    rest_violations = 0
    days_set = set(data.days)
    for e in data.employees:
        for d in data.days:
            if d + 1 not in days_set:
                continue
            s1 = by_emp_day_shift.get((e, d))
            s2 = by_emp_day_shift.get((e, d + 1))
            if s1 and s2 and (s1, s2) in data.forbidden_transitions:
                rest_violations += 1

    incompatibility_penalty = 0.0
    for cell, emps in cell_emp.items():
        if len(emps) < 2:
            continue
        emps_sorted = sorted(set(emps))
        for i in range(len(emps_sorted)):
            for j in range(i + 1, len(emps_sorted)):
                pair = (emps_sorted[i], emps_sorted[j])
                incompatibility_penalty += data.incompat_pairs.get(pair, 0.0)

    hard_violations = (
        total_shortage_cmin + total_over_cmax + skill_violations + availability_violations
        + one_per_day_violations + contract_violations + rest_violations + fixed_assignment_violations
    )

    score = (
        hard_penalty * hard_violations
        + 1000.0 * total_shortage_cmin
        + 500.0 * total_shortage_copt
        + 300.0 * total_over_cmax
        + 50.0 * total_over_copt
        + 20.0 * max(0.0, preference_penalty)
        + 15.0 * fairness_penalty
        + 10.0 * stability_penalty
        + incompatibility_penalty
    )

    return ScoreResult(
        score=score,
        hard_violations=hard_violations,
        total_shortage_cmin=total_shortage_cmin,
        total_shortage_copt=total_shortage_copt,
        total_over_copt=total_over_copt,
        total_over_cmax=total_over_cmax,
        preference_penalty=max(0.0, preference_penalty),
        fairness_penalty=fairness_penalty,
        stability_penalty=stability_penalty,
        rest_violations=rest_violations,
        skill_violations=skill_violations,
        availability_violations=availability_violations,
        one_per_day_violations=one_per_day_violations,
        contract_violations=contract_violations,
        fixed_assignment_violations=fixed_assignment_violations,
        incompatibility_penalty=incompatibility_penalty,
    )


# -----------------------------------------------------------------------------
# Neighborhood moves
# -----------------------------------------------------------------------------

def employee_assigned_day(state: SAState, e: str, d: int) -> bool:
    return any(a[0] == e and a[1] == d for a in state.assignments)


def can_assign_basic(data: InputData, state: SAState, e: str, d: int, s: str, r: str, ignore_assignment: Optional[Assignment] = None) -> bool:
    if e not in data.employee_set:
        return False
    if not data.is_qualified(e, r):
        return False
    if data.is_blocked(e, d):
        return False
    for a in state.assignments:
        if ignore_assignment is not None and a == ignore_assignment:
            continue
        if a[0] == e and a[1] == d:
            return False
    return True


def random_employee_for_cell(data: InputData, state: SAState, cell: Cell, rng: random.Random, ignore_assignment: Optional[Assignment] = None) -> Optional[str]:
    d, s, r = cell
    candidates = [e for e in data.employees if can_assign_basic(data, state, e, d, s, r, ignore_assignment=ignore_assignment)]
    if not candidates:
        return None
    return rng.choice(candidates)


def propose_reassign_same_cell(data: InputData, state: SAState, rng: random.Random) -> Optional[SAState]:
    if not state.assignments:
        return None
    a = rng.choice(tuple(state.assignments))
    e, d, s, r = a
    # Do not move fixed assignment away.
    if a in data.fixed_assignments:
        return None
    replacement = random_employee_for_cell(data, state, (d, s, r), rng, ignore_assignment=a)
    if not replacement or replacement == e:
        return None
    new = state.copy()
    new.assignments.remove(a)
    new.assignments.add((replacement, d, s, r))
    return new


def propose_swap_assignments(data: InputData, state: SAState, rng: random.Random) -> Optional[SAState]:
    if len(state.assignments) < 2:
        return None
    a1, a2 = rng.sample(tuple(state.assignments), 2)
    if a1 in data.fixed_assignments or a2 in data.fixed_assignments:
        return None
    e1, d1, s1, r1 = a1
    e2, d2, s2, r2 = a2
    if e1 == e2:
        return None
    # Swap employees, keep cells.
    if not data.is_qualified(e1, r2) or data.is_blocked(e1, d2):
        return None
    if not data.is_qualified(e2, r1) or data.is_blocked(e2, d1):
        return None
    # avoid double assignment after swap unless swapping same day assignments
    tmp = state.copy()
    tmp.assignments.remove(a1)
    tmp.assignments.remove(a2)
    if employee_assigned_day(tmp, e1, d2) or employee_assigned_day(tmp, e2, d1):
        return None
    tmp.assignments.add((e1, d2, s2, r2))
    tmp.assignments.add((e2, d1, s1, r1))
    return tmp


def find_shortage_cells(data: InputData, state: SAState, target: str = "C_opt") -> List[Cell]:
    cov, *_ = build_counts(state)
    cells = []
    for cell in data.cells:
        if cov[cell] < data.c_value(cell, target):
            cells.append(cell)
    return cells


def find_overstaff_cells(data: InputData, state: SAState, target: str = "C_opt") -> List[Cell]:
    cov, *_ = build_counts(state)
    cells = []
    for cell in data.cells:
        if cov[cell] > data.c_value(cell, target):
            cells.append(cell)
    return cells


def propose_add_to_shortage(data: InputData, state: SAState, rng: random.Random) -> Optional[SAState]:
    shortage_cells = find_shortage_cells(data, state, "C_opt")
    if not shortage_cells:
        return None
    cell = rng.choice(shortage_cells)
    d, s, r = cell
    e = random_employee_for_cell(data, state, cell, rng)
    if not e:
        return None
    new = state.copy()
    new.assignments.add((e, d, s, r))
    return new


def propose_drop_overstaff(data: InputData, state: SAState, rng: random.Random) -> Optional[SAState]:
    over_cells = find_overstaff_cells(data, state, "C_opt")
    if not over_cells:
        return None
    cell = rng.choice(over_cells)
    d, s, r = cell
    candidates = [a for a in state.assignments if a[1:] == (d, s, r) and a not in data.fixed_assignments]
    if not candidates:
        return None
    cov, emp_day, emp_workdays, *_ = build_counts(state)
    # Prefer dropping employees above min workdays.
    feasible = []
    for a in candidates:
        e = a[0]
        min_wd = as_int(data.emp_row.get(e, {}).get("min_work_days", 0), 0)
        if emp_workdays[e] > min_wd:
            feasible.append(a)
    if not feasible:
        feasible = candidates
    a = rng.choice(feasible)
    new = state.copy()
    new.assignments.remove(a)
    return new


def propose_move_over_to_short(data: InputData, state: SAState, rng: random.Random) -> Optional[SAState]:
    over_cells = find_overstaff_cells(data, state, "C_opt")
    short_cells = find_shortage_cells(data, state, "C_opt")
    if not over_cells or not short_cells:
        return None
    from_cell = rng.choice(over_cells)
    to_cell = rng.choice(short_cells)
    d1, s1, r1 = from_cell
    d2, s2, r2 = to_cell
    candidates = [a for a in state.assignments if a[1:] == from_cell and a not in data.fixed_assignments]
    rng.shuffle(candidates)
    for a in candidates[:50]:
        e = a[0]
        if can_assign_basic(data, state, e, d2, s2, r2, ignore_assignment=a):
            new = state.copy()
            new.assignments.remove(a)
            new.assignments.add((e, d2, s2, r2))
            return new
    return None


def propose_role_change_same_day(data: InputData, state: SAState, rng: random.Random) -> Optional[SAState]:
    """Change role/shift cell for same employee on the same day if it helps shortage."""
    if not state.assignments:
        return None
    a = rng.choice(tuple(state.assignments))
    if a in data.fixed_assignments:
        return None
    e, d, s, r = a
    same_day_short = [cell for cell in find_shortage_cells(data, state, "C_opt") if cell[0] == d]
    if not same_day_short:
        return None
    rng.shuffle(same_day_short)
    for cell in same_day_short[:20]:
        d2, s2, r2 = cell
        if can_assign_basic(data, state, e, d2, s2, r2, ignore_assignment=a):
            new = state.copy()
            new.assignments.remove(a)
            new.assignments.add((e, d2, s2, r2))
            return new
    return None


def propose_neighbor(data: InputData, state: SAState, rng: random.Random) -> Optional[SAState]:
    # Move weights tuned for medium/large: many-preserving moves first, then add/drop.
    moves = [
        (0.30, propose_reassign_same_cell),
        (0.20, propose_swap_assignments),
        (0.20, propose_move_over_to_short),
        (0.15, propose_role_change_same_day),
        (0.10, propose_drop_overstaff),
        (0.05, propose_add_to_shortage),
    ]
    u = rng.random()
    cum = 0.0
    chosen = moves[-1][1]
    for w, fn in moves:
        cum += w
        if u <= cum:
            chosen = fn
            break
    # Try chosen move first; if it fails, try alternatives.
    for fn in [chosen] + [m[1] for m in moves if m[1] is not chosen]:
        nb = fn(data, state, rng)
        if nb is not None and nb.assignments != state.assignments:
            return nb
    return None


# -----------------------------------------------------------------------------
# SA engine
# -----------------------------------------------------------------------------

def initial_state_from_milp(data: InputData) -> SAState:
    assignments = []
    for _, row in data.assignment_initial_df.iterrows():
        assignments.append((str(row["employee_id"]), as_int(row["day"]), normalize_shift(row["shift"]), str(row["role"]).upper()))
    # Ensure fixed assignments are present.
    for a in data.fixed_assignments:
        assignments.append(a)
    return SAState(assignments)


def run_sa(data: InputData, params: SAParams) -> Tuple[SAState, ScoreResult, ScoreResult, pd.DataFrame]:
    rng = random.Random(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    current = initial_state_from_milp(data)
    current_score = evaluate(current, data, hard_penalty=params.hard_penalty)
    best = current.copy()
    best_score = current_score
    initial_score = current_score

    T = params.initial_temp
    iteration = 0
    start = time.time()
    history = []

    while T > params.min_temp and iteration < params.max_iterations:
        for _ in range(params.cycles_per_temp):
            if iteration >= params.max_iterations:
                break
            if params.max_seconds and (time.time() - start) >= params.max_seconds:
                break
            iteration += 1
            neighbor = propose_neighbor(data, current, rng)
            if neighbor is None:
                continue
            neighbor_score = evaluate(neighbor, data, hard_penalty=params.hard_penalty)
            delta = neighbor_score.score - current_score.score
            accept = delta <= 0 or rng.random() < safe_exp(-delta / max(T, 1e-9))
            if accept:
                current = neighbor
                current_score = neighbor_score
                if current_score.score < best_score.score:
                    best = current.copy()
                    best_score = current_score

        history.append({
            "iteration": iteration,
            "temperature": T,
            "current_score": current_score.score,
            "best_score": best_score.score,
            "best_hard_violations": best_score.hard_violations,
            "best_shortage_Copt": best_score.total_shortage_copt,
            "best_over_Copt": best_score.total_over_copt,
            "elapsed_seconds": round(time.time() - start, 3),
        })
        if params.max_seconds and (time.time() - start) >= params.max_seconds:
            break
        T *= params.cooling_rate

    return best, initial_score, best_score, pd.DataFrame(history)


# -----------------------------------------------------------------------------
# Exporting results
# -----------------------------------------------------------------------------

def assignment_df_from_state(state: SAState, data: InputData) -> pd.DataFrame:
    rows = []
    name_map = {}
    if "employee_name" in data.employees_df.columns:
        name_map = dict(zip(data.employees_df["employee_id"].astype(str), data.employees_df["employee_name"].astype(str)))
    for e, d, s, r in sorted(state.assignments, key=lambda x: (x[1], x[2], x[3], x[0])):
        rows.append({
            "employee_id": e,
            "employee_name": name_map.get(e, e),
            "day": d,
            "shift": s,
            "shift_time": data.shift_time.get(s, SHIFT_TIME.get(s, "")),
            "role": r,
        })
    return pd.DataFrame(rows)


def make_coverage_summary(state: SAState, data: InputData) -> pd.DataFrame:
    cov, *_ = build_counts(state)
    rows = []
    for cell in data.cells:
        d, s, r = cell
        actual = cov[cell]
        cmin = data.c_value(cell, "C_min")
        copt = data.c_value(cell, "C_opt")
        cmax = data.c_value(cell, "C_max")
        rows.append({
            "day": d,
            "shift": s,
            "shift_time": data.shift_time.get(s, SHIFT_TIME.get(s, "")),
            "role": r,
            "C_min": cmin,
            "C_opt": copt,
            "C_max": cmax,
            "actual_staff": actual,
            "gap_to_Copt": actual - copt,
            "shortage_to_Cmin": max(0, cmin - actual),
            "shortage_to_Copt": max(0, copt - actual),
            "overstaff_above_Copt": max(0, actual - copt),
            "overstaff_above_Cmax": max(0, actual - cmax),
            "coverage_status": (
                "UNDER_CMIN" if actual < cmin else
                "UNDER_COPT" if actual < copt else
                "OVER_CMAX" if actual > cmax else
                "OK"
            )
        })
    return pd.DataFrame(rows).sort_values(["day", "shift", "role"])


def make_employee_summary(state: SAState, data: InputData) -> pd.DataFrame:
    assignment = assignment_df_from_state(state, data)
    rows = []
    for e in data.employees:
        e_assign = assignment[assignment["employee_id"] == e]
        work_days = int(e_assign["day"].nunique()) if not e_assign.empty else 0
        hours = 0.0
        night = 0
        shift_counts = {f"count_{s}": 0 for s in data.shifts}
        role_counts = {f"count_{r}": 0 for r in data.roles}
        for _, row in e_assign.iterrows():
            s = row["shift"]
            r = row["role"]
            hours += data.shift_hours.get(s, 8.0)
            night += data.is_night_shift.get(s, 0)
            shift_counts[f"count_{s}"] = shift_counts.get(f"count_{s}", 0) + 1
            role_counts[f"count_{r}"] = role_counts.get(f"count_{r}", 0) + 1
        c = data.emp_row.get(e, {})
        rows.append({
            "employee_id": e,
            "employee_name": c.get("employee_name", e),
            "work_days": work_days,
            "hours": hours,
            "night_shifts": night,
            "min_work_days": c.get("min_work_days", pd.NA),
            "max_work_days": c.get("max_work_days", pd.NA),
            "target_work_days": c.get("target_work_days", pd.NA),
            "standard_role": data.standard_role.get(e, pd.NA),
            "standard_shift": data.standard_shift.get(e, pd.NA),
            **shift_counts,
            **role_counts,
        })
    return pd.DataFrame(rows)


def make_roster_matrix(state: SAState, data: InputData) -> pd.DataFrame:
    assignment = assignment_df_from_state(state, data)
    lookup = {}
    for _, row in assignment.iterrows():
        lookup[(row["employee_id"], int(row["day"]))] = f"{row['shift']}-{row['role']}"
    rows = []
    for e in data.employees:
        row = {"employee_id": e, "employee_name": data.emp_row.get(e, {}).get("employee_name", e)}
        for d in data.days:
            if (e, d) in lookup:
                row[f"D{d}"] = lookup[(e, d)]
            elif data.is_blocked(e, d):
                row[f"D{d}"] = "UNAVAILABLE"
            else:
                row[f"D{d}"] = "OFF"
        rows.append(row)
    return pd.DataFrame(rows)


def score_to_rows(prefix: str, score: ScoreResult) -> List[Dict[str, object]]:
    return [
        {"metric": f"{prefix}_score", "value": score.score},
        {"metric": f"{prefix}_hard_violations", "value": score.hard_violations},
        {"metric": f"{prefix}_shortage_Cmin", "value": score.total_shortage_cmin},
        {"metric": f"{prefix}_shortage_Copt", "value": score.total_shortage_copt},
        {"metric": f"{prefix}_overstaff_Copt", "value": score.total_over_copt},
        {"metric": f"{prefix}_overstaff_Cmax", "value": score.total_over_cmax},
        {"metric": f"{prefix}_preference_penalty", "value": score.preference_penalty},
        {"metric": f"{prefix}_fairness_penalty", "value": score.fairness_penalty},
        {"metric": f"{prefix}_stability_penalty", "value": score.stability_penalty},
        {"metric": f"{prefix}_rest_violations", "value": score.rest_violations},
        {"metric": f"{prefix}_skill_violations", "value": score.skill_violations},
        {"metric": f"{prefix}_availability_violations", "value": score.availability_violations},
        {"metric": f"{prefix}_one_per_day_violations", "value": score.one_per_day_violations},
        {"metric": f"{prefix}_contract_violations", "value": score.contract_violations},
        {"metric": f"{prefix}_fixed_assignment_violations", "value": score.fixed_assignment_violations},
        {"metric": f"{prefix}_incompatibility_penalty", "value": score.incompatibility_penalty},
    ]


def export_outputs(output_dir: Path, data: InputData, best_state: SAState, initial_score: ScoreResult, final_score: ScoreResult, history: pd.DataFrame, params: SAParams, scale_level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment = assignment_df_from_state(best_state, data)
    coverage = make_coverage_summary(best_state, data)
    employees = make_employee_summary(best_state, data)
    roster = make_roster_matrix(best_state, data)

    improvement = initial_score.score - final_score.score
    improvement_pct = 100.0 * improvement / initial_score.score if initial_score.score else 0.0
    status_rows = [
        {"metric": "scale_level", "value": scale_level},
        {"metric": "initial_source_used", "value": "MILP assignment_long.csv"},
        {"metric": "temperature_initial", "value": params.initial_temp},
        {"metric": "temperature_min", "value": params.min_temp},
        {"metric": "cooling_rate", "value": params.cooling_rate},
        {"metric": "cycles_per_temp", "value": params.cycles_per_temp},
        {"metric": "max_iterations", "value": params.max_iterations},
        {"metric": "max_seconds", "value": params.max_seconds},
        {"metric": "score_improvement", "value": improvement},
        {"metric": "score_improvement_percent", "value": improvement_pct},
        {"metric": "final_assignments", "value": len(assignment)},
        {"metric": "total_C_min", "value": int(data.coverage_df["C_min"].sum())},
        {"metric": "total_C_opt", "value": int(data.coverage_df["C_opt"].sum())},
        {"metric": "total_C_max", "value": int(data.coverage_df["C_max"].sum())},
    ]
    status_rows.extend(score_to_rows("initial", initial_score))
    status_rows.extend(score_to_rows("final", final_score))
    status = pd.DataFrame(status_rows)

    outputs = {
        "sa_assignment_long": assignment,
        "sa_roster_matrix": roster,
        "sa_coverage_summary": coverage,
        "sa_employee_summary": employees,
        "sa_model_status": status,
        "sa_iteration_history": history,
        "sa_demand_input": data.coverage_df,
    }
    for name, df in outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(output_dir / "sa_phase2_output.xlsx", engine="openpyxl") as writer:
        for name, df in outputs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SA Phase 2 post-optimization for fixed MEDIUM scale MILP roster.")
    p.add_argument("--scale-level", choices=["medium"], default="medium", help=argparse.SUPPRESS)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Scaled employee_milp_inputs folder. Default: %(default)s")
    p.add_argument("--milp-output-dir", type=Path, default=DEFAULT_MILP_OUTPUT_DIR, help="MILP output folder containing assignment_long.csv. Default: %(default)s")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="SA output folder. Default: %(default)s")
    p.add_argument("--initial-temp", type=float, default=None)
    p.add_argument("--min-temp", type=float, default=None)
    p.add_argument("--cooling-rate", type=float, default=None)
    p.add_argument("--cycles-per-temp", type=int, default=None)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None, help="0 = no time limit. Recommended: medium 600, large 900-1800.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--hard-penalty", type=float, default=None)
    return p.parse_args()


def params_from_args(args: argparse.Namespace) -> SAParams:
    base = PRESET_PARAMS[args.scale_level]
    return SAParams(
        initial_temp=args.initial_temp if args.initial_temp is not None else base.initial_temp,
        min_temp=args.min_temp if args.min_temp is not None else base.min_temp,
        cooling_rate=args.cooling_rate if args.cooling_rate is not None else base.cooling_rate,
        cycles_per_temp=args.cycles_per_temp if args.cycles_per_temp is not None else base.cycles_per_temp,
        max_iterations=args.max_iterations if args.max_iterations is not None else base.max_iterations,
        max_seconds=args.max_seconds if args.max_seconds is not None else base.max_seconds,
        seed=args.seed if args.seed is not None else base.seed,
        hard_penalty=args.hard_penalty if args.hard_penalty is not None else base.hard_penalty,
    )


def main() -> None:
    args = parse_args()
    params = params_from_args(args)
    print("Loading inputs...")
    data = InputData(args.input_dir, args.milp_output_dir)
    print(f"Scale level        : {args.scale_level}")
    print(f"Employees          : {len(data.employees)}")
    print(f"Days               : {len(data.days)}")
    print(f"Coverage cells     : {len(data.cells)}")
    print(f"MILP assignments   : {len(data.assignment_initial_df)}")
    print(f"SA params          : T0={params.initial_temp}, Tmin={params.min_temp}, beta={params.cooling_rate}, cycles={params.cycles_per_temp}")

    best_state, initial_score, final_score, history = run_sa(data, params)
    export_outputs(args.output_dir, data, best_state, initial_score, final_score, history, params, args.scale_level)

    print("DONE: SA Phase 2 completed")
    print(f"Output dir          : {args.output_dir}")
    print(f"Initial score       : {initial_score.score:,.2f}")
    print(f"Final score         : {final_score.score:,.2f}")
    if initial_score.score:
        print(f"Improvement         : {(initial_score.score - final_score.score):,.2f} ({100*(initial_score.score-final_score.score)/initial_score.score:.2f}%)")
    print(f"Final hard violations: {final_score.hard_violations}")
    print(f"Final shortage Copt : {final_score.total_shortage_copt}")
    print(f"Final overstaff Copt: {final_score.total_over_copt}")


if __name__ == "__main__":
    main()
