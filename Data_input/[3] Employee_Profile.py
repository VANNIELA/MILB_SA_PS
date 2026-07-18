#!/usr/bin/env python3
r"""
employee_profile_generation_daily_workload.py

Generate consistent employee input files for the Airport Lounge MILP rostering model, with an option to build day-specific demand from role_daily_shift_workload.

Main correction compared with the previous employee profile generation code
--------------------------------------------------------------------------
The previous logic could produce a demand file with only minimum coverage slots
(e.g., 840 assignments) while still using a larger employee pool (e.g., 48 employees)
and a business rule requiring 20-24 working days per employee. This is infeasible
unless the model creates planned overstaffing. The corrected logic separates:

1. C_min  : minimum role coverage needed to prevent service breakdown.
2. C_opt  : target / buffered coverage used by the current MILP as scheduled demand.
3. C_max  : soft upper tolerance for overstaffing control.

For compatibility with the current MILP code, this script still exports
`staffing_demand_daily.csv` in the old wide format. By default, that file contains
C_opt values, not C_min values. Therefore, the MILP receives the operational target
demand rather than the minimum-only demand.

Generated outputs
-----------------
Core MILP-compatible files:
1. employee_master.csv
2. employee_skills.csv
3. employee_availability.csv
4. employee_preferences.csv
5. employee_history.csv
6. employee_incompatibility.csv
7. standard_role_shift_preferences.csv
8. staffing_demand_daily.csv              # old format, uses C_opt by default
9. skill_pool_targets.csv
10. generation_validation_report.csv

Extended formulation support files:
11. staffing_coverage_bands.csv           # C_min / C_opt / C_max by day-shift-role
12. shift_structure.csv
13. shift_transition_rest.csv
14. penalty_config.csv
15. employee_standard_profile.csv
16. employee_assignment_preferences.csv
17. employee_day_requests.csv
18. employee_fixed_assignments.csv
19. employee_pairing.csv
20. holiday_calendar.csv
21. weekend_policy.csv

Example
-------
python employee_profile_generation_fixed.py ^
  --input "E:\\NĂM 4\\Capstone\\Data_input\\Final_data_input_milp.xlsx" ^
  --output-dir "E:\\NĂM 4\\Capstone\\Data_input\\employee_milp_inputs" ^
  --days 28 ^
  --min-work-days 20 ^
  --max-work-days 24 ^
  --minimum-employee-count 48 ^
  --target-coverage-buffer-rate 0.15 ^
  --milp-demand-level opt ^
  --demand-source daily_workload ^
  --daily-required-column required_staff_by_workload

If you want to force the robust 1,064-slot version:
python employee_profile_generation_fixed.py --target-total-slots 1064 --minimum-employee-count 48
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROLES = ["RS", "DAS", "BLO", "FSTC", "DS", "SLS"]
SHIFTS = ["S1", "S2", "S3"]
SHIFT_TIME = {
    "S1": "00:00-08:00",
    "S2": "08:00-16:00",
    "S3": "16:00-24:00",
}
SHIFT_START_MIN = {"S1": 0, "S2": 8 * 60, "S3": 16 * 60}
SHIFT_END_MIN = {"S1": 8 * 60, "S2": 16 * 60, "S3": 24 * 60}
NIGHT_SHIFT = "S1"

# This fallback is the workload-driven version that produces 38 staff-shifts/day,
# 1,064 assignments over 28 days. It is used only if no valid workbook/pivot exists.
FALLBACK_STAFFING_PIVOT = pd.DataFrame(
    [
        {"shift": "S1", "shift_time": "00:00-08:00", "RS": 1, "DAS": 4, "BLO": 1, "FSTC": 1, "DS": 1, "SLS": 1},
        {"shift": "S2", "shift_time": "08:00-16:00", "RS": 1, "DAS": 8, "BLO": 2, "FSTC": 1, "DS": 1, "SLS": 1},
        {"shift": "S3", "shift_time": "16:00-24:00", "RS": 1, "DAS": 9, "BLO": 2, "FSTC": 1, "DS": 1, "SLS": 1},
    ]
)

DEFAULT_ROLE_BUFFER_WEIGHTS = {
    "RS": 1.00,
    "DAS": 4.00,
    "BLO": 2.00,
    "FSTC": 0.75,
    "DS": 1.50,
    "SLS": 1.00,
}

DEFAULT_SHIFT_BUFFER_WEIGHTS = {
    "S1": 0.75,
    "S2": 1.25,
    "S3": 1.25,
}

BAD_AVAILABILITY_STATUS = {
    "unavailable", "leave", "paid_leave", "unpaid_leave", "holiday",
    "vacation", "recovery", "not_available", "blocked",
}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def normalize_shift(value) -> str:
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
    if s in set(SHIFTS):
        return s
    return s


def safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(math.ceil(float(value)))
    except Exception:
        return default


def read_optional_csv(path: Path, required_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(required_cols or []))
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_columns(df)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Optional CSV {path} is missing required columns: {missing}")
    return df


def parse_weight_string(text: str, default: Dict[str, float], valid_keys: Sequence[str]) -> Dict[str, float]:
    weights = dict(default)
    if not text:
        return weights
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Weight token must use key=value format: {token}")
        key, val = token.split("=", 1)
        key = key.strip().upper()
        if key not in valid_keys:
            raise ValueError(f"Invalid weight key {key}. Valid keys: {valid_keys}")
        weights[key] = float(val)
    return weights


def minute_to_clock(minute: int) -> str:
    minute = int(minute) % (24 * 60)
    return f"{minute // 60:02d}:{minute % 60:02d}"


# -----------------------------------------------------------------------------
# Workload loading and demand construction
# -----------------------------------------------------------------------------


def find_workbook(input_path: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if input_path:
        p = Path(input_path).expanduser()
        candidates.append(p)
        if p.is_dir():
            candidates.extend([
                p / "Final_data_input_milp.xlsx",
                p / "airport_lounge_28day_event_sim_output.xlsx",
            ])
        else:
            candidates.extend([
                p.parent / "Final_data_input_milp.xlsx",
                p.parent / "airport_lounge_28day_event_sim_output.xlsx",
            ])

    cwd = Path.cwd()
    candidates.extend([
        cwd / "Final_data_input_milp.xlsx",
        cwd / "airport_lounge_28day_event_sim_output.xlsx",
    ])

    seen = set()
    for c in candidates:
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.is_file():
            return c
    return None


def load_staffing_pivot(workbook_path: Optional[Path]) -> Tuple[pd.DataFrame, str]:
    """Load role/shift minimum staffing requirement from a workbook or fallback."""
    if workbook_path is None:
        df = FALLBACK_STAFFING_PIVOT.copy()
        df[ROLES] = df[ROLES].astype(int)
        df["total_staff"] = df[ROLES].sum(axis=1)
        return df, "fallback_default_no_workbook"

    try:
        xl = pd.ExcelFile(workbook_path)
    except Exception as exc:
        df = FALLBACK_STAFFING_PIVOT.copy()
        df["total_staff"] = df[ROLES].sum(axis=1)
        return df, f"fallback_default_cannot_open_workbook: {exc}"

    if "staffing_pivot" not in xl.sheet_names:
        df = FALLBACK_STAFFING_PIVOT.copy()
        df["total_staff"] = df[ROLES].sum(axis=1)
        return df, "fallback_default_missing_staffing_pivot_sheet"

    raw = pd.read_excel(workbook_path, sheet_name="staffing_pivot")
    raw = clean_columns(raw)
    missing = [r for r in ROLES if r not in raw.columns]
    if missing:
        raise ValueError(f"staffing_pivot exists but is missing role columns: {missing}")

    if "shift" in raw.columns:
        raw["shift"] = raw["shift"].apply(normalize_shift)
    elif "shift_id" in raw.columns:
        raw["shift"] = raw["shift_id"].apply(normalize_shift)
    else:
        raw = raw.reset_index(drop=True)
        raw["shift"] = [SHIFTS[i] if i < len(SHIFTS) else f"S{i + 1}" for i in range(len(raw))]

    raw = raw[raw["shift"].isin(SHIFTS)].copy()
    if raw.empty:
        raise ValueError("staffing_pivot has no valid S1/S2/S3 rows.")

    for role in ROLES:
        raw[role] = pd.to_numeric(raw[role], errors="coerce").fillna(0).apply(lambda x: int(math.ceil(x)))

    pivot = raw.groupby("shift", as_index=False)[ROLES].max()
    pivot["shift_time"] = pivot["shift"].map(SHIFT_TIME)
    pivot["shift_order"] = pivot["shift"].map({"S1": 1, "S2": 2, "S3": 3})
    pivot = pivot.sort_values("shift_order").drop(columns=["shift_order"])
    pivot["total_staff"] = pivot[ROLES].sum(axis=1)
    return pivot[["shift", "shift_time", *ROLES, "total_staff"]], f"loaded_from_{workbook_path.name}:staffing_pivot"


def expand_pivot_by_day(staffing_pivot: pd.DataFrame, days: int) -> pd.DataFrame:
    rows = []
    for day in range(1, days + 1):
        for _, row in staffing_pivot.iterrows():
            rec = {"day": day, "shift": row["shift"], "shift_time": row.get("shift_time", SHIFT_TIME.get(row["shift"], ""))}
            for role in ROLES:
                rec[role] = max(0, safe_int(row[role]))
            rec["total_staff"] = sum(rec[role] for role in ROLES)
            rows.append(rec)
    return pd.DataFrame(rows)


def choose_daily_required_column(raw: pd.DataFrame, choice: str) -> str:
    """
    Select the staffing requirement column from role_daily_shift_workload.

    For daily-varying demand, required_staff_by_workload is preferred because it is
    directly computed from each day/shift/role workload minutes. If that column is
    missing, the function falls back to recommended_staff_initial. The user can
    force a specific column with --daily-required-column.
    """
    columns = set(raw.columns)
    choice = str(choice or "auto").strip()
    if choice != "auto":
        if choice not in columns:
            raise ValueError(
                f"Requested daily required column '{choice}' was not found in role_daily_shift_workload. "
                f"Available columns: {list(raw.columns)}"
            )
        return choice

    # Preferred order for day-specific variability.
    candidates = [
        "required_staff_by_workload",
        "recommended_staff_initial",
        "staff_required_p95_workload",
        "staff_required_mean_workload",
        "required_staff",
    ]
    for col in candidates:
        if col in columns:
            return col
    raise ValueError(
        "role_daily_shift_workload must contain one of: " + ", ".join(candidates)
    )


def load_role_daily_shift_workload(
    workbook_path: Optional[Path],
    days: int,
    min_staff: int,
    daily_required_column: str,
) -> Tuple[pd.DataFrame, str]:
    """
    Load day-specific role/shift demand from role_daily_shift_workload.

    Expected long-format columns:
    - day_index or day
    - shift_id or shift
    - role
    - required_staff_by_workload or recommended_staff_initial

    Output format is the same wide format used by the current MILP:
    day, shift, shift_time, RS, DAS, BLO, FSTC, DS, SLS, total_staff
    """
    if workbook_path is None:
        raise FileNotFoundError("No workload workbook was found, so role_daily_shift_workload cannot be loaded.")

    xl = pd.ExcelFile(workbook_path)
    if "role_daily_shift_workload" not in xl.sheet_names:
        raise ValueError(
            f"Workbook {workbook_path.name} does not contain sheet 'role_daily_shift_workload'. "
            f"Available sheets: {xl.sheet_names}"
        )

    raw = pd.read_excel(workbook_path, sheet_name="role_daily_shift_workload")
    raw = clean_columns(raw)
    if raw.empty:
        raise ValueError("role_daily_shift_workload sheet is empty.")

    day_col = "day_index" if "day_index" in raw.columns else "day" if "day" in raw.columns else None
    shift_col = "shift_id" if "shift_id" in raw.columns else "shift" if "shift" in raw.columns else None
    required_col = choose_daily_required_column(raw, daily_required_column)

    missing = []
    if day_col is None:
        missing.append("day_index or day")
    if shift_col is None:
        missing.append("shift_id or shift")
    if "role" not in raw.columns:
        missing.append("role")
    if missing:
        raise ValueError(f"role_daily_shift_workload is missing required columns: {missing}")

    df = raw.copy()
    df["day"] = pd.to_numeric(df[day_col], errors="coerce")
    df = df[df["day"].notna()].copy()
    df["day"] = df["day"].astype(int)
    df = df[df["day"].between(1, days)].copy()

    df["shift"] = df[shift_col].apply(normalize_shift)
    df = df[df["shift"].isin(SHIFTS)].copy()
    df["role"] = df["role"].astype(str).str.strip().str.upper()
    df = df[df["role"].isin(ROLES)].copy()
    df[required_col] = pd.to_numeric(df[required_col], errors="coerce").fillna(0)

    if df.empty:
        raise ValueError("No valid day/shift/role rows remain after filtering role_daily_shift_workload.")

    pivot = df.pivot_table(
        index=["day", "shift"],
        columns="role",
        values=required_col,
        aggfunc="max",
        fill_value=0,
    ).reset_index()

    # Ensure a complete day x shift grid. Missing cells receive mandatory min staff.
    complete = pd.MultiIndex.from_product(
        [range(1, days + 1), SHIFTS], names=["day", "shift"]
    ).to_frame(index=False)
    out = complete.merge(pivot, on=["day", "shift"], how="left")
    out["shift_time"] = out["shift"].map(SHIFT_TIME)

    for role in ROLES:
        if role not in out.columns:
            out[role] = 0
        out[role] = pd.to_numeric(out[role], errors="coerce").fillna(0).apply(
            lambda x: max(int(min_staff), int(math.ceil(float(x))))
        )

    out["total_staff"] = out[ROLES].sum(axis=1)
    out["shift_order"] = out["shift"].map({"S1": 1, "S2": 2, "S3": 3})
    out = out.sort_values(["day", "shift_order"]).drop(columns=["shift_order"]).reset_index(drop=True)
    return out[["day", "shift", "shift_time", *ROLES, "total_staff"]], (
        f"loaded_from_{workbook_path.name}:role_daily_shift_workload:{required_col}"
    )


def build_minimum_demand_from_source(args: argparse.Namespace, workbook: Optional[Path]) -> Tuple[pd.DataFrame, str]:
    """
    Build C_min demand from either:
    - role_daily_shift_workload: day-specific requirement, can vary by day.
    - staffing_pivot: stable shift-role requirement repeated for every day.
    - auto: try role_daily_shift_workload first, then fallback to staffing_pivot.
    """
    source = str(args.demand_source).strip().lower()

    if source in {"daily_workload", "auto"}:
        try:
            return load_role_daily_shift_workload(
                workbook_path=workbook,
                days=args.days,
                min_staff=args.min_staff,
                daily_required_column=args.daily_required_column,
            )
        except Exception as exc:
            if source == "daily_workload" or not args.allow_demand_source_fallback:
                raise
            print(f"WARNING: could not load role_daily_shift_workload ({exc}). Falling back to staffing_pivot.")

    staffing_pivot_min, workload_source = load_staffing_pivot(workbook)
    return expand_pivot_by_day(staffing_pivot_min, args.days), workload_source


# -----------------------------------------------------------------------------
# Headcount and coverage band correction
# -----------------------------------------------------------------------------


def initial_employee_count(
    total_min_slots: int,
    min_work_days: int,
    max_work_days: int,
    headcount_buffer: float,
    manual_employee_count: Optional[int],
    minimum_employee_count: int,
) -> Tuple[int, Dict[str, int | float | str]]:
    n_min = int(math.ceil(total_min_slots / max_work_days))
    n_max = int(math.floor(total_min_slots / min_work_days))
    if manual_employee_count is not None:
        n = int(manual_employee_count)
        rule = "manual_employee_count"
    else:
        buffered = int(math.ceil(n_min * (1.0 + headcount_buffer)))
        n = max(buffered, int(minimum_employee_count))
        rule = "max(ceil(n_min_by_min_coverage*(1+buffer)), minimum_employee_count)"

    return n, {
        "total_min_coverage_slots": total_min_slots,
        "n_min_by_min_coverage_max_days": n_min,
        "n_max_by_min_coverage_min_days": n_max,
        "employees_initial": n,
        "headcount_rule": rule,
    }


def compute_target_total_slots(
    total_min_slots: int,
    employee_count: int,
    min_work_days: int,
    max_work_days: int,
    target_coverage_buffer_rate: float,
    target_total_slots: Optional[int],
) -> int:
    if target_total_slots is not None:
        target = int(target_total_slots)
    else:
        target = max(
            total_min_slots,
            int(math.ceil(total_min_slots * (1.0 + target_coverage_buffer_rate))),
            int(employee_count * min_work_days),
        )

    if target < total_min_slots:
        raise ValueError(f"target_total_slots={target} cannot be lower than minimum coverage slots={total_min_slots}.")
    if target > employee_count * max_work_days:
        raise ValueError(
            f"Target slots {target} exceed capacity of {employee_count} employees at max_work_days={max_work_days}: "
            f"{employee_count * max_work_days}. Increase employees or reduce target coverage."
        )
    return target


def adjust_employee_count_until_feasible(
    total_min_slots: int,
    employee_count: int,
    min_work_days: int,
    max_work_days: int,
    target_coverage_buffer_rate: float,
    target_total_slots: Optional[int],
    manual_employee_count: Optional[int],
) -> Tuple[int, int]:
    """Ensure target slots fit within [N*min_days, N*max_days]."""
    n = int(employee_count)
    for _ in range(20):
        target = compute_target_total_slots(
            total_min_slots=total_min_slots,
            employee_count=n,
            min_work_days=min_work_days,
            max_work_days=max_work_days,
            target_coverage_buffer_rate=target_coverage_buffer_rate,
            target_total_slots=target_total_slots,
        )
        if n * min_work_days <= target <= n * max_work_days:
            return n, target
        if target > n * max_work_days:
            if manual_employee_count is not None:
                raise ValueError("Manual employee count is too small for target slots and max_work_days.")
            n = int(math.ceil(target / max_work_days))
            continue
        if target < n * min_work_days:
            # This should not occur because compute_target_total_slots includes n*min_work_days.
            target = n * min_work_days
            return n, target
    raise ValueError("Could not reconcile employee count and target slots after repeated adjustments.")


def allocate_extra_slots(
    demand_min: pd.DataFrame,
    target_total_slots: int,
    role_weights: Dict[str, float],
    shift_weights: Dict[str, float],
) -> pd.DataFrame:
    """Create C_opt by adding buffered slots to high-risk role/shift cells."""
    out = demand_min.copy()
    for role in ROLES:
        out[f"{role}_min"] = out[role].astype(int)
        out[f"{role}_opt"] = out[role].astype(int)

    current_total = int(out[[f"{r}_opt" for r in ROLES]].sum().sum())
    extra_needed = int(target_total_slots - current_total)
    if extra_needed <= 0:
        return out

    cell_rows = []
    for idx, row in out.iterrows():
        shift = row["shift"]
        for role in ROLES:
            base = max(1, int(row[f"{role}_min"]))
            weight = float(role_weights.get(role, 1.0)) * float(shift_weights.get(shift, 1.0)) * base
            cell_rows.append({"idx": idx, "role": role, "weight": weight})

    weight_sum = sum(c["weight"] for c in cell_rows)
    if weight_sum <= 0:
        raise ValueError("Cannot allocate buffer slots because all allocation weights are zero.")

    raw_alloc = [extra_needed * c["weight"] / weight_sum for c in cell_rows]
    floors = [int(math.floor(x)) for x in raw_alloc]
    remainder = extra_needed - sum(floors)
    fractions = sorted(
        [(i, raw_alloc[i] - floors[i]) for i in range(len(raw_alloc))],
        key=lambda x: x[1],
        reverse=True,
    )
    for i, f in enumerate(floors):
        if f:
            c = cell_rows[i]
            out.at[c["idx"], f"{c['role']}_opt"] += f
    for i, _ in fractions[:remainder]:
        c = cell_rows[i]
        out.at[c["idx"], f"{c['role']}_opt"] += 1

    return out


def build_coverage_bands(
    demand_min: pd.DataFrame,
    employee_count: int,
    min_work_days: int,
    max_work_days: int,
    target_coverage_buffer_rate: float,
    target_total_slots: Optional[int],
    max_coverage_buffer_rate: float,
    role_weights: Dict[str, float],
    shift_weights: Dict[str, float],
) -> Tuple[pd.DataFrame, int]:
    total_min = int(demand_min[ROLES].sum().sum())
    target_total = compute_target_total_slots(
        total_min_slots=total_min,
        employee_count=employee_count,
        min_work_days=min_work_days,
        max_work_days=max_work_days,
        target_coverage_buffer_rate=target_coverage_buffer_rate,
        target_total_slots=target_total_slots,
    )

    bands = allocate_extra_slots(demand_min, target_total, role_weights, shift_weights)
    for role in ROLES:
        bands[f"{role}_max"] = bands[f"{role}_opt"].apply(
            lambda x: max(int(x), int(math.ceil(float(x) * (1.0 + max_coverage_buffer_rate))))
        )

    ordered_cols = ["day", "shift", "shift_time"]
    for role in ROLES:
        ordered_cols.extend([f"{role}_min", f"{role}_opt", f"{role}_max"])
    bands["total_min"] = bands[[f"{r}_min" for r in ROLES]].sum(axis=1)
    bands["total_opt"] = bands[[f"{r}_opt" for r in ROLES]].sum(axis=1)
    bands["total_max"] = bands[[f"{r}_max" for r in ROLES]].sum(axis=1)
    ordered_cols.extend(["total_min", "total_opt", "total_max"])
    return bands[ordered_cols], target_total


def demand_for_milp_from_bands(coverage_bands: pd.DataFrame, demand_level: str) -> pd.DataFrame:
    level = demand_level.lower()
    if level not in {"min", "opt", "max"}:
        raise ValueError("demand_level must be one of: min, opt, max")
    out = coverage_bands[["day", "shift", "shift_time"]].copy()
    for role in ROLES:
        out[role] = coverage_bands[f"{role}_{level}"].astype(int)
    out["total_staff"] = out[ROLES].sum(axis=1)
    return out


def coverage_bands_long(coverage_bands: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in coverage_bands.iterrows():
        for role in ROLES:
            rows.append({
                "day": int(row["day"]),
                "shift": row["shift"],
                "shift_time": row["shift_time"],
                "role": role,
                "C_min": int(row[f"{role}_min"]),
                "C_opt": int(row[f"{role}_opt"]),
                "C_max": int(row[f"{role}_max"]),
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Skill pool and employee files
# -----------------------------------------------------------------------------


def compute_skill_pool_targets(demand_daily: pd.DataFrame, max_work_days: int, skill_buffer: float) -> pd.DataFrame:
    rows = []
    for role in ROLES:
        role_slots = int(demand_daily[role].sum())
        peak = int(demand_daily[role].max())
        pool_min = max(peak, int(math.ceil(role_slots / max_work_days)))
        pool_target = int(math.ceil(pool_min * (1.0 + skill_buffer)))
        rows.append({
            "role": role,
            "role_slots_28d": role_slots,
            "peak_concurrent_need": peak,
            "pool_min": pool_min,
            "pool_target": pool_target,
        })
    return pd.DataFrame(rows)


def generate_employee_master(
    n: int,
    min_work_days: int,
    max_work_days: int,
    max_night_shifts: int,
    max_consecutive_work_days: int,
    max_consecutive_night_shifts: int,
    min_consecutive_days_off: int,
    min_days_off_after_work_block: int,
    night_block_separation: int,
    max_working_weekends: int,
    max_working_public_holidays: int,
    complete_weekend_required: int,
) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "employee_id": f"E{i:03d}",
            "employee_name": f"Employee {i}",
            "min_work_days": min_work_days,
            "max_work_days": max_work_days,
            "target_work_days": round((min_work_days + max_work_days) / 2, 2),
            "min_hours": min_work_days * 8,
            "max_hours": max_work_days * 8,
            "target_hours": round(((min_work_days + max_work_days) / 2) * 8, 2),
            "min_days_off": max(0, 28 - max_work_days),
            "max_days_off": max(0, 28 - min_work_days),
            "max_night_shifts": max_night_shifts,
            "max_consecutive_work_days": max_consecutive_work_days,
            "max_consecutive_night_shifts": max_consecutive_night_shifts,
            "min_consecutive_days_off": min_consecutive_days_off,
            "min_days_off_after_work_block": min_days_off_after_work_block,
            "night_block_separation": night_block_separation,
            "max_working_weekends": max_working_weekends,
            "max_working_public_holidays": max_working_public_holidays,
            "complete_weekend_required": int(complete_weekend_required),
        })
    return pd.DataFrame(rows)


def generate_employee_skills(
    employee_ids: Sequence[str],
    skill_targets: pd.DataFrame,
    demand_daily: pd.DataFrame,
    rng: np.random.Generator,
    das_universal: bool,
    skill_multiplier: float,
    max_initial_probability: float = 0.80,
) -> pd.DataFrame:
    n = len(employee_ids)
    skills = pd.DataFrame({"employee_id": employee_ids})
    for role in ROLES:
        skills[role] = 0

    target_map = dict(zip(skill_targets["role"], skill_targets["pool_target"]))
    role_slots = np.array([max(1, int(demand_daily[role].sum())) for role in ROLES], dtype=float)
    role_weights = role_slots / role_slots.sum()

    for role in ROLES:
        target = int(target_map[role])
        if role == "DAS" and das_universal:
            skills[role] = 1
            continue
        p = min(max_initial_probability, max(0.10, (target / max(1, n)) * skill_multiplier))
        skills[role] = rng.binomial(1, p, size=n)
        current = int(skills[role].sum())
        if current < target:
            zero_idx = np.where(skills[role].to_numpy() == 0)[0]
            chosen = rng.choice(zero_idx, size=min(target - current, len(zero_idx)), replace=False)
            skills.loc[chosen, role] = 1

    # Prevent zero-skill employees.
    zero_skill_idx = np.where(skills[ROLES].sum(axis=1).to_numpy() == 0)[0]
    for idx in zero_skill_idx:
        selected_role = str(rng.choice(ROLES, p=role_weights))
        skills.loc[idx, selected_role] = 1

    # Final target check.
    for role in ROLES:
        target = int(target_map[role])
        current = int(skills[role].sum())
        if current < target:
            zero_idx = np.where(skills[role].to_numpy() == 0)[0]
            chosen = rng.choice(zero_idx, size=min(target - current, len(zero_idx)), replace=False)
            skills.loc[chosen, role] = 1

    return skills


def generate_standard_profiles(
    skills_df: pd.DataFrame,
    demand_daily: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    role_slots = np.array([max(1, int(demand_daily[role].sum())) for role in ROLES], dtype=float)
    global_role_probs = role_slots / role_slots.sum()
    shift_probs = np.array([0.25, 0.45, 0.30])
    rows = []
    for _, row in skills_df.iterrows():
        employee_id = row["employee_id"]
        qualified = [role for role in ROLES if int(row[role]) == 1]
        if not qualified:
            qualified = [str(rng.choice(ROLES, p=global_role_probs))]
        probs = np.array([global_role_probs[ROLES.index(role)] for role in qualified], dtype=float)
        probs = probs / probs.sum()
        standard_role = str(rng.choice(qualified, p=probs))
        standard_shift = str(rng.choice(SHIFTS, p=shift_probs))
        rows.append({"employee_id": employee_id, "standard_role": standard_role, "standard_shift": standard_shift})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Availability and requests
# -----------------------------------------------------------------------------


def availability_is_feasible(
    availability_rows: List[Dict],
    skills_df: pd.DataFrame,
    demand_daily: pd.DataFrame,
    safety_margin: int,
) -> Tuple[bool, List[str]]:
    skill_by_role = {role: set(skills_df.loc[skills_df[role] == 1, "employee_id"].astype(str)) for role in ROLES}
    blocked_by_day: Dict[int, set] = {}
    for row in availability_rows:
        status = str(row.get("status", "")).strip().lower().replace(" ", "_")
        if status not in BAD_AVAILABILITY_STATUS:
            continue
        day = int(row["day"])
        employee_id = str(row["employee_id"])
        blocked_by_day.setdefault(day, set()).add(employee_id)

    errors = []
    peak_req = demand_daily.groupby("day")[ROLES].max().reset_index()
    for _, req_row in peak_req.iterrows():
        day = int(req_row["day"])
        blocked = blocked_by_day.get(day, set())
        for role in ROLES:
            available_qualified = len(skill_by_role[role] - blocked)
            required = int(req_row[role]) + int(safety_margin)
            if available_qualified < required:
                errors.append(f"Day {day}, role {role}: available qualified {available_qualified} < required {required}")
    return len(errors) == 0, errors


def add_availability_row_if_feasible(
    rows: List[Dict],
    candidate: Dict,
    skills_df: pd.DataFrame,
    demand_daily: pd.DataFrame,
    safety_margin: int,
) -> bool:
    key = (str(candidate["employee_id"]), int(candidate["day"]))
    for row in rows:
        if (str(row["employee_id"]), int(row["day"])) == key:
            return False
    feasible, _ = availability_is_feasible(rows + [candidate], skills_df, demand_daily, safety_margin)
    if feasible:
        rows.append(candidate)
        return True
    return False


def load_real_availability_requests(input_dir: Path, valid_employee_ids: set, days: int) -> List[Dict]:
    specs = [
        ("leave_requests.csv", "leave"),
        ("unavailable_requests.csv", "unavailable"),
        ("fixed_days_off.csv", "unavailable"),
    ]
    rows: List[Dict] = []
    for filename, default_status in specs:
        df = read_optional_csv(input_dir / filename, required_cols=["employee_id", "day"])
        if df.empty:
            continue
        for _, row in df.iterrows():
            employee_id = str(row["employee_id"]).strip()
            day = int(row["day"])
            if employee_id not in valid_employee_ids or not (1 <= day <= days):
                continue
            status = str(row.get("status", default_status)).strip().lower().replace(" ", "_")
            if status in {"dayoff", "day_off", "off", "fixed_off"}:
                status = "unavailable"
            if status not in BAD_AVAILABILITY_STATUS:
                status = default_status
            rows.append({"employee_id": employee_id, "day": day, "status": status})

    seen = set()
    unique = []
    for row in rows:
        key = (row["employee_id"], int(row["day"]))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def generate_employee_availability(
    employee_ids: Sequence[str],
    skills_df: pd.DataFrame,
    demand_daily: pd.DataFrame,
    days: int,
    rng: np.random.Generator,
    input_dir: Path,
    safety_margin: int,
    synthetic_leave_rate: float,
    synthetic_unavailable_rate: float,
    max_leave_days_per_employee: int = 2,
    max_unavailable_days_per_employee: int = 2,
) -> Tuple[pd.DataFrame, List[str]]:
    valid_employee_ids = set(employee_ids)
    rows: List[Dict] = []
    warnings: List[str] = []

    for row in load_real_availability_requests(input_dir, valid_employee_ids, days):
        ok = add_availability_row_if_feasible(rows, row, skills_df, demand_daily, safety_margin)
        if not ok:
            warnings.append(f"Skipped real availability request due to infeasibility or duplicate: {row}")

    for employee_id in employee_ids:
        if rng.random() < synthetic_leave_rate:
            for day in rng.choice(np.arange(1, days + 1), size=int(rng.integers(1, max_leave_days_per_employee + 1)), replace=False):
                add_availability_row_if_feasible(
                    rows,
                    {"employee_id": employee_id, "day": int(day), "status": "leave"},
                    skills_df,
                    demand_daily,
                    safety_margin,
                )
        if rng.random() < synthetic_unavailable_rate:
            for day in rng.choice(np.arange(1, days + 1), size=int(rng.integers(1, max_unavailable_days_per_employee + 1)), replace=False):
                add_availability_row_if_feasible(
                    rows,
                    {"employee_id": employee_id, "day": int(day), "status": "unavailable"},
                    skills_df,
                    demand_daily,
                    safety_margin,
                )

    df = pd.DataFrame(rows, columns=["employee_id", "day", "status"])
    if not df.empty:
        df = df.sort_values(["employee_id", "day"]).reset_index(drop=True)
    return df, warnings


# -----------------------------------------------------------------------------
# Preferences, fixed assignments, history, pairing
# -----------------------------------------------------------------------------


def load_real_preferences(input_dir: Path, valid_employee_ids: set, days: int) -> List[Dict]:
    df = read_optional_csv(input_dir / "preferred_shifts.csv", required_cols=["employee_id", "day", "shift", "preference_type", "penalty"])
    rows = []
    if df.empty:
        return rows
    for _, row in df.iterrows():
        employee_id = str(row["employee_id"]).strip()
        day = int(row["day"])
        shift = normalize_shift(row["shift"])
        pref_type = str(row["preference_type"]).strip().lower()
        if employee_id in valid_employee_ids and 1 <= day <= days and shift in SHIFTS and pref_type in {"prefer", "avoid"}:
            rows.append({
                "employee_id": employee_id,
                "day": day,
                "shift": shift,
                "preference_type": pref_type,
                "penalty": safe_int(row["penalty"], 5),
            })
    return rows


def generate_employee_preferences(
    employee_ids: Sequence[str],
    days: int,
    rng: np.random.Generator,
    input_dir: Path,
    standard_profiles: pd.DataFrame,
    synthetic_prefer_days_per_employee: int,
    synthetic_avoid_days_per_employee: int,
) -> pd.DataFrame:
    valid_employee_ids = set(employee_ids)
    rows = load_real_preferences(input_dir, valid_employee_ids, days)
    seen = {(r["employee_id"], int(r["day"]), r["shift"], r["preference_type"]) for r in rows}
    profile_map = standard_profiles.set_index("employee_id")["standard_shift"].to_dict()

    for employee_id in employee_ids:
        preferred_shift = str(profile_map.get(employee_id, rng.choice(SHIFTS)))
        avoid_shift = "S1" if preferred_shift != "S1" else "S3"
        prefer_days = rng.choice(np.arange(1, days + 1), size=min(days, synthetic_prefer_days_per_employee), replace=False)
        avoid_days = rng.choice(np.arange(1, days + 1), size=min(days, synthetic_avoid_days_per_employee), replace=False)

        for day in prefer_days:
            rec = {"employee_id": employee_id, "day": int(day), "shift": preferred_shift, "preference_type": "prefer", "penalty": 5}
            key = (rec["employee_id"], rec["day"], rec["shift"], rec["preference_type"])
            if key not in seen:
                rows.append(rec)
                seen.add(key)
        for day in avoid_days:
            rec = {"employee_id": employee_id, "day": int(day), "shift": avoid_shift, "preference_type": "avoid", "penalty": 10}
            key = (rec["employee_id"], rec["day"], rec["shift"], rec["preference_type"])
            if key not in seen:
                rows.append(rec)
                seen.add(key)

    df = pd.DataFrame(rows, columns=["employee_id", "day", "shift", "preference_type", "penalty"])
    if not df.empty:
        df = df.sort_values(["employee_id", "day", "shift", "preference_type"]).reset_index(drop=True)
    return df


def generate_assignment_preferences(
    standard_profiles: pd.DataFrame,
    days: int,
    rng: np.random.Generator,
    rows_per_employee: int = 1,
) -> pd.DataFrame:
    rows = []
    for _, profile in standard_profiles.iterrows():
        days_selected = rng.choice(np.arange(1, days + 1), size=min(days, rows_per_employee), replace=False)
        for day in days_selected:
            rows.append({
                "employee_id": profile["employee_id"],
                "day": int(day),
                "shift": profile["standard_shift"],
                "role": profile["standard_role"],
                "preference_type": "prefer",
                "penalty": 3,
            })
    return pd.DataFrame(rows, columns=["employee_id", "day", "shift", "role", "preference_type", "penalty"])


def generate_employee_day_requests(
    employee_ids: Sequence[str],
    days: int,
    rng: np.random.Generator,
    desired_day_off_per_employee: int,
    preferred_work_day_per_employee: int,
) -> pd.DataFrame:
    rows = []
    for employee_id in employee_ids:
        if desired_day_off_per_employee > 0:
            for day in rng.choice(np.arange(1, days + 1), size=min(days, desired_day_off_per_employee), replace=False):
                rows.append({"employee_id": employee_id, "day": int(day), "request_type": "desired_day_off", "penalty": 8, "is_fixed": 0})
        if preferred_work_day_per_employee > 0:
            for day in rng.choice(np.arange(1, days + 1), size=min(days, preferred_work_day_per_employee), replace=False):
                rows.append({"employee_id": employee_id, "day": int(day), "request_type": "preferred_working_day", "penalty": 4, "is_fixed": 0})
    return pd.DataFrame(rows, columns=["employee_id", "day", "request_type", "penalty", "is_fixed"])


def load_fixed_assignments(input_dir: Path, valid_employee_ids: set, days: int) -> pd.DataFrame:
    df = read_optional_csv(input_dir / "fixed_assignments.csv", required_cols=["employee_id", "day", "shift", "role"])
    rows = []
    if df.empty:
        return pd.DataFrame(columns=["employee_id", "day", "shift", "role"])
    for _, row in df.iterrows():
        employee_id = str(row["employee_id"]).strip()
        day = int(row["day"])
        shift = normalize_shift(row["shift"])
        role = str(row["role"]).strip().upper()
        if employee_id in valid_employee_ids and 1 <= day <= days and shift in SHIFTS and role in ROLES:
            rows.append({"employee_id": employee_id, "day": day, "shift": shift, "role": role})
    return pd.DataFrame(rows, columns=["employee_id", "day", "shift", "role"])


def generate_employee_history(
    employee_ids: Sequence[str],
    rng: np.random.Generator,
    max_consecutive_work_days: int,
    max_consecutive_night_shifts: int,
) -> pd.DataFrame:
    rows = []
    for employee_id in employee_ids:
        previous_last_shift = str(rng.choice(["OFF", "S1", "S2", "S3"], p=[0.25, 0.25, 0.30, 0.20]))
        if previous_last_shift == "OFF":
            previous_consecutive_work_days = 0
            previous_consecutive_days_off = int(rng.integers(1, 4))
            previous_consecutive_night_shifts = 0
            previous_last_status = "OFF"
        else:
            previous_consecutive_work_days = int(rng.integers(1, max_consecutive_work_days + 1))
            previous_consecutive_days_off = 0
            previous_consecutive_night_shifts = int(rng.integers(1, max_consecutive_night_shifts + 1)) if previous_last_shift == "S1" else 0
            previous_last_status = "WORK"
        rows.append({
            "employee_id": employee_id,
            "previous_last_shift": previous_last_shift,
            "previous_last_status": previous_last_status,
            "previous_consecutive_work_days": previous_consecutive_work_days,
            "previous_consecutive_days_off": previous_consecutive_days_off,
            "previous_consecutive_night_shifts": previous_consecutive_night_shifts,
        })
    return pd.DataFrame(rows)


def load_or_generate_pairing(
    employee_ids: Sequence[str],
    input_dir: Path,
    rng: np.random.Generator,
    incompatible_pair_rate: float,
    preferred_pair_rate: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid = set(employee_ids)
    incompat_rows: List[Dict] = []
    pair_rows: List[Dict] = []

    inc_real = read_optional_csv(input_dir / "incompatible_pairs.csv", required_cols=["employee_i", "employee_j"])
    if not inc_real.empty:
        for _, row in inc_real.iterrows():
            i = str(row["employee_i"]).strip()
            j = str(row["employee_j"]).strip()
            if i != j and i in valid and j in valid:
                a, b = sorted([i, j])
                priority = str(row.get("priority", "medium")).strip().lower() or "medium"
                penalty = safe_int(row.get("penalty", 100), 100)
                incompat_rows.append({"employee_i": a, "employee_j": b, "incompatibility_type": "same_shift_block", "priority": priority, "penalty": penalty})
                pair_rows.append({"employee_i": a, "employee_j": b, "pair_type": "incompatible", "priority": priority, "penalty": penalty})
    else:
        pair_count = max(0, int(round(len(employee_ids) * incompatible_pair_rate)))
        used = set()
        while len(incompat_rows) < pair_count:
            a, b = sorted(rng.choice(employee_ids, size=2, replace=False).astype(str))
            if a == b or (a, b) in used:
                continue
            used.add((a, b))
            incompat_rows.append({"employee_i": a, "employee_j": b, "incompatibility_type": "same_shift_block", "priority": "medium", "penalty": 100})
            pair_rows.append({"employee_i": a, "employee_j": b, "pair_type": "incompatible", "priority": "medium", "penalty": 100})

    preferred_count = max(0, int(round(len(employee_ids) * preferred_pair_rate)))
    used_pairs = {(r["employee_i"], r["employee_j"]) for r in pair_rows}
    preferred_rows = []
    while len(preferred_rows) < preferred_count:
        a, b = sorted(rng.choice(employee_ids, size=2, replace=False).astype(str))
        if a == b or (a, b) in used_pairs:
            continue
        used_pairs.add((a, b))
        preferred_rows.append({"employee_i": a, "employee_j": b, "pair_type": "preferred", "priority": "low", "penalty": 20})
    pair_rows.extend(preferred_rows)

    incompat_df = pd.DataFrame(incompat_rows, columns=["employee_i", "employee_j", "incompatibility_type", "priority", "penalty"])
    pair_df = pd.DataFrame(pair_rows, columns=["employee_i", "employee_j", "pair_type", "priority", "penalty"])
    return incompat_df.drop_duplicates().reset_index(drop=True), pair_df.drop_duplicates().reset_index(drop=True)


def generate_standard_role_shift_preferences() -> pd.DataFrame:
    return pd.DataFrame([
        {"role": "DAS", "shift": "S2", "preference_type": "prefer", "penalty": 2, "note": "High dining-area workload during daytime"},
        {"role": "DAS", "shift": "S3", "preference_type": "prefer", "penalty": 2, "note": "Evening table-turnover and cleaning support"},
        {"role": "BLO", "shift": "S2", "preference_type": "prefer", "penalty": 2, "note": "Daytime buffet line support"},
        {"role": "BLO", "shift": "S3", "preference_type": "prefer", "penalty": 2, "note": "Evening buffet replenishment"},
        {"role": "DS", "shift": "S2", "preference_type": "prefer", "penalty": 1, "note": "Clean-ware recovery support"},
        {"role": "RS", "shift": "S1", "preference_type": "avoid", "penalty": 1, "note": "Lower reception pressure at night"},
    ])


# -----------------------------------------------------------------------------
# Extended support tables
# -----------------------------------------------------------------------------


def build_shift_structure(min_rest_hours: float) -> pd.DataFrame:
    rows = []
    for shift in SHIFTS:
        rows.append({
            "shift": shift,
            "shift_time": SHIFT_TIME[shift],
            "start_time": minute_to_clock(SHIFT_START_MIN[shift]),
            "end_time": minute_to_clock(SHIFT_END_MIN[shift]),
            "start_minute": SHIFT_START_MIN[shift],
            "end_minute": SHIFT_END_MIN[shift],
            "duration_hours": 8,
            "is_night_shift": 1 if shift == NIGHT_SHIFT else 0,
            "min_rest_hours": min_rest_hours,
        })
    return pd.DataFrame(rows)


def build_shift_transition_rest(min_rest_hours: float) -> pd.DataFrame:
    rows = []
    min_rest_minutes = min_rest_hours * 60
    for from_shift in SHIFTS:
        for to_shift in SHIFTS:
            rest_minutes = (24 * 60 - SHIFT_END_MIN[from_shift]) + SHIFT_START_MIN[to_shift]
            rows.append({
                "from_shift": from_shift,
                "to_shift_next_day": to_shift,
                "rest_hours": round(rest_minutes / 60, 2),
                "min_rest_hours": min_rest_hours,
                "allowed": 1 if rest_minutes >= min_rest_minutes else 0,
                "is_forbidden_successession": 0 if rest_minutes >= min_rest_minutes else 1,
            })
    return pd.DataFrame(rows)


def build_penalty_config() -> pd.DataFrame:
    return pd.DataFrame([
        {"penalty_component": "coverage_shortage", "weight": 1000, "meaning": "Highest operational penalty"},
        {"penalty_component": "target_shortage", "weight": 300, "meaning": "Below target but above minimum"},
        {"penalty_component": "overstaff", "weight": 50, "meaning": "Labor efficiency penalty"},
        {"penalty_component": "skill_violation", "weight": 100000, "meaning": "Hard infeasibility"},
        {"penalty_component": "availability_violation", "weight": 100000, "meaning": "Hard infeasibility"},
        {"penalty_component": "duplicate_assignment", "weight": 100000, "meaning": "Hard infeasibility"},
        {"penalty_component": "rest_violation", "weight": 800, "meaning": "Fatigue and legal feasibility"},
        {"penalty_component": "consecutive_night_violation", "weight": 600, "meaning": "Fatigue control"},
        {"penalty_component": "night_excess", "weight": 300, "meaning": "Monthly night fairness"},
        {"penalty_component": "workday_or_hour_violation", "weight": 200, "meaning": "Contract balance"},
        {"penalty_component": "preference_violation", "weight": 10, "meaning": "Employee satisfaction"},
        {"penalty_component": "incompatibility", "weight": 100, "meaning": "Team compatibility"},
        {"penalty_component": "standard_profile_change", "weight": 3, "meaning": "Assignment stability"},
    ])


def build_holiday_calendar(start_day: int, days: int, holiday_days: Sequence[int]) -> pd.DataFrame:
    rows = []
    holidays = set(int(d) for d in holiday_days if 1 <= int(d) <= days)
    for day in range(1, days + 1):
        # This assumes day 1 is a weekday index start_day, where Monday=0 and Sunday=6.
        weekday = (start_day + day - 1) % 7
        rows.append({
            "day": day,
            "weekday_index": weekday,
            "is_saturday": 1 if weekday == 5 else 0,
            "is_sunday": 1 if weekday == 6 else 0,
            "is_weekend": 1 if weekday in {5, 6} else 0,
            "is_public_holiday": 1 if day in holidays else 0,
        })
    return pd.DataFrame(rows)


def build_weekend_policy(max_working_weekends: int, complete_weekend_required: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "policy_name": "default_weekend_policy",
        "max_working_weekends": max_working_weekends,
        "complete_weekend_required": int(complete_weekend_required),
        "saturday_sunday_should_match": int(complete_weekend_required),
    }])


# -----------------------------------------------------------------------------
# Validation report
# -----------------------------------------------------------------------------


def build_validation_report(
    master_df: pd.DataFrame,
    skills_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    preferences_df: pd.DataFrame,
    history_df: pd.DataFrame,
    incompat_df: pd.DataFrame,
    demand_min: pd.DataFrame,
    demand_milp: pd.DataFrame,
    coverage_bands: pd.DataFrame,
    skill_targets: pd.DataFrame,
    headcount_info: Dict[str, int | float | str],
    availability_warnings: List[str],
    safety_margin: int,
) -> pd.DataFrame:
    rows = []
    n = len(master_df)
    min_work_days = int(master_df["min_work_days"].min())
    max_work_days = int(master_df["max_work_days"].max())
    min_slots = int(demand_min[ROLES].sum().sum())
    milp_slots = int(demand_milp[ROLES].sum().sum())

    rows.append({
        "check": "minimum_coverage_vs_contract_minimum",
        "status": "WARN" if min_slots < n * min_work_days else "PASS",
        "detail": f"C_min slots={min_slots}; employee minimum requirement={n}*{min_work_days}={n * min_work_days}. "
                  "If WARN, C_min is only minimum service coverage and must not be used as final roster demand.",
    })
    rows.append({
        "check": "milp_demand_contract_feasibility",
        "status": "PASS" if n * min_work_days <= milp_slots <= n * max_work_days else "FAIL",
        "detail": f"{n}*{min_work_days} <= {milp_slots} <= {n}*{max_work_days}",
    })
    rows.append({
        "check": "coverage_band_order",
        "status": "PASS" if all((coverage_bands[f"{r}_min"] <= coverage_bands[f"{r}_opt"]).all() and (coverage_bands[f"{r}_opt"] <= coverage_bands[f"{r}_max"]).all() for r in ROLES) else "FAIL",
        "detail": "For every role and cell: C_min <= C_opt <= C_max",
    })
    for role in ROLES:
        actual = int(skills_df[role].sum())
        target = int(skill_targets.loc[skill_targets["role"] == role, "pool_target"].iloc[0])
        rows.append({
            "check": f"skill_pool_{role}",
            "status": "PASS" if actual >= target else "FAIL",
            "detail": f"qualified={actual}, target={target}",
        })
    zero_skill = int((skills_df[ROLES].sum(axis=1) == 0).sum())
    rows.append({"check": "no_zero_skill_employee", "status": "PASS" if zero_skill == 0 else "FAIL", "detail": f"zero_skill_count={zero_skill}"})

    feasible, errors = availability_is_feasible(
        availability_df.to_dict("records") if not availability_df.empty else [],
        skills_df,
        demand_milp,
        safety_margin,
    )
    rows.append({
        "check": "daily_availability_sufficiency_for_milp_demand",
        "status": "PASS" if feasible else "FAIL",
        "detail": "; ".join(errors[:5]) if errors else "available qualified pool is sufficient",
    })
    invalid_pref = set(preferences_df["preference_type"].unique()) - {"prefer", "avoid"} if not preferences_df.empty else set()
    rows.append({"check": "preference_sanity", "status": "PASS" if not invalid_pref else "FAIL", "detail": f"rows={len(preferences_df)}, invalid_types={sorted(invalid_pref)}"})
    valid_history = set(history_df["previous_last_shift"].unique()) <= {"OFF", "S1", "S2", "S3"}
    rows.append({"check": "history_previous_shift_values", "status": "PASS" if valid_history else "FAIL", "detail": f"values={sorted(history_df['previous_last_shift'].unique())}"})
    rows.append({"check": "incompatible_pairs_generated", "status": "INFO", "detail": f"pairs={len(incompat_df)}"})
    if availability_warnings:
        rows.append({"check": "availability_warnings", "status": "WARN", "detail": " | ".join(availability_warnings[:10])})
    rows.append({"check": "headcount_and_demand_summary", "status": "INFO", "detail": ", ".join(f"{k}={v}" for k, v in headcount_info.items())})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate corrected employee MILP input CSV files.")
    parser.add_argument("--input", default=r"E:\NĂM 4\Capstone\Data_input\Final_data_input_milp.xlsx")
    parser.add_argument("--output-dir", default=r"E:\NĂM 4\Capstone\Data_input\employee_milp_inputs")
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)

    # Contract rules.
    parser.add_argument("--min-work-days", type=int, default=20)
    parser.add_argument("--max-work-days", type=int, default=24)
    parser.add_argument("--max-night-shifts", type=int, default=6)
    parser.add_argument("--max-consecutive-work-days", type=int, default=6)
    parser.add_argument("--max-consecutive-night-shifts", type=int, default=3)
    parser.add_argument("--min-consecutive-days-off", type=int, default=1)
    parser.add_argument("--min-days-off-after-work-block", type=int, default=1)
    parser.add_argument("--night-block-separation", type=int, default=1)
    parser.add_argument("--max-working-weekends", type=int, default=4)
    parser.add_argument("--max-working-public-holidays", type=int, default=2)
    parser.add_argument("--complete-weekend-required", type=int, default=0)

    # Headcount and coverage correction.
    parser.add_argument("--headcount-buffer", type=float, default=0.10)
    parser.add_argument("--employee-count", type=int, default=None, help="Manual employee count. Use this when HR headcount is fixed.")
    parser.add_argument("--minimum-employee-count", type=int, default=48, help="Operational minimum employee pool for robustness.")
    parser.add_argument("--target-total-slots", type=int, default=None, help="Force C_opt total slots, e.g., 1064.")
    parser.add_argument("--target-coverage-buffer-rate", type=float, default=0.15, help="Applied if target-total-slots is not provided.")
    parser.add_argument("--max-coverage-buffer-rate", type=float, default=0.15)
    parser.add_argument("--milp-demand-level", choices=["min", "opt", "max"], default="opt")
    parser.add_argument(
        "--demand-source",
        choices=["auto", "staffing_pivot", "daily_workload"],
        default="daily_workload",
        help="daily_workload uses role_daily_shift_workload and can vary by day; staffing_pivot repeats the same shift-role demand across all days; auto tries daily_workload then falls back.",
    )
    parser.add_argument(
        "--daily-required-column",
        choices=["auto", "required_staff_by_workload", "recommended_staff_initial", "staff_required_p95_workload", "staff_required_mean_workload", "required_staff"],
        default="required_staff_by_workload",
        help="Column used from role_daily_shift_workload. Use required_staff_by_workload for stronger day-by-day variability; use recommended_staff_initial if you want the policy-adjusted recommendation.",
    )
    parser.add_argument("--min-staff", type=int, default=1, help="Mandatory minimum staff per role when daily workload is missing or rounds to zero.")
    parser.add_argument("--allow-demand-source-fallback", action="store_true", help="If daily_workload is missing, fallback to staffing_pivot instead of raising an error.")
    parser.add_argument("--role-buffer-weights", default="", help="Example: DAS=4,BLO=2,DS=1.5")
    parser.add_argument("--shift-buffer-weights", default="", help="Example: S1=0.75,S2=1.25,S3=1.25")

    # Skills.
    parser.add_argument("--skill-buffer", type=float, default=0.20)
    parser.add_argument("--skill-multiplier", type=float, default=1.05)
    parser.add_argument("--das-universal", action="store_true")

    # Availability and preferences.
    parser.add_argument("--availability-safety-margin", type=int, default=1)
    parser.add_argument("--synthetic-leave-rate", type=float, default=0.14)
    parser.add_argument("--synthetic-unavailable-rate", type=float, default=0.25)
    parser.add_argument("--prefer-days-per-employee", type=int, default=3)
    parser.add_argument("--avoid-days-per-employee", type=int, default=2)
    parser.add_argument("--assignment-pref-rows-per-employee", type=int, default=1)
    parser.add_argument("--desired-day-off-per-employee", type=int, default=1)
    parser.add_argument("--preferred-work-day-per-employee", type=int, default=1)
    parser.add_argument("--incompatible-pair-rate", type=float, default=0.04)
    parser.add_argument("--preferred-pair-rate", type=float, default=0.03)

    # Shift/rest and calendar.
    parser.add_argument("--min-rest-hours", type=float, default=12.0)
    parser.add_argument("--calendar-start-weekday", type=int, default=0, help="Monday=0, Sunday=6 for day 1.")
    parser.add_argument("--holiday-days", type=int, nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    input_path = Path(args.input)
    input_dir = input_path if input_path.is_dir() else input_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    role_weights = parse_weight_string(args.role_buffer_weights, DEFAULT_ROLE_BUFFER_WEIGHTS, ROLES)
    shift_weights = parse_weight_string(args.shift_buffer_weights, DEFAULT_SHIFT_BUFFER_WEIGHTS, SHIFTS)

    workbook = find_workbook(args.input)
    demand_min, workload_source = build_minimum_demand_from_source(args, workbook)
    total_min_slots = int(demand_min[ROLES].sum().sum())

    employee_count_initial, headcount_info = initial_employee_count(
        total_min_slots=total_min_slots,
        min_work_days=args.min_work_days,
        max_work_days=args.max_work_days,
        headcount_buffer=args.headcount_buffer,
        manual_employee_count=args.employee_count,
        minimum_employee_count=args.minimum_employee_count,
    )

    employee_count, target_total_slots = adjust_employee_count_until_feasible(
        total_min_slots=total_min_slots,
        employee_count=employee_count_initial,
        min_work_days=args.min_work_days,
        max_work_days=args.max_work_days,
        target_coverage_buffer_rate=args.target_coverage_buffer_rate,
        target_total_slots=args.target_total_slots,
        manual_employee_count=args.employee_count,
    )
    headcount_info.update({
        "employees_generated": employee_count,
        "target_total_slots_C_opt": target_total_slots,
        "milp_demand_level_exported": args.milp_demand_level,
    })

    coverage_bands, target_total_slots = build_coverage_bands(
        demand_min=demand_min,
        employee_count=employee_count,
        min_work_days=args.min_work_days,
        max_work_days=args.max_work_days,
        target_coverage_buffer_rate=args.target_coverage_buffer_rate,
        target_total_slots=target_total_slots,
        max_coverage_buffer_rate=args.max_coverage_buffer_rate,
        role_weights=role_weights,
        shift_weights=shift_weights,
    )
    demand_milp = demand_for_milp_from_bands(coverage_bands, args.milp_demand_level)

    master_df = generate_employee_master(
        n=employee_count,
        min_work_days=args.min_work_days,
        max_work_days=args.max_work_days,
        max_night_shifts=args.max_night_shifts,
        max_consecutive_work_days=args.max_consecutive_work_days,
        max_consecutive_night_shifts=args.max_consecutive_night_shifts,
        min_consecutive_days_off=args.min_consecutive_days_off,
        min_days_off_after_work_block=args.min_days_off_after_work_block,
        night_block_separation=args.night_block_separation,
        max_working_weekends=args.max_working_weekends,
        max_working_public_holidays=args.max_working_public_holidays,
        complete_weekend_required=args.complete_weekend_required,
    )
    employee_ids = master_df["employee_id"].tolist()

    skill_targets = compute_skill_pool_targets(demand_milp, args.max_work_days, args.skill_buffer)
    skills_df = generate_employee_skills(
        employee_ids=employee_ids,
        skill_targets=skill_targets,
        demand_daily=demand_milp,
        rng=rng,
        das_universal=args.das_universal,
        skill_multiplier=args.skill_multiplier,
    )
    standard_profiles = generate_standard_profiles(skills_df, demand_milp, rng)
    master_df = master_df.merge(standard_profiles, on="employee_id", how="left")

    availability_df, availability_warnings = generate_employee_availability(
        employee_ids=employee_ids,
        skills_df=skills_df,
        demand_daily=demand_milp,
        days=args.days,
        rng=rng,
        input_dir=input_dir,
        safety_margin=args.availability_safety_margin,
        synthetic_leave_rate=args.synthetic_leave_rate,
        synthetic_unavailable_rate=args.synthetic_unavailable_rate,
    )
    preferences_df = generate_employee_preferences(
        employee_ids=employee_ids,
        days=args.days,
        rng=rng,
        input_dir=input_dir,
        standard_profiles=standard_profiles,
        synthetic_prefer_days_per_employee=args.prefer_days_per_employee,
        synthetic_avoid_days_per_employee=args.avoid_days_per_employee,
    )
    assignment_pref_df = generate_assignment_preferences(
        standard_profiles=standard_profiles,
        days=args.days,
        rng=rng,
        rows_per_employee=args.assignment_pref_rows_per_employee,
    )
    day_requests_df = generate_employee_day_requests(
        employee_ids=employee_ids,
        days=args.days,
        rng=rng,
        desired_day_off_per_employee=args.desired_day_off_per_employee,
        preferred_work_day_per_employee=args.preferred_work_day_per_employee,
    )
    fixed_assignments_df = load_fixed_assignments(input_dir, set(employee_ids), args.days)
    history_df = generate_employee_history(
        employee_ids=employee_ids,
        rng=rng,
        max_consecutive_work_days=args.max_consecutive_work_days,
        max_consecutive_night_shifts=args.max_consecutive_night_shifts,
    )
    incompat_df, pairing_df = load_or_generate_pairing(
        employee_ids=employee_ids,
        input_dir=input_dir,
        rng=rng,
        incompatible_pair_rate=args.incompatible_pair_rate,
        preferred_pair_rate=args.preferred_pair_rate,
    )

    shift_structure_df = build_shift_structure(args.min_rest_hours)
    shift_transition_df = build_shift_transition_rest(args.min_rest_hours)
    penalty_config_df = build_penalty_config()
    role_shift_pref_df = generate_standard_role_shift_preferences()
    holiday_calendar_df = build_holiday_calendar(args.calendar_start_weekday, args.days, args.holiday_days)
    weekend_policy_df = build_weekend_policy(args.max_working_weekends, args.complete_weekend_required)

    validation_df = build_validation_report(
        master_df=master_df,
        skills_df=skills_df,
        availability_df=availability_df,
        preferences_df=preferences_df,
        history_df=history_df,
        incompat_df=incompat_df,
        demand_min=demand_min,
        demand_milp=demand_milp,
        coverage_bands=coverage_bands,
        skill_targets=skill_targets,
        headcount_info=headcount_info,
        availability_warnings=availability_warnings,
        safety_margin=args.availability_safety_margin,
    )

    # Save core files.
    master_df.to_csv(output_dir / "employee_master.csv", index=False, encoding="utf-8-sig")
    skills_df.to_csv(output_dir / "employee_skills.csv", index=False, encoding="utf-8-sig")
    availability_df.to_csv(output_dir / "employee_availability.csv", index=False, encoding="utf-8-sig")
    preferences_df.to_csv(output_dir / "employee_preferences.csv", index=False, encoding="utf-8-sig")
    history_df.to_csv(output_dir / "employee_history.csv", index=False, encoding="utf-8-sig")
    incompat_df.to_csv(output_dir / "employee_incompatibility.csv", index=False, encoding="utf-8-sig")
    role_shift_pref_df.to_csv(output_dir / "standard_role_shift_preferences.csv", index=False, encoding="utf-8-sig")
    demand_milp.to_csv(output_dir / "staffing_demand_daily.csv", index=False, encoding="utf-8-sig")
    skill_targets.to_csv(output_dir / "skill_pool_targets.csv", index=False, encoding="utf-8-sig")
    validation_df.to_csv(output_dir / "generation_validation_report.csv", index=False, encoding="utf-8-sig")

    # Save extended formulation support files.
    coverage_bands.to_csv(output_dir / "staffing_coverage_bands.csv", index=False, encoding="utf-8-sig")
    coverage_bands_long(coverage_bands).to_csv(output_dir / "staffing_coverage_bands_long.csv", index=False, encoding="utf-8-sig")
    shift_structure_df.to_csv(output_dir / "shift_structure.csv", index=False, encoding="utf-8-sig")
    shift_transition_df.to_csv(output_dir / "shift_transition_rest.csv", index=False, encoding="utf-8-sig")
    penalty_config_df.to_csv(output_dir / "penalty_config.csv", index=False, encoding="utf-8-sig")
    standard_profiles.to_csv(output_dir / "employee_standard_profile.csv", index=False, encoding="utf-8-sig")
    assignment_pref_df.to_csv(output_dir / "employee_assignment_preferences.csv", index=False, encoding="utf-8-sig")
    day_requests_df.to_csv(output_dir / "employee_day_requests.csv", index=False, encoding="utf-8-sig")
    fixed_assignments_df.to_csv(output_dir / "employee_fixed_assignments.csv", index=False, encoding="utf-8-sig")
    pairing_df.to_csv(output_dir / "employee_pairing.csv", index=False, encoding="utf-8-sig")
    holiday_calendar_df.to_csv(output_dir / "holiday_calendar.csv", index=False, encoding="utf-8-sig")
    weekend_policy_df.to_csv(output_dir / "weekend_policy.csv", index=False, encoding="utf-8-sig")

    print("\nDONE: corrected employee MILP input files generated")
    print(f"Demand source requested  : {args.demand_source}")
    print(f"Workload source used     : {workload_source}")
    print(f"Workbook used            : {workbook if workbook else 'None - fallback demand used'}")
    print(f"Output directory         : {output_dir.resolve()}")
    print(f"Days                     : {args.days}")
    print(f"Minimum coverage slots   : {total_min_slots}")
    print(f"Target C_opt slots       : {int(demand_milp[ROLES].sum().sum())}")
    print(f"Employees generated      : {employee_count}")
    print(f"Contract check           : {employee_count * args.min_work_days} <= {int(demand_milp[ROLES].sum().sum())} <= {employee_count * args.max_work_days}")
    print(f"MILP demand level         : {args.milp_demand_level}")
    print("\nValidation summary:")
    print(validation_df.to_string(index=False))


if __name__ == "__main__":
    main()
