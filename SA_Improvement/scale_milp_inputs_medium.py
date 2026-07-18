r"""
scale_milp_inputs_medium.py

Create MEDIUM input folder for the airport lounge MILP/SA validation.

This script takes the current SMALL baseline employee_milp_inputs folder and
creates a scaled folder by:
  1) extending the planning horizon,
  2) multiplying C_min, C_opt, C_max staffing bands,
  3) resizing/cloning employee resources,
  4) updating monthly workday/hour bounds,
  5) exporting MILP-compatible CSV files.

Fixed MEDIUM preset:
  - 56 days, demand multiplier 1.50, 70 employees

Example:
python scale_milp_inputs_medium.py ^
  --baseline-input-dir "E:\NĂM 4\Capstone\Data_input\employee_milp_inputs" ^
  --output-dir "E:\NĂM 4\Capstone\SA_Improvement\employee_milp_inputs_medium"
"""
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROLES = ["RS", "DAS", "BLO", "FSTC", "DS", "SLS"]
SHIFTS = ["S1", "S2", "S3"]
SHIFT_TIME = {"S1": "00:00-08:00", "S2": "08:00-16:00", "S3": "16:00-24:00"}

# Default paths for running directly without CLI arguments.
# Change these two lines if your project folder is different.
DEFAULT_BASELINE_INPUT_DIR = Path(r"E:\NĂM 4\Capstone\Data_input\employee_milp_inputs")
DEFAULT_OUTPUT_DIR = Path(r"E:\NĂM 4\Capstone\SA_Improvement\employee_milp_inputs_medium")

OPTIONAL_FILE_COLUMNS = {
    "employee_availability.csv": ["employee_id", "day", "status"],
    "employee_preferences.csv": ["employee_id", "day", "shift", "preference_type", "penalty"],
    "employee_assignment_preferences.csv": ["employee_id", "day", "shift", "role", "preference_type", "penalty"],
    "employee_day_requests.csv": ["employee_id", "day", "request_type", "penalty", "is_fixed"],
    "employee_fixed_assignments.csv": ["employee_id", "day", "shift", "role"],
    "employee_history.csv": [
        "employee_id", "previous_last_shift", "previous_last_status",
        "previous_consecutive_work_days", "previous_consecutive_days_off",
        "previous_consecutive_night_shifts",
    ],
    "employee_incompatibility.csv": ["employee_i", "employee_j", "incompatibility_type", "priority", "penalty"],
    "employee_pairing.csv": ["employee_i", "employee_j", "pair_type", "priority", "penalty"],
    "employee_standard_profile.csv": ["employee_id", "standard_role", "standard_shift"],
    "standard_role_shift_preferences.csv": ["role", "shift", "preference_type", "penalty", "note"],
    "weekend_policy.csv": ["policy_name", "max_working_weekends", "complete_weekend_required", "saturday_sunday_should_match"],
}


@dataclass(frozen=True)
class ScalePreset:
    name: str
    days: int
    demand_multiplier: float
    employees: int
    min_work_days: int
    max_work_days: int
    target_work_days: int
    expected_customers_per_day: int


PRESETS: Dict[str, ScalePreset] = {
    "medium": ScalePreset(
        name="medium", days=28, demand_multiplier=1.50, employees=70,
        min_work_days=20, max_work_days=24, target_work_days=22,
        expected_customers_per_day=1350,
    ),
    
}


def ceil_int(x: float, minimum: int = 0) -> int:
    try:
        if pd.isna(x):
            return minimum
        return max(minimum, int(math.ceil(float(x))))
    except Exception:
        return minimum


def read_csv(path: Path, required: bool = True, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return pd.DataFrame(columns=columns or [])
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_shift(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().upper().replace("SHIFT", "").replace("_", "").replace("-", "")
    if s in {"1", "01"}: return "S1"
    if s in {"2", "02"}: return "S2"
    if s in {"3", "03"}: return "S3"
    return s


def ensure_coverage_long(input_dir: Path) -> pd.DataFrame:
    """Load staffing_coverage_bands_long.csv, or convert staffing_coverage_bands.csv to long."""
    long_path = input_dir / "staffing_coverage_bands_long.csv"
    if long_path.exists():
        df = read_csv(long_path)
        required = {"day", "shift", "role", "C_min", "C_opt", "C_max"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{long_path} missing columns: {missing}")
        if "shift_time" not in df.columns:
            df["shift_time"] = df["shift"].map(SHIFT_TIME)
        df["shift"] = df["shift"].apply(normalize_shift)
        df["role"] = df["role"].astype(str).str.strip().str.upper()
        return df[["day", "shift", "shift_time", "role", "C_min", "C_opt", "C_max"]].copy()

    wide_path = input_dir / "staffing_coverage_bands.csv"
    if not wide_path.exists():
        raise FileNotFoundError("Need either staffing_coverage_bands_long.csv or staffing_coverage_bands.csv")
    wide = read_csv(wide_path)
    rows = []
    for _, row in wide.iterrows():
        day = ceil_int(row.get("day"), 1)
        shift = normalize_shift(row.get("shift"))
        shift_time = row.get("shift_time", SHIFT_TIME.get(shift, ""))
        for role in ROLES:
            rows.append({
                "day": day,
                "shift": shift,
                "shift_time": shift_time,
                "role": role,
                "C_min": ceil_int(row.get(f"{role}_min"), 1),
                "C_opt": ceil_int(row.get(f"{role}_opt"), 1),
                "C_max": ceil_int(row.get(f"{role}_max"), 1),
            })
    return pd.DataFrame(rows)


def scale_coverage(coverage: pd.DataFrame, preset: ScalePreset, min_each_role: int = 1) -> pd.DataFrame:
    base_days = int(coverage["day"].max())
    base = coverage.copy()
    base["day"] = pd.to_numeric(base["day"], errors="coerce").fillna(1).astype(int)
    base["shift"] = base["shift"].apply(normalize_shift)
    base["role"] = base["role"].astype(str).str.strip().str.upper()

    rows = []
    for new_day in range(1, preset.days + 1):
        pattern_day = ((new_day - 1) % base_days) + 1
        day_rows = base[base["day"] == pattern_day]
        for _, row in day_rows.iterrows():
            c_min = ceil_int(float(row["C_min"]) * preset.demand_multiplier, min_each_role)
            c_opt = ceil_int(float(row["C_opt"]) * preset.demand_multiplier, c_min)
            c_max = ceil_int(float(row["C_max"]) * preset.demand_multiplier, c_opt)
            c_opt = max(c_opt, c_min)
            c_max = max(c_max, c_opt)
            rows.append({
                "day": new_day,
                "shift": normalize_shift(row["shift"]),
                "shift_time": row.get("shift_time", SHIFT_TIME.get(normalize_shift(row["shift"]), "")),
                "role": row["role"],
                "C_min": c_min,
                "C_opt": c_opt,
                "C_max": c_max,
                "source_pattern_day": pattern_day,
                "demand_multiplier": preset.demand_multiplier,
            })
    out = pd.DataFrame(rows)
    return out.sort_values(["day", "shift", "role"]).reset_index(drop=True)


def coverage_long_to_wide(coverage_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (day, shift, shift_time), grp in coverage_long.groupby(["day", "shift", "shift_time"], sort=True):
        rec = {"day": int(day), "shift": shift, "shift_time": shift_time}
        total_min = total_opt = total_max = 0
        role_map = grp.set_index("role").to_dict("index")
        for role in ROLES:
            cmin = ceil_int(role_map.get(role, {}).get("C_min", 0), 0)
            copt = ceil_int(role_map.get(role, {}).get("C_opt", 0), 0)
            cmax = ceil_int(role_map.get(role, {}).get("C_max", 0), 0)
            rec[f"{role}_min"] = cmin
            rec[f"{role}_opt"] = copt
            rec[f"{role}_max"] = cmax
            total_min += cmin; total_opt += copt; total_max += cmax
        rec["total_min"] = total_min
        rec["total_opt"] = total_opt
        rec["total_max"] = total_max
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["day", "shift"]).reset_index(drop=True)


def coverage_opt_to_demand_daily(coverage_long: pd.DataFrame) -> pd.DataFrame:
    wide_rows = []
    for (day, shift, shift_time), grp in coverage_long.groupby(["day", "shift", "shift_time"], sort=True):
        rec = {"day": int(day), "shift": shift, "shift_time": shift_time}
        total = 0
        for role in ROLES:
            val = int(grp.loc[grp["role"] == role, "C_opt"].max()) if (grp["role"] == role).any() else 0
            rec[role] = val
            total += val
        rec["total_staff"] = total
        wide_rows.append(rec)
    return pd.DataFrame(wide_rows).sort_values(["day", "shift"]).reset_index(drop=True)


def resize_employees(input_dir: Path, output_dir: Path, preset: ScalePreset) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master = read_csv(input_dir / "employee_master.csv")
    skills = read_csv(input_dir / "employee_skills.csv")
    standard = read_csv(input_dir / "employee_standard_profile.csv", required=False, columns=["employee_id", "standard_role", "standard_shift"])

    if "employee_id" not in master.columns:
        raise ValueError("employee_master.csv must contain employee_id")
    if "employee_id" not in skills.columns:
        raise ValueError("employee_skills.csv must contain employee_id")

    master = master.copy()
    master["employee_id"] = master["employee_id"].astype(str)
    skills = skills.copy()
    skills["employee_id"] = skills["employee_id"].astype(str)
    if not standard.empty:
        standard = standard.copy()
        standard["employee_id"] = standard["employee_id"].astype(str)

    base_ids = master["employee_id"].tolist()
    if not base_ids:
        raise ValueError("employee_master.csv has no employees")

    # Preserve original employees first, then clone by cycling source rows.
    selected_rows = []
    skill_rows = []
    standard_rows = []
    skill_by_id = skills.set_index("employee_id").to_dict("index")
    std_by_id = standard.set_index("employee_id").to_dict("index") if not standard.empty else {}

    for i in range(preset.employees):
        src_idx = i % len(base_ids)
        src_id = base_ids[src_idx]
        src_master = master.iloc[src_idx].copy()
        if i < len(base_ids):
            new_id = src_id
        else:
            new_id = f"{preset.name.upper()}_E{i+1:03d}"
        src_master["employee_id"] = new_id
        if "employee_name" in src_master.index:
            src_master["employee_name"] = str(src_master.get("employee_name", new_id)) if i < len(base_ids) else f"{preset.name.title()} Employee {i+1:03d}"

        # Scale contract limits to horizon length.
        updates = {
            "min_work_days": preset.min_work_days,
            "max_work_days": preset.max_work_days,
            "target_work_days": preset.target_work_days,
            "min_hours": preset.min_work_days * 8,
            "max_hours": preset.max_work_days * 8,
            "target_hours": preset.target_work_days * 8,
            "min_days_off": preset.days - preset.max_work_days,
            "max_days_off": preset.days - preset.min_work_days,
            "max_night_shifts": max(1, math.ceil(preset.max_work_days * 0.35)),
            "max_consecutive_work_days": 6,
            "max_consecutive_night_shifts": 3,
        }
        for c, v in updates.items():
            src_master[c] = v
        selected_rows.append(src_master)

        src_skill = dict(skill_by_id.get(src_id, {}))
        src_skill["employee_id"] = new_id
        for r in ROLES:
            src_skill[r] = int(float(src_skill.get(r, 0) or 0))
        # Safety: every employee must have at least one role.
        if sum(src_skill.get(r, 0) for r in ROLES) == 0:
            src_skill[ROLES[i % len(ROLES)]] = 1
        skill_rows.append(src_skill)

        if std_by_id:
            src_std = dict(std_by_id.get(src_id, {}))
            src_std["employee_id"] = new_id
            if not src_std.get("standard_role"):
                skilled_roles = [r for r in ROLES if src_skill.get(r, 0) == 1]
                src_std["standard_role"] = skilled_roles[0] if skilled_roles else ROLES[i % len(ROLES)]
            if not src_std.get("standard_shift"):
                src_std["standard_shift"] = SHIFTS[i % len(SHIFTS)]
            standard_rows.append(src_std)

    out_master = pd.DataFrame(selected_rows).reset_index(drop=True)
    out_skills = pd.DataFrame(skill_rows).reset_index(drop=True)
    out_standard = pd.DataFrame(standard_rows).reset_index(drop=True) if standard_rows else pd.DataFrame(columns=["employee_id", "standard_role", "standard_shift"])

    out_master.to_csv(output_dir / "employee_master.csv", index=False, encoding="utf-8-sig")
    out_skills.to_csv(output_dir / "employee_skills.csv", index=False, encoding="utf-8-sig")
    out_standard.to_csv(output_dir / "employee_standard_profile.csv", index=False, encoding="utf-8-sig")
    return out_master, out_skills, out_standard


def write_shift_and_required_files(input_dir: Path, output_dir: Path, preset: ScalePreset) -> None:
    # Required static files for MILP.
    for name in ["shift_structure.csv", "shift_transition_rest.csv", "penalty_config.csv"]:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
        elif name == "shift_structure.csv":
            pd.DataFrame([
                {"shift": "S1", "shift_time": "00:00-08:00", "duration_hours": 8, "is_night_shift": 1},
                {"shift": "S2", "shift_time": "08:00-16:00", "duration_hours": 8, "is_night_shift": 0},
                {"shift": "S3", "shift_time": "16:00-24:00", "duration_hours": 8, "is_night_shift": 0},
            ]).to_csv(output_dir / name, index=False, encoding="utf-8-sig")
        elif name == "shift_transition_rest.csv":
            pd.DataFrame([
                {"from_shift": "S1", "to_shift_next_day": "S1", "allowed": 1, "is_forbidden_successession": 0},
                {"from_shift": "S1", "to_shift_next_day": "S2", "allowed": 1, "is_forbidden_successession": 0},
                {"from_shift": "S1", "to_shift_next_day": "S3", "allowed": 1, "is_forbidden_successession": 0},
                {"from_shift": "S2", "to_shift_next_day": "S1", "allowed": 0, "is_forbidden_successession": 1},
                {"from_shift": "S2", "to_shift_next_day": "S2", "allowed": 1, "is_forbidden_successession": 0},
                {"from_shift": "S2", "to_shift_next_day": "S3", "allowed": 1, "is_forbidden_successession": 0},
                {"from_shift": "S3", "to_shift_next_day": "S1", "allowed": 0, "is_forbidden_successession": 1},
                {"from_shift": "S3", "to_shift_next_day": "S2", "allowed": 1, "is_forbidden_successession": 0},
                {"from_shift": "S3", "to_shift_next_day": "S3", "allowed": 1, "is_forbidden_successession": 0},
            ]).to_csv(output_dir / name, index=False, encoding="utf-8-sig")
        elif name == "penalty_config.csv":
            pd.DataFrame([
                {"penalty_component": "coverage_shortage", "weight": 1000},
                {"penalty_component": "target_shortage", "weight": 500},
                {"penalty_component": "overstaff", "weight": 50},
                {"penalty_component": "preference_violation", "weight": 20},
                {"penalty_component": "fairness", "weight": 15},
                {"penalty_component": "stability", "weight": 10},
                {"penalty_component": "rest_violation", "weight": 800},
            ]).to_csv(output_dir / name, index=False, encoding="utf-8-sig")


def write_optional_files(input_dir: Path, output_dir: Path, employees: pd.DataFrame, preset: ScalePreset) -> None:
    employee_ids = set(employees["employee_id"].astype(str))

    for name, cols in OPTIONAL_FILE_COLUMNS.items():
        if name == "employee_standard_profile.csv":
            continue  # written by resize_employees
        src = input_dir / name
        if not src.exists():
            pd.DataFrame(columns=cols).to_csv(output_dir / name, index=False, encoding="utf-8-sig")
            continue
        df = pd.read_csv(src, encoding="utf-8-sig")
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[cols].copy()
        # Keep only rows that still refer to existing original employees.
        if "employee_id" in df.columns:
            df = df[df["employee_id"].astype(str).isin(employee_ids)]
        if "employee_i" in df.columns:
            df = df[df["employee_i"].astype(str).isin(employee_ids)]
        if "employee_j" in df.columns:
            df = df[df["employee_j"].astype(str).isin(employee_ids)]
        if "day" in df.columns:
            df["day"] = pd.to_numeric(df["day"], errors="coerce").fillna(0).astype(int)
            df = df[df["day"].between(1, preset.days)]
        df.to_csv(output_dir / name, index=False, encoding="utf-8-sig")

    # Holiday calendar for full horizon.
    cal = []
    for d in range(1, preset.days + 1):
        weekday_index = (d - 1) % 7
        is_sat = weekday_index == 5
        is_sun = weekday_index == 6
        cal.append({
            "day": d,
            "weekday_index": weekday_index,
            "is_saturday": int(is_sat),
            "is_sunday": int(is_sun),
            "is_weekend": int(is_sat or is_sun),
            "is_public_holiday": 0,
        })
    pd.DataFrame(cal).to_csv(output_dir / "holiday_calendar.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create fixed MEDIUM MILP-SA input scenario from baseline.")
    p.add_argument("--baseline-input-dir", type=Path, default=DEFAULT_BASELINE_INPUT_DIR, help="Current SMALL employee_milp_inputs folder.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output folder for scaled inputs.")
    p.add_argument("--scale-level", choices=["medium"], default="medium", help=argparse.SUPPRESS)
    p.add_argument("--days", type=int, default=None, help="Override preset days.")
    p.add_argument("--demand-multiplier", type=float, default=None, help="Override preset demand multiplier.")
    p.add_argument("--employees", type=int, default=None, help="Override preset employee count.")
    p.add_argument("--min-work-days", type=int, default=None)
    p.add_argument("--max-work-days", type=int, default=None)
    p.add_argument("--target-work-days", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_preset = PRESETS[args.scale_level]
    preset = ScalePreset(
        name=base_preset.name,
        days=args.days or base_preset.days,
        demand_multiplier=args.demand_multiplier or base_preset.demand_multiplier,
        employees=args.employees or base_preset.employees,
        min_work_days=args.min_work_days or base_preset.min_work_days,
        max_work_days=args.max_work_days or base_preset.max_work_days,
        target_work_days=args.target_work_days or base_preset.target_work_days,
        expected_customers_per_day=base_preset.expected_customers_per_day,
    )

    input_dir = args.baseline_input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage = ensure_coverage_long(input_dir)
    scaled_long = scale_coverage(coverage, preset)
    scaled_wide = coverage_long_to_wide(scaled_long)
    demand_daily = coverage_opt_to_demand_daily(scaled_long)

    # Export coverage files used by MILP and diagnostics.
    scaled_long.drop(columns=[c for c in ["source_pattern_day", "demand_multiplier"] if c in scaled_long.columns]).to_csv(
        output_dir / "staffing_coverage_bands_long.csv", index=False, encoding="utf-8-sig"
    )
    scaled_long.to_csv(output_dir / "staffing_coverage_bands_long_scaled_trace.csv", index=False, encoding="utf-8-sig")
    scaled_wide.to_csv(output_dir / "staffing_coverage_bands.csv", index=False, encoding="utf-8-sig")
    demand_daily.to_csv(output_dir / "staffing_demand_daily.csv", index=False, encoding="utf-8-sig")

    # Employees and static files.
    employees, skills, standard = resize_employees(input_dir, output_dir, preset)
    write_shift_and_required_files(input_dir, output_dir, preset)
    write_optional_files(input_dir, output_dir, employees, preset)

    # Scenario summary.
    role_summary = scaled_long.groupby("role")[["C_min", "C_opt", "C_max"]].sum().reset_index()
    scenario_summary = pd.DataFrame([
        {"metric": "scale_level", "value": preset.name},
        {"metric": "days", "value": preset.days},
        {"metric": "demand_multiplier", "value": preset.demand_multiplier},
        {"metric": "expected_customers_per_day", "value": preset.expected_customers_per_day},
        {"metric": "employees", "value": len(employees)},
        {"metric": "roles", "value": len(ROLES)},
        {"metric": "shifts_per_day", "value": len(SHIFTS)},
        {"metric": "coverage_cells", "value": len(scaled_long)},
        {"metric": "total_C_min", "value": int(scaled_long["C_min"].sum())},
        {"metric": "total_C_opt", "value": int(scaled_long["C_opt"].sum())},
        {"metric": "total_C_max", "value": int(scaled_long["C_max"].sum())},
        {"metric": "min_work_days", "value": preset.min_work_days},
        {"metric": "max_work_days", "value": preset.max_work_days},
        {"metric": "target_work_days", "value": preset.target_work_days},
        {"metric": "approx_binary_variables", "value": len(employees) * preset.days * len(SHIFTS) * len(ROLES)},
    ])
    scenario_summary.to_csv(output_dir / "scale_scenario_summary.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(output_dir / f"scaled_inputs_{preset.name}.xlsx", engine="openpyxl") as writer:
        scenario_summary.to_excel(writer, sheet_name="scenario_summary", index=False)
        scaled_wide.to_excel(writer, sheet_name="coverage_wide", index=False)
        scaled_long.to_excel(writer, sheet_name="coverage_long", index=False)
        role_summary.to_excel(writer, sheet_name="role_summary", index=False)
        employees.to_excel(writer, sheet_name="employees", index=False)
        skills.to_excel(writer, sheet_name="skills", index=False)

    print("DONE: scaled input folder created")
    print(f"Scale level           : {preset.name}")
    print(f"Output dir            : {output_dir}")
    print(f"Days                  : {preset.days}")
    print(f"Employees             : {len(employees)}")
    print(f"Total C_opt           : {int(scaled_long['C_opt'].sum()):,}")
    print(f"Approx binary vars    : {len(employees) * preset.days * len(SHIFTS) * len(ROLES):,}")
    print("Next step: run MILP on this output folder, then run SA Phase 2.")


if __name__ == "__main__":
    main()
