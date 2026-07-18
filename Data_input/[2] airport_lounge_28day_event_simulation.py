"""
Airport Lounge 28-Day Event-Driven Operational Simulation
=========================================================

Purpose
-------
This script upgrades the staffing model from a static workload calculator into an
EVENT-DRIVEN operational simulation.

It uses arrival_baseline_48_slices.xlsx as the 1-day baseline arrival wave, then
creates a 28-day arrival plan with weekday/weekend/holiday multipliers. The first
simulated day can be treated as a warm-up day and excluded from KPI summaries.

Operational logic implemented
-----------------------------
Customer arrival -> admission capacity check -> RS reception service -> lounge stay
-> buffet visits / stock consumption -> dirty dishes -> customer departure -> DAS table reset

Other scheduled/background tasks:
- FSTC scheduled temperature checks
- SLS stock refill tasks when stock <= reorder point
- DS dishwashing batches when dirty dishes >= batch size

Main output
-----------
An Excel workbook with customer flow, service events, role workload summaries,
KPI validation, and staffing requirements by role and shift.

Dependencies
------------
pip install simpy pandas numpy openpyxl

Example run
-----------
python airport_lounge_28day_event_simulation.py \
    --input arrival_baseline_48_slices.xlsx \
    --output airport_lounge_28day_event_sim_output.xlsx \
    --days 28 \
    --warmup-days 1 \
    --start-date 2026-01-01
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import simpy
except ImportError as exc:
    raise ImportError(
        "SimPy is required. Install it with: pip install simpy pandas numpy openpyxl"
    ) from exc


# ============================================================
# Utility functions
# ============================================================


def tria_mean(a: float, m: float, b: float) -> float:
    return (a + m + b) / 3.0


def positive_normal(rng: np.random.Generator, mean: float, sd: float, min_value: float = 0.01) -> float:
    """Sample a positive normal value by resampling until positive."""
    for _ in range(100):
        x = rng.normal(mean, sd)
        if x > min_value:
            return float(x)
    return float(max(mean, min_value))


def minute_to_clock(minute: float) -> str:
    minute_int = int(round(minute)) % (24 * 60)
    h = minute_int // 60
    m = minute_int % 60
    return f"{h:02d}:{m:02d}"


def sim_day_from_time(t: float) -> int:
    """1-based day index from simulation minute."""
    return int(t // 1440) + 1


def minute_in_day(t: float) -> float:
    return t % 1440


def shift_id_from_time(t: float) -> int:
    """Shift 1: 00:00-08:00, Shift 2: 08:00-16:00, Shift 3: 16:00-24:00."""
    return int(minute_in_day(t) // 480) + 1


def shift_time_label(shift_id: int) -> str:
    return {1: "00:00-08:00", 2: "08:00-16:00", 3: "16:00-24:00"}.get(shift_id, "Unknown")


def percentile_95(x: pd.Series) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.percentile(x.dropna().values, 95)) if len(x.dropna()) else 0.0


def parse_holidays(raw: str) -> set:
    """
    Parse holidays passed as comma-separated values.
    Accepts either YYYY-MM-DD dates or 1-based day indexes, e.g.:
    --holidays 2026-01-01,2026-01-02
    --holidays 1,2,15
    """
    if not raw:
        return set()
    items = [x.strip() for x in raw.split(",") if x.strip()]
    parsed = set()
    for item in items:
        parsed.add(item)
    return parsed


# ============================================================
# Configuration
# ============================================================


@dataclass
class Config:
    input_file: str
    output_file: str
    days: int = 28
    warmup_days: int = 1
    start_date: str = "2026-01-01"
    holidays: str = ""
    base_seed: int = 42

    # Demand multipliers
    weekday_multiplier: float = 1.00
    weekend_multiplier: float = 1.15
    holiday_multiplier: float = 1.35
    daily_cv: float = 0.05

    # Staffing capacity defaults
    staff_rs: int = 2
    staff_das: int = 10
    staff_blo: int = 6
    staff_fstc: int = 2
    staff_ds: int = 2
    staff_sls: int = 2
    min_staff: int = 1

    # KPI and utilization
    target_utilization: float = 0.85
    effective_minutes_override: Optional[float] = None
    kpi_rs: float = 5.0
    kpi_das: float = 4.0
    kpi_blo: float = 6.0
    kpi_fstc: float = 5.0
    kpi_ds: float = 10.0
    kpi_sls: float = 5.0

    # Lounge capacity
    lounge_capacity: int = 200
    max_admission_wait: float = 30.0

    # Service distributions, minutes
    rs_tria: Tuple[float, float, float] = (0.15, 0.25, 0.40)
    blo_tria: Tuple[float, float, float] = (0.50, 0.75, 1.20)
    dwell_tria: Tuple[float, float, float] = (45.0, 90.0, 180.0)
    dwell_cap: float = 180.0
    das_effective_mean: float = 7.1
    das_use_effective_mean: bool = True
    das_tria: Tuple[float, float, float] = (2.0, 3.0, 4.0)
    sls_tria: Tuple[float, float, float] = (3.0, 4.5, 6.0)
    fstc_check_interval: float = 60.0
    fstc_tria: Tuple[float, float, float] = (2.0, 3.0, 5.0)

    # DAS formula assumption
    # TableTurns_s ≈ Departures_s × TableResetRate
    # W_DAS,s = TableTurns_s × 7.1
    table_reset_rate: float = 0.65

    # BLO formula assumption
    # BuffetVisits_s = Arrivals_s × VisitRate
    # BLOServiceEvents_s = BuffetVisits_s × InterventionRate
    # W_BLO,s = BLOServiceEvents_s × E[TRIA(0.50, 0.75, 1.20)]
    buffet_visit_rate: float = 1.5
    buffet_visit_min: int = 0
    blo_intervention_rate: float = 0.20

    # Dishes
    dirty_dishes_initial: int = 0
    dirty_dishes_lambda: float = 3.0
    dish_batch_size: int = 80
    ds_prewash_mean: float = 4.99
    ds_prewash_sd: float = 1.14
    ds_machine_cycle: float = 6.0
    ds_postwash_mean: float = 2.47
    ds_postwash_sd: float = 1.08

    # Stock / refill
    stock_max: float = 200.0
    stock_reorder: float = 80.0
    stock_refill_qty: float = 220.0
    stock_unit_per_visit: float = 1.0

    # Run-out time after last arrival day to allow late departures and tasks
    runout_hours: float = 4.0

    @property
    def effective_minutes(self) -> float:
        if self.effective_minutes_override is not None:
            return self.effective_minutes_override
        return 8.0 * 60.0 * self.target_utilization

    @property
    def staff_by_role(self) -> Dict[str, int]:
        return {
            "RS": self.staff_rs,
            "DAS": self.staff_das,
            "BLO": self.staff_blo,
            "FSTC": self.staff_fstc,
            "DS": self.staff_ds,
            "SLS": self.staff_sls,
        }

    @property
    def wait_kpi_by_role(self) -> Dict[str, float]:
        return {
            "RS": self.kpi_rs,
            "DAS": self.kpi_das,
            "BLO": self.kpi_blo,
            "FSTC": self.kpi_fstc,
            "DS": self.kpi_ds,
            "SLS": self.kpi_sls,
        }


# ============================================================
# Baseline loading and 28-day demand generation
# ============================================================


def load_baseline_48_slices(input_file: str) -> pd.DataFrame:
    """Load 48-slice baseline from arrival_baseline_48_slices.xlsx."""
    input_file = r"E:\NĂM 4\Capstone\Data_input\arrival_baseline_48_slices.xlsx"
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    xl = pd.ExcelFile(path)
    preferred_sheets = ["demand_shape_48", "generated_arrivals"]
    sheet = None
    for s in preferred_sheets:
        if s in xl.sheet_names:
            sheet = s
            break
    if sheet is None:
        raise ValueError(
            f"Could not find a baseline sheet. Expected one of {preferred_sheets}; found {xl.sheet_names}"
        )

    df = pd.read_excel(path, sheet_name=sheet)
    if "slice_id" not in df.columns:
        raise ValueError("Baseline sheet must contain 'slice_id'.")
    if "expected_arrivals_30min" not in df.columns:
        raise ValueError("Baseline sheet must contain 'expected_arrivals_30min'.")

    df = df.copy()
    df["slice_id"] = df["slice_id"].astype(int)
    df = df.sort_values("slice_id").reset_index(drop=True)

    # Reconstruct columns if missing.
    if "start_minute" not in df.columns:
        df["start_minute"] = (df["slice_id"] - 1) * 30
    if "end_minute" not in df.columns:
        df["end_minute"] = df["slice_id"] * 30
    if "start_time" not in df.columns:
        df["start_time"] = df["start_minute"].apply(minute_to_clock)
    if "end_time" not in df.columns:
        df["end_time"] = df["end_minute"].apply(minute_to_clock)
    if "shift_id" not in df.columns:
        df["shift_id"] = ((df["slice_id"] - 1) // 16) + 1
    if "slice_in_shift" not in df.columns:
        df["slice_in_shift"] = ((df["slice_id"] - 1) % 16) + 1
    if "demand_level" not in df.columns:
        df["demand_level"] = "Unknown"
    if "lambda_per_hour" not in df.columns:
        df["lambda_per_hour"] = df["expected_arrivals_30min"] / 0.5

    if len(df) != 48:
        raise ValueError(f"Expected 48 slices, but found {len(df)} rows.")

    return df


def create_day_wave_plan(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.base_seed + 10_000)
    start = datetime.strptime(cfg.start_date, "%Y-%m-%d").date()
    holiday_set = parse_holidays(cfg.holidays)
    rows = []
    for d in range(1, cfg.days + 1):
        current_date = start + timedelta(days=d - 1)
        weekday_name = current_date.strftime("%A")
        is_weekend = current_date.weekday() >= 5
        is_holiday = (current_date.isoformat() in holiday_set) or (str(d) in holiday_set)

        if is_holiday:
            day_type = "holiday"
            base_multiplier = cfg.holiday_multiplier
        elif is_weekend:
            day_type = "weekend"
            base_multiplier = cfg.weekend_multiplier
        else:
            day_type = "weekday"
            base_multiplier = cfg.weekday_multiplier

        # Random day-to-day noise, truncated to avoid unrealistic negative/too low demand.
        noise = float(np.clip(rng.normal(1.0, cfg.daily_cv), 0.70, 1.40))
        total_multiplier = base_multiplier * noise

        rows.append({
            "day_index": d,
            "date": current_date.isoformat(),
            "weekday": weekday_name,
            "day_type": day_type,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "base_multiplier": base_multiplier,
            "daily_random_multiplier": noise,
            "total_demand_multiplier": total_multiplier,
            "is_warmup_day": d <= cfg.warmup_days,
        })
    return pd.DataFrame(rows)


def generate_28day_arrivals(baseline: pd.DataFrame, day_plan: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.base_seed)
    slice_rows = []
    event_rows = []
    customer_id = 1

    for _, day_row in day_plan.iterrows():
        day = int(day_row["day_index"])
        day_offset = (day - 1) * 1440
        multiplier = float(day_row["total_demand_multiplier"])

        for _, srow in baseline.iterrows():
            expected = float(srow["expected_arrivals_30min"]) * multiplier
            actual = int(rng.poisson(expected))
            start_min = float(srow["start_minute"])
            end_min = float(srow["end_minute"])
            global_start = day_offset + start_min
            global_end = day_offset + end_min

            slice_rows.append({
                "day_index": day,
                "date": day_row["date"],
                "weekday": day_row["weekday"],
                "day_type": day_row["day_type"],
                "is_warmup_day": bool(day_row["is_warmup_day"]),
                "slice_id": int(srow["slice_id"]),
                "shift_id": int(srow["shift_id"]),
                "shift_time": shift_time_label(int(srow["shift_id"])),
                "start_time": srow["start_time"],
                "end_time": srow["end_time"],
                "global_start_minute": global_start,
                "global_end_minute": global_end,
                "demand_level": srow.get("demand_level", "Unknown"),
                "baseline_expected_arrivals_30min": float(srow["expected_arrivals_30min"]),
                "demand_multiplier": multiplier,
                "expected_arrivals_30min": expected,
                "lambda_per_hour": expected / 0.5,
                "actual_arrivals": actual,
            })

            if actual > 0:
                arrivals = np.sort(rng.uniform(global_start, global_end, size=actual))
                for at in arrivals:
                    event_rows.append({
                        "customer_id": customer_id,
                        "arrival_time_min": float(at),
                        "arrival_day_index": day,
                        "arrival_clock_time": minute_to_clock(at),
                        "date": day_row["date"],
                        "weekday": day_row["weekday"],
                        "day_type": day_row["day_type"],
                        "is_warmup_day": bool(day_row["is_warmup_day"]),
                        "slice_id": int(srow["slice_id"]),
                        "shift_id": int(srow["shift_id"]),
                        "shift_time": shift_time_label(int(srow["shift_id"])),
                        "demand_level": srow.get("demand_level", "Unknown"),
                    })
                    customer_id += 1

    slice_df = pd.DataFrame(slice_rows)
    events_df = pd.DataFrame(event_rows).sort_values("arrival_time_min").reset_index(drop=True)
    if not events_df.empty:
        events_df["customer_id"] = np.arange(1, len(events_df) + 1)
    return slice_df, events_df


# ============================================================
# Event-driven simulation model
# ============================================================


class LoungeSimulation:
    def __init__(self, cfg: Config, arrivals_df: pd.DataFrame):
        self.cfg = cfg
        self.arrivals_df = arrivals_df
        self.rng = np.random.default_rng(cfg.base_seed + 999)
        self.env = simpy.Environment()

        # Resources by operational role.
        self.resources = {
            "RS": simpy.Resource(self.env, capacity=cfg.staff_rs),
            "DAS": simpy.Resource(self.env, capacity=cfg.staff_das),
            "BLO": simpy.Resource(self.env, capacity=cfg.staff_blo),
            "FSTC": simpy.Resource(self.env, capacity=cfg.staff_fstc),
            "DS": simpy.Resource(self.env, capacity=cfg.staff_ds),
            "SLS": simpy.Resource(self.env, capacity=cfg.staff_sls),
        }

        # Lounge capacity is modeled as a resource slot held during the stay.
        self.lounge_space = simpy.Resource(self.env, capacity=cfg.lounge_capacity)

        # Stock is modeled as a container capped at stock_max.
        self.stock = simpy.Container(self.env, init=cfg.stock_max, capacity=cfg.stock_max)
        self.stock_lock = simpy.Resource(self.env, capacity=1)
        self.refill_in_progress = False

        # Dishes are accumulated and converted into DS batch tasks.
        self.dirty_dishes = int(cfg.dirty_dishes_initial)

        # Logs
        self.customer_log: List[dict] = []
        self.service_log: List[dict] = []
        self.stock_log: List[dict] = []
        self.dish_log: List[dict] = []

    # ---------- Sampling ----------

    def sample_tria(self, params: Tuple[float, float, float]) -> float:
        a, m, b = params
        return float(self.rng.triangular(a, m, b))

    def sample_rs_service(self) -> float:
        return self.sample_tria(self.cfg.rs_tria)

    def sample_blo_service(self) -> float:
        return self.sample_tria(self.cfg.blo_tria)

    def sample_dwell_time(self) -> float:
        dwell = self.sample_tria(self.cfg.dwell_tria)
        return float(min(self.cfg.dwell_cap, dwell))

    def sample_das_service(self) -> float:
        if self.cfg.das_use_effective_mean:
            return float(self.cfg.das_effective_mean)
        return self.sample_tria(self.cfg.das_tria)

    def sample_fstc_service(self) -> float:
        return self.sample_tria(self.cfg.fstc_tria)

    def sample_sls_service(self) -> float:
        return self.sample_tria(self.cfg.sls_tria)

    def sample_dirty_dishes(self) -> int:
        return int(max(1, self.rng.poisson(self.cfg.dirty_dishes_lambda)))

    def sample_ds_batch_service(self) -> float:
        pre = positive_normal(self.rng, self.cfg.ds_prewash_mean, self.cfg.ds_prewash_sd)
        post = positive_normal(self.rng, self.cfg.ds_postwash_mean, self.cfg.ds_postwash_sd)
        return float(pre + self.cfg.ds_machine_cycle + post)

    def sample_buffet_visits(self) -> int:
        """
        BuffetVisits_i follows the report formula logic:
            BuffetVisits_s = A_s × VisitRate

        At customer level, this is simulated as a Poisson number of buffet trips.
        Default buffet_visit_min = 0 so the expected number of trips remains close
        to VisitRate = 1.5. If you want every customer to have at least one buffet
        trip, run with --buffet-visit-min 1.
        """
        return int(max(self.cfg.buffet_visit_min, self.rng.poisson(self.cfg.buffet_visit_rate)))

    # ---------- Service / resource helper ----------

    def service_task(self, role: str, task_type: str, service_time: float, customer_id: Optional[int] = None):
        request_time = self.env.now
        resource = self.resources[role]
        with resource.request() as req:
            yield req
            start_time = self.env.now
            wait_time = start_time - request_time
            yield self.env.timeout(service_time)
            end_time = self.env.now

        self.service_log.append({
            "role": role,
            "task_type": task_type,
            "customer_id": customer_id,
            "request_time_min": request_time,
            "service_start_min": start_time,
            "service_end_min": end_time,
            "wait_time_min": wait_time,
            "service_time_min": service_time,
            "request_day": sim_day_from_time(request_time),
            "service_day": sim_day_from_time(start_time),
            "service_clock_time": minute_to_clock(start_time),
            "shift_id": shift_id_from_time(start_time),
            "shift_time": shift_time_label(shift_id_from_time(start_time)),
        })
        return wait_time, service_time

    # ---------- Stock / SLS ----------

    def consume_stock(self, customer_id: int, units: float):
        with self.stock_lock.request() as lock_req:
            yield lock_req
            before = float(self.stock.level)
            consume_qty = min(before, units)
            stockout_qty = max(0.0, units - before)
            if consume_qty > 0:
                yield self.stock.get(consume_qty)
            after = float(self.stock.level)

            self.stock_log.append({
                "time_min": self.env.now,
                "day_index": sim_day_from_time(self.env.now),
                "clock_time": minute_to_clock(self.env.now),
                "event": "consume",
                "customer_id": customer_id,
                "units_requested": units,
                "units_consumed": consume_qty,
                "stockout_units": stockout_qty,
                "stock_before": before,
                "stock_after": after,
            })

            if after <= self.cfg.stock_reorder and not self.refill_in_progress:
                self.refill_in_progress = True
                self.env.process(self.refill_stock())

    def refill_stock(self):
        before_request = float(self.stock.level)
        service_time = self.sample_sls_service()
        yield self.env.process(self.service_task("SLS", "stock_refill", service_time, customer_id=None))

        with self.stock_lock.request() as lock_req:
            yield lock_req
            before = float(self.stock.level)
            qty_to_add = min(self.cfg.stock_refill_qty, self.cfg.stock_max - before)
            if qty_to_add > 0:
                yield self.stock.put(qty_to_add)
            after = float(self.stock.level)
            self.refill_in_progress = False

            self.stock_log.append({
                "time_min": self.env.now,
                "day_index": sim_day_from_time(self.env.now),
                "clock_time": minute_to_clock(self.env.now),
                "event": "refill",
                "customer_id": None,
                "units_requested": self.cfg.stock_refill_qty,
                "units_consumed": None,
                "stockout_units": None,
                "stock_before": before,
                "stock_after": after,
                "stock_before_request": before_request,
                "refill_added": qty_to_add,
            })

            # If demand consumed again while refill was happening, trigger another refill if needed.
            if after <= self.cfg.stock_reorder and not self.refill_in_progress:
                self.refill_in_progress = True
                self.env.process(self.refill_stock())

    # ---------- Dirty dishes / DS ----------

    def add_dirty_dishes_and_maybe_start_batches(self, customer_id: int, dishes: int):
        self.dirty_dishes += dishes
        self.dish_log.append({
            "time_min": self.env.now,
            "day_index": sim_day_from_time(self.env.now),
            "clock_time": minute_to_clock(self.env.now),
            "event": "dirty_dishes_added",
            "customer_id": customer_id,
            "dishes_added": dishes,
            "dirty_dishes_after": self.dirty_dishes,
        })

        while self.dirty_dishes >= self.cfg.dish_batch_size:
            self.dirty_dishes -= self.cfg.dish_batch_size
            self.dish_log.append({
                "time_min": self.env.now,
                "day_index": sim_day_from_time(self.env.now),
                "clock_time": minute_to_clock(self.env.now),
                "event": "dish_batch_triggered",
                "customer_id": None,
                "dishes_added": 0,
                "dirty_dishes_after": self.dirty_dishes,
                "batch_size": self.cfg.dish_batch_size,
            })
            self.env.process(self.dishwash_batch())

    def dishwash_batch(self):
        service_time = self.sample_ds_batch_service()
        yield self.env.process(self.service_task("DS", "dishwashing_batch", service_time, customer_id=None))

    # ---------- Customer lifecycle ----------

    def customer_process(self, event: dict):
        customer_id = int(event["customer_id"])
        arrival_time = float(event["arrival_time_min"])

        # Wait until the arrival time.
        yield self.env.timeout(max(0.0, arrival_time - self.env.now))

        log = {
            "customer_id": customer_id,
            "arrival_time_min": self.env.now,
            "arrival_day_index": int(event["arrival_day_index"]),
            "arrival_clock_time": minute_to_clock(self.env.now),
            "arrival_shift_id": int(event["shift_id"]),
            "arrival_shift_time": event["shift_time"],
            "arrival_day_type": event["day_type"],
            "admitted": False,
            "rejected": False,
            "admission_wait_min": None,
            "rs_wait_min": None,
            "rs_service_min": None,
            "dwell_time_min": None,
            "departure_time_min": None,
            "departure_day_index": None,
            "departure_clock_time": None,
            "buffet_visits": 0,
            "table_reset_required": False,
            "dirty_dishes": 0,
        }

        # Lounge admission capacity. If full, wait up to max_admission_wait.
        admission_request_time = self.env.now
        space_req = self.lounge_space.request()
        result = yield space_req | self.env.timeout(self.cfg.max_admission_wait)
        if space_req not in result:
            # Customer gives up because lounge capacity wait exceeded threshold.
            space_req.cancel()
            log["rejected"] = True
            log["admission_wait_min"] = self.cfg.max_admission_wait
            self.customer_log.append(log)
            return

        log["admitted"] = True
        log["admission_wait_min"] = self.env.now - admission_request_time

        # Reception service after admission.
        rs_service_time = self.sample_rs_service()
        rs_wait, rs_service = yield self.env.process(
            self.service_task("RS", "reception_checkin", rs_service_time, customer_id=customer_id)
        )
        log["rs_wait_min"] = rs_wait
        log["rs_service_min"] = rs_service

        # Customer lounge stay.
        enter_time = self.env.now
        dwell = self.sample_dwell_time()
        log["dwell_time_min"] = dwell

        # Buffet visit process happens during the dwell period.
        visits = self.sample_buffet_visits()
        log["buffet_visits"] = visits
        self.env.process(self.buffet_visit_process(customer_id, enter_time, dwell, visits))

        # Customer stays, then leaves.
        yield self.env.timeout(dwell)
        departure_time = self.env.now
        log["departure_time_min"] = departure_time
        log["departure_day_index"] = sim_day_from_time(departure_time)
        log["departure_clock_time"] = minute_to_clock(departure_time)

        # Release lounge space.
        self.lounge_space.release(space_req)

        # Departures trigger dirty dishes.
        dishes = self.sample_dirty_dishes()
        log["dirty_dishes"] = dishes
        self.add_dirty_dishes_and_maybe_start_batches(customer_id, dishes)

        # DAS revised formula:
        #     TableTurns_s ≈ Departures_s × TableResetRate
        #     W_DAS,s = TableTurns_s × T_DAS
        # In the event simulation, each departure becomes a DAS table-reset task
        # only with probability TableResetRate = 0.65. This represents mixed
        # individual/shared seating and avoids assuming every customer departure
        # creates one full table reset.
        table_reset_required = self.rng.random() < self.cfg.table_reset_rate
        log["table_reset_required"] = bool(table_reset_required)
        if table_reset_required:
            das_service_time = self.sample_das_service()
            self.env.process(self.service_task("DAS", "table_clear_reset", das_service_time, customer_id=customer_id))

        self.customer_log.append(log)

    def buffet_visit_process(self, customer_id: int, enter_time: float, dwell: float, visits: int):
        if visits <= 0:
            return
        visit_offsets = np.sort(self.rng.uniform(0.0, max(0.01, dwell), size=visits))
        for idx, offset in enumerate(visit_offsets, start=1):
            target_time = enter_time + float(offset)
            yield self.env.timeout(max(0.0, target_time - self.env.now))
            # BLO revised formula:
            #     BuffetVisits_s = A_s × VisitRate
            #     BLOServiceEvents_s = BuffetVisits_s × InterventionRate
            # Because the buffet is mostly self-service, every buffet trip consumes
            # stock, but only a proportion requires BLO staff intervention.
            if self.rng.random() < self.cfg.blo_intervention_rate:
                service_time = self.sample_blo_service()
                yield self.env.process(self.service_task("BLO", "buffet_service_intervention", service_time, customer_id=customer_id))

            yield self.env.process(self.consume_stock(customer_id, self.cfg.stock_unit_per_visit))

    # ---------- Scheduled FSTC process ----------

    def fstc_process(self, until_minute: float):
        t = 0.0
        while t < until_minute:
            yield self.env.timeout(max(0.0, t - self.env.now))
            service_time = self.sample_fstc_service()
            self.env.process(self.service_task("FSTC", "temperature_check", service_time, customer_id=None))
            t += self.cfg.fstc_check_interval

    # ---------- Run ----------

    def run(self) -> Dict[str, pd.DataFrame]:
        horizon = self.cfg.days * 1440
        runout = self.cfg.runout_hours * 60
        until = horizon + runout

        # Start scheduled FSTC checks.
        self.env.process(self.fstc_process(until))

        # Start all customer processes.
        for _, row in self.arrivals_df.iterrows():
            self.env.process(self.customer_process(row.to_dict()))

        self.env.run(until=until)

        return {
            "customer_log": pd.DataFrame(self.customer_log),
            "service_log": pd.DataFrame(self.service_log),
            "stock_log": pd.DataFrame(self.stock_log),
            "dish_log": pd.DataFrame(self.dish_log),
        }


# ============================================================
# Output summaries
# ============================================================


def add_measurement_flags(df: pd.DataFrame, cfg: Config, day_col: str) -> pd.DataFrame:
    if df.empty or day_col not in df.columns:
        return df
    out = df.copy()
    out["is_warmup_day"] = out[day_col] <= cfg.warmup_days
    out["is_planning_horizon_day"] = out[day_col] <= cfg.days
    out["is_measured_day"] = (out[day_col] > cfg.warmup_days) & (out[day_col] <= cfg.days)
    return out


def create_formula_assumptions(cfg: Config) -> pd.DataFrame:
    """Create a report sheet documenting the revised DAS and BLO formulas."""
    t_blo = tria_mean(*cfg.blo_tria)
    rows = [
        {
            "role": "DAS",
            "driver": "Customer departures converted into table clear/reset tasks",
            "formula_step_1": "DepartureTime_i = ArrivalTime_i + DwellTime_i",
            "formula_step_2": "DwellTime_i = min(180, TRIA(45, 90, 180)); E[DwellTime] = 105 min",
            "formula_step_3": f"TableTurns_s ≈ Departures_s × TableResetRate; TableResetRate = {cfg.table_reset_rate}",
            "workload_formula": f"W_DAS,s = TableTurns_s × T_DAS; T_DAS = {cfg.das_effective_mean} min/task",
            "staff_formula": "Staff_DAS,s = max(1, ceil(W_DAS,s / E))",
            "code_implementation": "At each customer departure, a DAS table_clear_reset event is generated with probability table_reset_rate.",
        },
        {
            "role": "BLO",
            "driver": "Buffet trips that require staff intervention",
            "formula_step_1": f"BuffetVisits_s = Arrivals_s × VisitRate; VisitRate = {cfg.buffet_visit_rate} trips/customer",
            "formula_step_2": f"BLOServiceEvents_s = BuffetVisits_s × InterventionRate; InterventionRate = {cfg.blo_intervention_rate}",
            "formula_step_3": f"T_BLO = E[TRIA({cfg.blo_tria[0]}, {cfg.blo_tria[1]}, {cfg.blo_tria[2]})] = {t_blo:.3f} min/intervention",
            "workload_formula": f"W_BLO,s = Arrivals_s × {cfg.buffet_visit_rate} × {cfg.blo_intervention_rate} × {t_blo:.3f}",
            "staff_formula": "Staff_BLO,s = max(1, ceil(W_BLO,s / E))",
            "code_implementation": "Every buffet trip consumes stock, but only blo_intervention_rate of trips generates a BLO service task.",
        },
    ]
    return pd.DataFrame(rows)


def build_summaries(
    cfg: Config,
    day_plan: pd.DataFrame,
    slice_plan: pd.DataFrame,
    arrival_events: pd.DataFrame,
    logs: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    customer = logs["customer_log"].copy()
    service = logs["service_log"].copy()
    stock = logs["stock_log"].copy()
    dishes = logs["dish_log"].copy()

    customer = add_measurement_flags(customer, cfg, "arrival_day_index")
    service = add_measurement_flags(service, cfg, "service_day")
    stock = add_measurement_flags(stock, cfg, "day_index")
    dishes = add_measurement_flags(dishes, cfg, "day_index")

    # KPI fields for service events.
    if not service.empty:
        service["wait_kpi_min"] = service["role"].map(cfg.wait_kpi_by_role)
        service["wait_kpi_pass"] = service["wait_time_min"] <= service["wait_kpi_min"]
        service["staff_config"] = service["role"].map(cfg.staff_by_role)

    # Customer daily summary.
    if not customer.empty:
        customer_daily = (
            customer.groupby("arrival_day_index")
            .agg(
                total_arrivals=("customer_id", "count"),
                admitted_customers=("admitted", "sum"),
                rejected_customers=("rejected", "sum"),
                avg_admission_wait_min=("admission_wait_min", "mean"),
                p95_admission_wait_min=("admission_wait_min", percentile_95),
                avg_rs_wait_min=("rs_wait_min", "mean"),
                avg_dwell_time_min=("dwell_time_min", "mean"),
                total_buffet_visits=("buffet_visits", "sum"),
                total_table_resets=("table_reset_required", "sum"),
                total_dirty_dishes=("dirty_dishes", "sum"),
            )
            .reset_index()
            .rename(columns={"arrival_day_index": "day_index"})
        )
        customer_daily = customer_daily.merge(day_plan, on="day_index", how="left")
    else:
        customer_daily = pd.DataFrame()

    # Role/day/shift workload summary.
    if not service.empty:
        role_daily_shift = (
            service.groupby(["service_day", "shift_id", "shift_time", "role"])
            .agg(
                task_count=("role", "count"),
                workload_minutes=("service_time_min", "sum"),
                avg_wait_min=("wait_time_min", "mean"),
                p95_wait_min=("wait_time_min", percentile_95),
                max_wait_min=("wait_time_min", "max"),
                kpi_pass_rate=("wait_kpi_pass", "mean"),
            )
            .reset_index()
            .rename(columns={"service_day": "day_index"})
        )
        role_daily_shift["staff_config"] = role_daily_shift["role"].map(cfg.staff_by_role)
        role_daily_shift["available_minutes"] = role_daily_shift["staff_config"] * 480.0
        role_daily_shift["utilization"] = role_daily_shift["workload_minutes"] / role_daily_shift["available_minutes"]
        role_daily_shift["required_staff_by_workload"] = np.ceil(
            role_daily_shift["workload_minutes"] / cfg.effective_minutes
        ).astype(int)
        role_daily_shift["required_staff_by_workload"] = role_daily_shift["required_staff_by_workload"].clip(lower=cfg.min_staff)
        role_daily_shift["wait_kpi_min"] = role_daily_shift["role"].map(cfg.wait_kpi_by_role)
        role_daily_shift["kpi_status"] = np.where(
            role_daily_shift["p95_wait_min"] <= role_daily_shift["wait_kpi_min"], "PASS", "FAIL"
        )
        role_daily_shift["utilization_status"] = np.where(
            role_daily_shift["utilization"] <= cfg.target_utilization, "PASS", "FAIL"
        )
        role_daily_shift = add_measurement_flags(role_daily_shift, cfg, "day_index")
    else:
        role_daily_shift = pd.DataFrame()

    # Aggregate recommendation by role and shift across measured days.
    measured = role_daily_shift[role_daily_shift.get("is_measured_day", False) == True].copy() if not role_daily_shift.empty else pd.DataFrame()
    if not measured.empty:
        role_shift_recommendation = (
            measured.groupby(["shift_id", "shift_time", "role"])
            .agg(
                measured_days=("day_index", "nunique"),
                avg_task_count=("task_count", "mean"),
                avg_workload_minutes=("workload_minutes", "mean"),
                p95_workload_minutes=("workload_minutes", percentile_95),
                avg_wait_min=("avg_wait_min", "mean"),
                p95_wait_min=("p95_wait_min", "mean"),
                worst_wait_min=("max_wait_min", "max"),
                avg_utilization=("utilization", "mean"),
                max_utilization=("utilization", "max"),
                kpi_pass_rate=("kpi_pass_rate", "mean"),
                staff_config=("staff_config", "first"),
            )
            .reset_index()
        )
        role_shift_recommendation["wait_kpi_min"] = role_shift_recommendation["role"].map(cfg.wait_kpi_by_role)
        role_shift_recommendation["staff_required_avg_workload"] = np.ceil(
            role_shift_recommendation["avg_workload_minutes"] / cfg.effective_minutes
        ).astype(int).clip(cfg.min_staff)
        role_shift_recommendation["staff_required_p95_workload"] = np.ceil(
            role_shift_recommendation["p95_workload_minutes"] / cfg.effective_minutes
        ).astype(int).clip(cfg.min_staff)
        role_shift_recommendation["kpi_status"] = np.where(
            role_shift_recommendation["p95_wait_min"] <= role_shift_recommendation["wait_kpi_min"], "PASS", "FAIL"
        )
        role_shift_recommendation["utilization_status"] = np.where(
            role_shift_recommendation["avg_utilization"] <= cfg.target_utilization, "PASS", "FAIL"
        )
        role_shift_recommendation["recommended_staff_initial"] = role_shift_recommendation[
            "staff_required_p95_workload"
        ].clip(lower=cfg.min_staff)
        role_shift_recommendation["recommended_action"] = np.where(
            (role_shift_recommendation["kpi_status"] == "PASS")
            & (role_shift_recommendation["utilization_status"] == "PASS"),
            "Accept workload-based staff; rerun if staffing is changed",
            "Increase staff for this role/shift and rerun simulation",
        )
    else:
        role_shift_recommendation = pd.DataFrame()

    # Pivot staffing view.
    if not role_shift_recommendation.empty:
        staffing_pivot = role_shift_recommendation.pivot_table(
            index=["shift_id", "shift_time"],
            columns="role",
            values="recommended_staff_initial",
            aggfunc="max",
            fill_value=cfg.min_staff,
        ).reset_index()
        role_cols = [c for c in ["RS", "DAS", "BLO", "FSTC", "DS", "SLS"] if c in staffing_pivot.columns]
        staffing_pivot["total_staff"] = staffing_pivot[role_cols].sum(axis=1)
    else:
        staffing_pivot = pd.DataFrame()

    # Run parameters sheet.
    parameter_rows = []
    for k, v in vars(cfg).items():
        parameter_rows.append({"parameter": k, "value": str(v)})
    parameter_rows.append({"parameter": "effective_minutes_per_staff_shift", "value": str(cfg.effective_minutes)})
    run_parameters = pd.DataFrame(parameter_rows)

    return {
        "run_parameters": run_parameters,
        "formula_assumptions": create_formula_assumptions(cfg),
        "daily_wave_plan": day_plan,
        "arrivals_28d_slices": slice_plan,
        "arrival_events_28d": arrival_events,
        "customer_log": customer,
        "customer_daily_summary": customer_daily,
        "service_events": service,
        "role_daily_shift_workload": role_daily_shift,
        "role_shift_recommendation": role_shift_recommendation,
        "staffing_pivot": staffing_pivot,
        "stock_events": stock,
        "dish_events": dishes,
    }


def export_outputs(outputs: Dict[str, pd.DataFrame], output_file: str) -> Path:
    """
    Export outputs.

    To keep the 28-day run fast and Excel-readable, detailed event logs are saved
    as CSV files next to the workbook, while the Excel workbook contains summary
    sheets and pointers to the detailed CSV files.
    """
    path = Path(output_file)
    output_dir = path.parent
    stem = path.stem

    detailed_sheets = {
        "arrival_events_28d",
        "customer_log",
        "service_events",
        "stock_events",
        "dish_events",
    }

    excel_outputs: Dict[str, pd.DataFrame] = {}
    detail_index_rows = []

    for sheet_name, df in outputs.items():
        if sheet_name in detailed_sheets and df is not None and not df.empty:
            csv_name = f"{stem}_{sheet_name}.csv"
            csv_path = output_dir / csv_name
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            detail_index_rows.append({
                "detail_dataset": sheet_name,
                "rows": len(df),
                "file_name": csv_name,
                "file_path": str(csv_path.resolve()),
                "note": "Saved as CSV to keep Excel output fast and readable."
            })
        else:
            excel_outputs[sheet_name] = df

    excel_outputs["detail_csv_index"] = pd.DataFrame(detail_index_rows)

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#D9EAF7",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        })
        number_fmt = workbook.add_format({"num_format": "0.00"})

        for sheet_name, df in excel_outputs.items():
            safe_name = sheet_name[:31]
            if df is None or df.empty:
                df_to_write = pd.DataFrame({"message": ["No records generated"]})
            else:
                df_to_write = df

            df_to_write.to_excel(writer, sheet_name=safe_name, index=False)
            worksheet = writer.sheets[safe_name]
            worksheet.freeze_panes(1, 0)

            for col_idx, col_name in enumerate(df_to_write.columns):
                worksheet.write(0, col_idx, col_name, header_fmt)

            sample = df_to_write.head(500)
            for col_idx, col_name in enumerate(df_to_write.columns):
                sample_width = sample[col_name].astype(str).map(len).max() if len(sample) else 0
                header_width = len(str(col_name))
                width = min(max(int(max(sample_width, header_width)) + 2, 10), 35)
                fmt = number_fmt if any(key in str(col_name).lower() for key in ["wait", "time", "minute", "utilization", "workload", "multiplier"]) else None
                worksheet.set_column(col_idx, col_idx, width, fmt)

    return path.resolve()


# ============================================================
# Main
# ============================================================


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Airport lounge 28-day event-driven staffing simulation")
    parser.add_argument("--input", default="arrival_baseline_48_slices.xlsx")
    parser.add_argument("--output", default=r"E:\NĂM 4\Capstone\Data_input\airport_lounge_28day_event_sim_output.xlsx")
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--warmup-days", type=int, default=1)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--holidays", default="", help="Comma-separated dates YYYY-MM-DD or day indexes, e.g. 1,2,2026-01-15")
    parser.add_argument("--base-seed", type=int, default=42)

    # Demand
    parser.add_argument("--weekday-multiplier", type=float, default=1.00)
    parser.add_argument("--weekend-multiplier", type=float, default=1.15)
    parser.add_argument("--holiday-multiplier", type=float, default=1.35)
    parser.add_argument("--daily-cv", type=float, default=0.05)

    # Staffing
    parser.add_argument("--staff-rs", type=int, default=2)
    parser.add_argument("--staff-das", type=int, default=10)
    parser.add_argument("--staff-blo", type=int, default=6)
    parser.add_argument("--staff-fstc", type=int, default=2)
    parser.add_argument("--staff-ds", type=int, default=2)
    parser.add_argument("--staff-sls", type=int, default=2)
    parser.add_argument("--min-staff", type=int, default=1)
    parser.add_argument("--target-utilization", type=float, default=0.85)
    parser.add_argument("--effective-minutes", type=float, default=None, dest="effective_minutes_override")

    # KPI
    parser.add_argument("--kpi-rs", type=float, default=5.0)
    parser.add_argument("--kpi-das", type=float, default=4.0)
    parser.add_argument("--kpi-blo", type=float, default=6.0)
    parser.add_argument("--kpi-fstc", type=float, default=5.0)
    parser.add_argument("--kpi-ds", type=float, default=10.0)
    parser.add_argument("--kpi-sls", type=float, default=5.0)

    # Capacity
    parser.add_argument("--lounge-capacity", type=int, default=200)
    parser.add_argument("--max-admission-wait", type=float, default=30.0)

    # Behavior/service assumptions
    parser.add_argument("--fstc-check-interval", type=float, default=60.0)
    parser.add_argument("--das-effective-mean", type=float, default=7.1)
    parser.add_argument("--use-das-tria", action="store_true", help="Use TRIA(2,3,4) instead of effective mean 7.1 for DAS")
    parser.add_argument("--table-reset-rate", type=float, default=0.65, help="DAS table reset rate: TableTurns = Departures × rate")
    parser.add_argument("--buffet-visit-rate", type=float, default=1.5, help="Average buffet trips per customer")
    parser.add_argument("--buffet-visit-min", type=int, default=0, help="Minimum buffet trips per customer; 0 keeps expected visits close to VisitRate")
    parser.add_argument("--blo-intervention-rate", type=float, default=0.20, help="Share of buffet trips that require BLO staff intervention")
    parser.add_argument("--stock-refill-qty", type=float, default=220.0)
    parser.add_argument("--stock-unit-per-visit", type=float, default=1.0)
    parser.add_argument("--runout-hours", type=float, default=4.0)

    args = parser.parse_args()

    cfg = Config(
        input_file=args.input,
        output_file=args.output,
        days=args.days,
        warmup_days=args.warmup_days,
        start_date=args.start_date,
        holidays=args.holidays,
        base_seed=args.base_seed,
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
        effective_minutes_override=args.effective_minutes_override,
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
    return cfg


def main():
    cfg = parse_args()

    print("[1/6] Loading 48-slice baseline...")
    baseline = load_baseline_48_slices(cfg.input_file)

    print("[2/6] Creating 28-day day-type wave plan...")
    day_plan = create_day_wave_plan(cfg)

    print("[3/6] Generating 28-day arrival slices and customer events...")
    slice_plan, arrival_events = generate_28day_arrivals(baseline, day_plan, cfg)
    print(f"      Generated customer arrivals: {len(arrival_events):,}")

    print("[4/6] Running event-driven operational simulation...")
    model = LoungeSimulation(cfg, arrival_events)
    logs = model.run()

    print("[5/6] Building summaries and staffing recommendations...")
    outputs = build_summaries(cfg, day_plan, slice_plan, arrival_events, logs)

    print("[6/6] Exporting Excel output...")
    output_path = export_outputs(outputs, cfg.output_file)

    print("\nDONE")
    print(f"Output file: {output_path}")
    print(f"Days simulated: {cfg.days}")
    print(f"Warm-up days excluded from recommendation summaries: {cfg.warmup_days}")
    print(f"Effective minutes per staff per shift: {cfg.effective_minutes:.2f}")

    # Compact console summary.
    customer_daily = outputs.get("customer_daily_summary", pd.DataFrame())
    staff_pivot = outputs.get("staffing_pivot", pd.DataFrame())
    if not customer_daily.empty:
        measured_daily = customer_daily[customer_daily["is_warmup_day"] == False]
        print("\nCustomer summary after warm-up:")
        print(measured_daily[["day_index", "date", "day_type", "total_arrivals", "admitted_customers", "rejected_customers"]].head(10).to_string(index=False))
    if not staff_pivot.empty:
        print("\nInitial recommended staff by shift from p95 workload:")
        print(staff_pivot.to_string(index=False))


if __name__ == "__main__":
    main()
