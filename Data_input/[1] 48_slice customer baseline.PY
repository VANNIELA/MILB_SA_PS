# generate_48_slice_customer_baseline.py

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
OPERATING_HOURS = 24
SLICE_MINUTES = 30
SLICE_HOURS = SLICE_MINUTES / 60

N_SLICES = int(OPERATING_HOURS / SLICE_HOURS)

# Tổng expected lounge customers trong 1 ngày.
# Bạn có thể chỉnh số này theo scenario:
# low day = 600, base day = 900, high day = 1200, etc.
TOTAL_DAILY_EXPECTED_CUSTOMERS = 900

OUTPUT_FILE = "arrival_baseline_48_slices.xlsx"

# AIRPORT DEMAND SHAPE
# Based on Postorino et al. logic:
# airport passenger arrivals vary by time of day / flight-related behavior

def cyclic_hour_distance(hour, center):
    """
    Tính khoảng cách giờ theo vòng 24h.
    Ví dụ: distance giữa 23:30 và 00:30 là 1 giờ, không phải 23 giờ.
    """
    diff = abs(hour - center)
    return min(diff, 24 - diff)


def gaussian_peak(hour, center, width, amplitude):
    """
    Tạo một peak demand dạng Gaussian.
    center: giờ cao điểm, ví dụ 7.5 = 07:30
    width: độ rộng peak
    amplitude: độ cao peak
    """
    d = cyclic_hour_distance(hour, center)
    return amplitude * np.exp(-(d ** 2) / (2 * width ** 2))


def build_airport_demand_weight(mid_hour):
    """
    Tạo demand weight theo logic airport passenger flow.
    Đây không phải dữ liệu thật từ Postorino et al.
    Đây là scenario-based demand shape lấy cảm hứng từ paper:
    passenger arrival ở airport thay đổi theo time-of-day và flight wave.
    """

    base_floor = 0.15

    # Các cụm demand wave giả định trong ngày.
    # Bạn có thể chỉnh center/width/amplitude theo airport context hoặc flight schedule.
    early_morning_peak = gaussian_peak(
        hour=mid_hour,
        center=7.5,
        width=1.3,
        amplitude=1.00
    )

    mid_day_peak = gaussian_peak(
        hour=mid_hour,
        center=12.0,
        width=1.6,
        amplitude=0.55
    )

    evening_peak = gaussian_peak(
        hour=mid_hour,
        center=18.5,
        width=1.8,
        amplitude=0.90
    )

    late_night_peak = gaussian_peak(
        hour=mid_hour,
        center=22.5,
        width=1.2,
        amplitude=0.40
    )

    demand_weight = (
        base_floor
        + early_morning_peak
        + mid_day_peak
        + evening_peak
        + late_night_peak
    )

    return demand_weight

def minute_to_clock(minute):
    """
    Convert minute in day to HH:MM.
    """
    minute = int(round(minute)) % (24 * 60)
    h = minute // 60
    m = minute % 60
    return f"{h:02d}:{m:02d}"


def classify_demand_level(normalized_weight):
    """
    Gán nhãn Low / Medium / High / Peak để dễ đọc report.
    """
    if normalized_weight >= 0.030:
        return "Peak"
    elif normalized_weight >= 0.023:
        return "High"
    elif normalized_weight >= 0.017:
        return "Medium"
    else:
        return "Low"


def create_48_slice_baseline():
    rows = []

    for slice_id in range(1, N_SLICES + 1):
        start_min = (slice_id - 1) * SLICE_MINUTES
        end_min = slice_id * SLICE_MINUTES
        mid_min = (start_min + end_min) / 2
        mid_hour = mid_min / 60

        # 1 shift = 8 hours = 16 slices
        shift_id = ((slice_id - 1) // 16) + 1
        slice_in_shift = ((slice_id - 1) % 16) + 1

        raw_weight = build_airport_demand_weight(mid_hour)

        rows.append({
            "slice_id": slice_id,
            "start_time": minute_to_clock(start_min),
            "end_time": minute_to_clock(end_min),
            "start_minute": start_min,
            "end_minute": end_min,
            "mid_hour": round(mid_hour, 2),
            "shift_id": shift_id,
            "slice_in_shift": slice_in_shift,
            "raw_airport_demand_weight": raw_weight
        })

    df = pd.DataFrame(rows)

    # Normalize weight để tổng expected arrivals = TOTAL_DAILY_EXPECTED_CUSTOMERS
    total_weight = df["raw_airport_demand_weight"].sum()

    df["normalized_demand_weight"] = (
        df["raw_airport_demand_weight"] / total_weight
    )

    df["expected_arrivals_30min"] = (
        df["normalized_demand_weight"] * TOTAL_DAILY_EXPECTED_CUSTOMERS
    )

    # Kim & Whitt logic:
    # piecewise NHPP: trong mỗi lát 30 phút, arrival rate gần như constant
    # lambda_i = expected arrivals in slice / 0.5 hour
    df["lambda_per_hour"] = df["expected_arrivals_30min"] / SLICE_HOURS

    df["demand_level"] = df["normalized_demand_weight"].apply(
        classify_demand_level
    )

    return df

# Based on Kim & Whitt:
# N_i ~ Poisson(lambda_i * delta_t)
# Since expected_arrivals_30min = lambda_i * 0.5,
# N_i ~ Poisson(expected_arrivals_30min)

def generate_actual_arrivals(slice_df, seed=42):
    rng = np.random.default_rng(seed)

    generated_df = slice_df.copy()

    generated_df["actual_arrivals"] = rng.poisson(
        lam=generated_df["expected_arrivals_30min"].values
    )

    return generated_df

def generate_event_level_arrivals(generated_df, seed=42):
    """
    Với piecewise-constant NHPP:
    - Số khách trong mỗi slice được generate bằng Poisson.
    - Nếu biết có N khách trong slice, arrival time của họ được rải đều ngẫu nhiên
      trong interval đó.
    """

    rng = np.random.default_rng(seed + 100)
    event_rows = []
    customer_id = 1

    for _, row in generated_df.iterrows():
        n = int(row["actual_arrivals"])

        if n == 0:
            continue

        start_min = row["start_minute"]
        end_min = row["end_minute"]

        # Generate arrival time inside the 30-minute slice
        arrival_times = rng.uniform(
            low=start_min,
            high=end_min,
            size=n
        )

        arrival_times = np.sort(arrival_times)

        for arrival_min in arrival_times:
            event_rows.append({
                "customer_id": customer_id,
                "arrival_time_min": round(float(arrival_min), 3),
                "arrival_clock_time": minute_to_clock(arrival_min),
                "slice_id": int(row["slice_id"]),
                "shift_id": int(row["shift_id"]),
                "demand_level": row["demand_level"]
            })
            customer_id += 1

    events_df = pd.DataFrame(event_rows)

    if not events_df.empty:
        events_df = events_df.sort_values("arrival_time_min").reset_index(drop=True)
        events_df["customer_id"] = np.arange(1, len(events_df) + 1)

    return events_df

def create_shift_summary(generated_df):
    summary = (
        generated_df
        .groupby("shift_id")
        .agg(
            slices=("slice_id", "count"),
            expected_arrivals=("expected_arrivals_30min", "sum"),
            actual_arrivals=("actual_arrivals", "sum"),
            avg_lambda_per_hour=("lambda_per_hour", "mean"),
            max_lambda_per_hour=("lambda_per_hour", "max")
        )
        .reset_index()
    )

    summary["shift_time"] = summary["shift_id"].map({
        1: "00:00-08:00",
        2: "08:00-16:00",
        3: "16:00-24:00"
    })

    return summary


def create_paper_mapping():
    """
    Bảng này để bạn đưa vào report hoặc appendix:
    giải thích phần nào của model dựa vào paper nào.
    """
    rows = [
        {
            "paper": "Postorino et al. (2019)",
            "model_role": "Create airport-based demand shape",
            "how_used_in_code": (
                "The function build_airport_demand_weight() creates time-of-day "
                "demand weights using airport passenger flow logic."
            ),
            "output_supported": "raw_airport_demand_weight, normalized_demand_weight"
        },
        {
            "paper": "Kim & Whitt (2014)",
            "model_role": "Convert expected demand into stochastic arrivals",
            "how_used_in_code": (
                "Each 30-minute slice is treated as a piecewise-constant NHPP interval. "
                "Actual arrivals are generated using Poisson(lambda_i * delta_t)."
            ),
            "output_supported": "lambda_per_hour, actual_arrivals, arrival_events"
        }
    ]

    return pd.DataFrame(rows)


# =========================
# 7. EXPORT TO EXCEL
# =========================

def export_to_excel(slice_df, generated_df, events_df, shift_summary_df, output_file):
    paper_mapping_df = create_paper_mapping()

    output_path = Path(output_file)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        paper_mapping_df.to_excel(
            writer,
            sheet_name="paper_mapping",
            index=False
        )

        slice_df.to_excel(
            writer,
            sheet_name="demand_shape_48",
            index=False
        )

        generated_df[
            [
                "slice_id",
                "start_time",
                "end_time",
                "shift_id",
                "slice_in_shift",
                "demand_level",
                "raw_airport_demand_weight",
                "normalized_demand_weight",
                "expected_arrivals_30min",
                "lambda_per_hour",
                "actual_arrivals"
            ]
        ].to_excel(
            writer,
            sheet_name="generated_arrivals",
            index=False
        )

        events_df.to_excel(
            writer,
            sheet_name="arrival_events",
            index=False
        )

        shift_summary_df.to_excel(
            writer,
            sheet_name="shift_summary",
            index=False
        )

    return output_path

def main():
    slice_df = create_48_slice_baseline()
    generated_df = generate_actual_arrivals(slice_df, seed=RANDOM_SEED)
    events_df = generate_event_level_arrivals(generated_df, seed=RANDOM_SEED)
    shift_summary_df = create_shift_summary(generated_df)

    output_path = export_to_excel(
        slice_df=slice_df,
        generated_df=generated_df,
        events_df=events_df,
        shift_summary_df=shift_summary_df,
        output_file=OUTPUT_FILE
    )

    print("DONE")
    print(f"Output file: {output_path.resolve()}")
    print(f"Total expected arrivals: {generated_df['expected_arrivals_30min'].sum():.2f}")
    print(f"Total actual arrivals: {generated_df['actual_arrivals'].sum()}")
    print()
    print("Shift summary:")
    print(shift_summary_df)


if __name__ == "__main__":
    main()