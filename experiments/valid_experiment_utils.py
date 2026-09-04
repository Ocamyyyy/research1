import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import experiment1 as base


def stable_seed_for_point(point, attempt_idx):
    payload = json.dumps({"point": point, "attempt_idx": attempt_idx}, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return 100000 + int(digest[:12], 16) % 900000000


def format_case_value(value):
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value)}.0"
        return str(value)
    return str(value)


def compact_msg(point, point_keys):
    abbreviations = {
        "N": "N",
        "V": "V",
        "tot": "T",
        "alpha": "A",
        "max_out": "M",
        "NWR": "W",
        "RP_number": "R",
    }
    parts = []
    for key in point_keys:
        label = abbreviations.get(key, key[:1].upper())
        parts.append(f"{label}{format_case_value(point[key])}")
    return "_".join(parts)


def case_name(point, attempt_idx, point_keys):
    parts = [f"{key}{format_case_value(point[key])}" for key in point_keys]
    parts.append(f"attempt{attempt_idx}")
    return "_".join(parts)


def atomic_write_json(path, payload, retries=20, retry_delay_seconds=0.2):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    for attempt in range(retries):
        try:
            tmp_path.write_text(text, encoding="utf-8")
            tmp_path.replace(path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(retry_delay_seconds * (attempt + 1))


def payload_is_complete(payload):
    algorithms = payload.get("algorithms", {})
    return payload.get("status") == "completed" and all(
        name in algorithms for name in base.ALGORITHM_ORDER
    )


def payload_is_valid(payload):
    if "valid_experiment" in payload:
        return bool(payload["valid_experiment"])
    heft = payload.get("algorithms", {}).get("HEFT", {})
    return payload_is_complete(payload) and bool(heft.get("feasible"))


def result_is_feasible(metrics):
    return bool(metrics.get("feasible", False))


def load_payload(payload_path, point, attempt_idx, seed):
    if payload_path.exists():
        return json.loads(payload_path.read_text(encoding="utf-8"))
    return {
        "point": point,
        "attempt_idx": attempt_idx,
        "seed": seed,
        "algorithms": {},
        "status": "running",
        "valid_experiment": False,
    }


def iter_payloads(case_root):
    for payload_path in sorted(case_root.glob("*/result.json")):
        try:
            yield payload_path, json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception:
            continue


def summarize_point_progress(case_root, point, point_keys):
    matching = []
    for _path, payload in iter_payloads(case_root):
        payload_point = payload.get("point", {})
        if all(payload_point.get(key) == point.get(key) for key in point_keys):
            matching.append(payload)
    valid_count = sum(1 for payload in matching if payload_is_valid(payload))
    next_attempt_idx = 0
    if matching:
        next_attempt_idx = max(int(payload.get("attempt_idx", -1)) for payload in matching) + 1
    return valid_count, next_attempt_idx, matching


def collect_rows(case_root, point_keys):
    rows = []
    for _payload_path, payload in iter_payloads(case_root):
        point = payload.get("point", {})
        algorithms = payload.get("algorithms", {})
        row = {
            key: point.get(key) for key in point_keys
        }
        row["attempt_idx"] = payload.get("attempt_idx")
        row["status"] = payload.get("status", "unknown")
        row["valid_experiment"] = payload_is_valid(payload)
        row["seed"] = payload.get("seed")
        for algorithm_name in base.ALGORITHM_ORDER:
            metrics = algorithms.get(algorithm_name, {})
            row[f"{algorithm_name} energy"] = metrics.get("energy_percent")
            row[f"{algorithm_name} delay"] = metrics.get("delay")
            row[f"{algorithm_name} feasible"] = metrics.get("feasible")
            row[f"{algorithm_name} runtime_seconds"] = metrics.get("runtime_seconds")
        rows.append(row)
    return rows


def write_summary_files(
    result_root,
    case_root,
    point_keys,
    file_prefix,
    plot_callback=None,
):
    rows = collect_rows(case_root, point_keys)
    if not rows:
        return

    result_root.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    sort_cols = [key for key in point_keys if key in raw_df.columns] + ["attempt_idx"]
    raw_df = raw_df.sort_values(by=sort_cols, kind="stable")

    raw_csv_path = result_root / f"{file_prefix}_raw_results.csv"
    raw_xlsx_path = result_root / f"{file_prefix}_raw_results.xlsx"
    raw_df.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
    try:
        raw_df.to_excel(raw_xlsx_path, index=False)
    except Exception as exc:
        print(f"[WARN] Failed to write raw Excel file: {exc}")

    completed_df = raw_df[raw_df["status"] == "completed"].copy()
    if completed_df.empty:
        summary_df = pd.DataFrame(columns=point_keys)
    else:
        summary_rows = []
        for group_values, group in completed_df.groupby(point_keys, sort=True):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            row = {key: value for key, value in zip(point_keys, group_values)}
            row["valid_count"] = int(len(group))
            row["total_attempts"] = int(
                len(raw_df[(raw_df[point_keys] == pd.Series(row)[point_keys]).all(axis=1)])
            )
            for algorithm_name in base.ALGORITHM_ORDER:
                energy_col = f"{algorithm_name} energy"
                delay_col = f"{algorithm_name} delay"
                feasible_col = f"{algorithm_name} feasible"
                runtime_col = f"{algorithm_name} runtime_seconds"
                row[f"{algorithm_name} avg_energy"] = group[energy_col].dropna().mean()
                row[f"{algorithm_name} avg_delay"] = group[delay_col].dropna().mean()
                feasible_series = group[feasible_col].dropna()
                row[f"{algorithm_name} feasible_rate"] = feasible_series.astype(float).mean() if not feasible_series.empty else float("nan")
                row[f"{algorithm_name} all_feasible"] = bool(feasible_series.all()) if not feasible_series.empty else False
                row[f"{algorithm_name} avg_runtime_seconds"] = group[runtime_col].dropna().mean()
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(by=point_keys, kind="stable")

    summary_csv_path = result_root / f"{file_prefix}_summary.csv"
    summary_xlsx_path = result_root / f"{file_prefix}_summary.xlsx"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    try:
        summary_df.to_excel(summary_xlsx_path, index=False)
    except Exception as exc:
        print(f"[WARN] Failed to write summary Excel file: {exc}")

    try:
        base.highlight_best_rows(raw_xlsx_path)
    except Exception as exc:
        print(f"[WARN] Failed to highlight raw Excel file: {exc}")

    if plot_callback is not None:
        try:
            plot_callback(summary_df, result_root)
        except Exception as exc:
            print(f"[WARN] Failed to draw result figure: {exc}")


def print_algorithm_metrics(case_name_text, algorithm_name, metrics):
    energy = metrics.get("energy_percent")
    delay = metrics.get("delay")
    runtime_seconds = metrics.get("runtime_seconds")
    feasible = metrics.get("feasible")

    energy_text = "NA" if energy is None else f"{float(energy):.2f}%"
    delay_text = "NA" if delay is None else f"{float(delay):.2f}"
    runtime_text = "NA" if runtime_seconds is None else f"{float(runtime_seconds):.2f}s"
    feasible_text = ""
    if feasible is not None:
        feasible_text = f" | feasible={bool(feasible)}"

    print(
        f"  [DONE] {case_name_text} -> {algorithm_name}"
        f" | energy={energy_text}"
        f" | delay={delay_text}"
        f" | time={runtime_text}"
        f"{feasible_text}"
    )


def run_attempt(
    point,
    attempt_idx,
    point_keys,
    case_root,
    build_task_fn,
    require_heft_feasible=False,
    require_all_algorithms_feasible=False,
):
    seed = stable_seed_for_point(point, attempt_idx)
    name = case_name(point, attempt_idx, point_keys)
    case_dir = case_root / name
    artifact_root = case_dir / "artifacts"
    payload_path = case_dir / "result.json"
    payload = load_payload(payload_path, point, attempt_idx, seed)

    if payload.get("taskset") is None:
        task, _ = build_task_fn(point, attempt_idx, seed)
        payload["taskset"] = base.serialize_taskset(task)
        payload.setdefault("artifacts", {})["task_graph_images"] = base.save_task_graph_images(
            task, case_dir / "task_graphs"
        )
        atomic_write_json(payload_path, payload)

    if payload_is_complete(payload) or payload.get("status") in {"invalid_heft", "invalid_algorithm"}:
        return payload

    task, _ = build_task_fn(point, attempt_idx, seed)
    payload["status"] = "running"
    atomic_write_json(payload_path, payload)

    msg = compact_msg(point, point_keys)

    try:
        heft_result = base.run_heft(task, seed, artifact_root / "HEFT")
        payload["algorithms"]["HEFT"] = heft_result
        atomic_write_json(payload_path, payload)
        print_algorithm_metrics(name, "HEFT", heft_result)

        if require_heft_feasible and not heft_result.get("feasible", False):
            payload["status"] = "invalid_heft"
            payload["valid_experiment"] = False
            atomic_write_json(payload_path, payload)
            print(f"[INVALID] {name} skipped because HEFT is infeasible")
            return payload

        runners = {
            "SAFLA_adapted": lambda: base.run_safla(task, seed, artifact_root / "SAFLA_adapted"),
            "eFLS_adapted": lambda: base.run_efls(task, seed, artifact_root / "eFLS_adapted"),
            "GA": lambda: base.run_ga(task, artifact_root / "GA", msg, attempt_idx, seed),
            "HHO": lambda: base.run_hho(task, artifact_root / "HHO", msg, attempt_idx, seed),
            "QHA": lambda: base.run_qha(task, artifact_root / "QHA", msg, attempt_idx, seed),
        }

        for algorithm_name in base.ALGORITHM_ORDER[1:]:
            if algorithm_name in payload["algorithms"]:
                continue
            (artifact_root / algorithm_name).mkdir(parents=True, exist_ok=True)
            print(f"  [RUN ] {name} -> {algorithm_name}")
            payload["algorithms"][algorithm_name] = runners[algorithm_name]()
            atomic_write_json(payload_path, payload)
            print_algorithm_metrics(name, algorithm_name, payload["algorithms"][algorithm_name])
            if require_all_algorithms_feasible and not result_is_feasible(payload["algorithms"][algorithm_name]):
                payload["status"] = "invalid_algorithm"
                payload["valid_experiment"] = False
                payload["invalid_algorithm"] = algorithm_name
                atomic_write_json(payload_path, payload)
                print(f"[INVALID] {name} skipped because {algorithm_name} is infeasible")
                return payload

        payload["status"] = "completed"
        payload["valid_experiment"] = True
        atomic_write_json(payload_path, payload)
        return payload
    except Exception as exc:
        payload["status"] = "failed"
        payload["valid_experiment"] = False
        payload["error"] = repr(exc)
        atomic_write_json(payload_path, payload)
        raise


def run_experiment(
    points,
    point_keys,
    result_root,
    build_task_fn,
    file_prefix,
    required_valid_count=5,
    plot_callback=None,
    require_heft_feasible=False,
    require_all_algorithms_feasible=False,
    refresh_summary_during_run=True,
):
    case_root = result_root / "cases"
    result_root.mkdir(parents=True, exist_ok=True)
    case_root.mkdir(parents=True, exist_ok=True)

    total_points = len(points)
    for point_index, point in enumerate(points, start=1):
        valid_count, next_attempt_idx, _matching = summarize_point_progress(case_root, point, point_keys)
        print(
            f"[POINT] {point_index}/{total_points} "
            f"{json.dumps(point, ensure_ascii=False)} | valid={valid_count}/{required_valid_count}"
        )
        while valid_count < required_valid_count:
            payload = run_attempt(
                point,
                next_attempt_idx,
                point_keys,
                case_root,
                build_task_fn,
                require_heft_feasible=require_heft_feasible,
                require_all_algorithms_feasible=require_all_algorithms_feasible,
            )
            if payload_is_valid(payload):
                valid_count += 1
            next_attempt_idx += 1
            if refresh_summary_during_run:
                write_summary_files(
                    result_root=result_root,
                    case_root=case_root,
                    point_keys=point_keys,
                    file_prefix=file_prefix,
                    plot_callback=plot_callback,
                )
            print(
                f"[PROGRESS] point={json.dumps(point, ensure_ascii=False)} "
                f"valid={valid_count}/{required_valid_count} next_attempt={next_attempt_idx}"
            )

    if refresh_summary_during_run:
        write_summary_files(
            result_root=result_root,
            case_root=case_root,
            point_keys=point_keys,
            file_prefix=file_prefix,
            plot_callback=plot_callback,
        )
