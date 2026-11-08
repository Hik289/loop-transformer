"""Atomically refresh the original experiment tables from validated final runs.

Only the standard library is required.  A manifest defines the row universe;
only its direct children under the explicitly supplied results root are read.
Configuration ``output_dir`` values are deliberately never used as paths.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .sparsedrop_export import build_final_sparsity


BEGIN = "<!-- BEGIN SPARSEDROP AUTO RESULTS -->"
END = "<!-- END SPARSEDROP AUTO RESULTS -->"
LABELS = {"none": "Baseline", "ffn": "FFN only", "attention": "Self-Attention only", "both": "Both"}
MODE_ORDER = {"none": 0, "ffn": 1, "attention": 2, "both": 3}
CONDITION_KEYS = ("loops", "mode", "p", "schedule", "delta", "seed")
SCIENTIFIC_KEYS = ("batch_size", "grad_accum", "seq_len", "max_steps", "token_budget", "stop_after_steps",
                   "data_seed", "lr", "min_lr_ratio", "weight_decay", "warmup_steps", "clip_grad",
                   "vocab_size", "hidden_size", "num_layers", "num_heads", "intermediate_size",
                   "gradient_checkpointing", "diagnostics", "diagnostic_tokens", "eval_chunks", "eval_batch_size")
STATUS_LABELS = {"complete": "完成（精确预算）", "overbudget": "超额；不纳入主表",
                 "unfinished": "未完成", "invalid": "最终结果校验失败",
                 "budget_mismatch": "预算不匹配；不纳入主表"}


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return float(value)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _target(config: dict[str, Any]) -> tuple[int, int]:
    """Derive the current manifest target, independently of stale run metadata."""
    args = config.get("arguments", config)
    per_step = math.prod(_integer(args.get(k), k, 1) for k in ("batch_size", "grad_accum", "seq_len"))
    steps = _integer(args.get("max_steps"), "max_steps", 1)
    cap = args.get("stop_after_steps")
    if cap is not None:
        cap = _integer(cap, "stop_after_steps")
        if cap:
            steps = min(steps, cap)
    tokens = args.get("token_budget")
    if tokens is not None:
        tokens = _integer(tokens, "token_budget", 1)
        steps = min(steps, math.ceil(tokens / per_step))
    return steps, min(steps * per_step, tokens) if tokens is not None else steps * per_step


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Manifest must be a nonempty list")
    entries, names, conditions = [], set(), set()
    for entry in manifest:
        name = entry.get("name")
        if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Manifest run names must be direct directory names")
        if name in names:
            raise ValueError(f"Duplicate manifest run name: {name}")
        names.add(name)
        # Manifest paths are repository-relative; a colocated basename also
        # supports copied manifests without consulting machine-specific paths.
        supplied = Path(entry["config"])
        candidates = [manifest_path.parent / supplied.name]
        if not supplied.is_absolute():
            candidates += [parent / supplied for parent in manifest_path.parents]
        config_path = next((p for p in candidates if p.is_file()), None)
        if config_path is None:
            raise ValueError(f"Manifest configuration not found: {entry['config']}")
        config = _read_json(config_path)
        if not isinstance(config, dict):
            raise ValueError(f"Manifest configuration must be an object: {config_path}")
        args = config.get("arguments", config)
        condition = {key: args[key] for key in CONDITION_KEYS}
        if condition["mode"] not in LABELS or condition["schedule"] not in {"constant", "increasing", "decreasing"}:
            raise ValueError(f"Unknown experiment condition: {name}")
        key = tuple(condition.values())
        if key in conditions:
            raise ValueError(f"Duplicate manifest condition: {name}")
        conditions.add(key)
        steps, tokens = _target(config)
        entries.append({"name": name, "index": entry.get("index", len(entries)), **condition,
                        "effective_target_steps": steps, "effective_target_tokens": tokens,
                        "expected_arguments": {key: args[key] for key in SCIENTIFIC_KEYS if key in args}})
    return sorted(entries, key=lambda x: x["index"])


def _read_train(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], []
    records, warnings = [], []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
                records.append(record)
            except (ValueError, TypeError) as error:
                warnings.append(f"{path.name}:{number}: {error}")
    return records, warnings


def _quality(final: Any) -> dict[str, Any]:
    if not isinstance(final, dict):
        raise ValueError("final_full_validation must be an object")
    loss = _finite(final.get("loss"), "final_full_validation.loss", minimum=0)
    ppl = _finite(final.get("perplexity"), "final_full_validation.perplexity", minimum=1)
    if not math.isclose(math.log(ppl), loss, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("final_full_validation perplexity does not equal exp(loss)")
    tokens = _integer(final.get("tokens"), "final_full_validation.tokens", 1)
    return {"loss": loss, "perplexity": ppl, "validation_tokens": tokens}


def _validate_full_scope(final: dict[str, Any], config: dict[str, Any]) -> None:
    if final.get("evaluation_scope") != "all_complete_validation_chunks":
        raise ValueError("Final validation evaluation_scope must be all_complete_validation_chunks")
    data = config.get("data")
    validation = data.get("validation") if isinstance(data, dict) else None
    if not isinstance(validation, dict):
        raise ValueError("Run config.data.validation signature is required")
    seq_len = _integer(config["arguments"].get("seq_len"), "config.arguments.seq_len", 1)
    raw_tokens = _integer(validation.get("tokens"), "config.data.validation.tokens", 2)
    chunks, tail = divmod(raw_tokens - 1, seq_len)
    expected = {"chunks": chunks, "tokens": chunks * seq_len, "trailing_prediction_tokens_excluded": tail}
    for key, value in expected.items():
        if _integer(final.get(key), f"final_full_validation.{key}") != value:
            raise ValueError(f"Final validation {key} does not cover all complete validation chunks")
    for key, value in {"chunks": chunks, "prediction_tokens": chunks * seq_len,
                       "trailing_prediction_tokens_excluded": tail}.items():
        if _integer(validation.get(key), f"config.data.validation.{key}") != value:
            raise ValueError(f"Run validation signature {key} disagrees with original token count")


def _diagnostic_metrics(diagnostics: dict[str, Any], sparsity: dict[str, Any], loops: int) -> dict[str, Any]:
    result = {}
    for branch in ("ffn", "attention"):
        rows = [x for x in sparsity["activations"] if x["branch"] == branch]
        weights = [r["tokens"] * r["hidden_size"] for r in rows]
        ratios = [next(x["near_zero_ratio"] for x in r["threshold_sensitivity"] if x["threshold"] == .01) for r in rows]
        result[f"{branch}_near_zero_eps0.01"] = sum(w * x for w, x in zip(weights, ratios)) / sum(weights)
    states = diagnostics.get("loop_states")
    if (not isinstance(states, list) or not all(isinstance(x, dict) for x in states)
            or sorted(_integer(x.get("loop"), "loop_states.loop", 1) for x in states) != list(range(1, loops + 1))):
        raise ValueError("diagnostics.loop_states has incomplete or duplicate loop coverage")
    result["mean_update_l2"] = sum(_finite(x.get("update_mean_l2"), "update_mean_l2", minimum=0) for x in states) / loops
    result["mean_update_cosine"] = None
    cross = diagnostics.get("cross_loop")
    if not isinstance(cross, dict) or cross.get("loop_indices") != list(range(1, loops + 1)):
        raise ValueError("diagnostics.cross_loop must be an object covering all configured loops")
    if loops > 1:
        value = _finite(cross.get("mean_update_cosine"), "mean_update_cosine")
        if not -1.000000000001 <= value <= 1.000000000001 or cross.get("status") != "ok":
            raise ValueError("Invalid cross-loop update cosine")
        matrix = cross.get("update_cosine")
        if not isinstance(matrix, list) or len(matrix) != loops or not all(isinstance(r, list) and len(r) == loops for r in matrix):
            raise ValueError("cross_loop.update_cosine must be a complete H × H matrix")
        for i in range(loops):
            for j in range(loops):
                point = _finite(matrix[i][j], "update_cosine matrix")
                if not -1.000000000001 <= point <= 1.000000000001 or not math.isclose(point, matrix[j][i], abs_tol=1e-12):
                    raise ValueError("update_cosine matrix must be bounded and symmetric")
        measured = sum(matrix[i][j] for i in range(loops) for j in range(i + 1, loops)) * 2 / (loops * (loops - 1))
        if not math.isclose(value, measured, abs_tol=1e-12):
            raise ValueError("mean_update_cosine differs from the matrix off-diagonal mean")
        result["mean_update_cosine"] = value
    elif cross.get("status") != "not_applicable_H1" or cross.get("mean_update_cosine") is not None:
        raise ValueError("H1 cross-loop diagnostics must be not_applicable_H1")
    return result


def _relative(path: Path, document: Path) -> str:
    return Path(os.path.relpath(path, document.parent)).as_posix()


def scan_results(manifest_path: Path, results_root: Path) -> list[dict[str, Any]]:
    """Return one row per manifest condition, without using partial quality data."""
    root = Path(results_root).resolve()
    rows = []
    for entry in load_manifest(manifest_path):
        row = {**entry, "status": "unfinished", "stage": "待运行", "actual_steps": None,
               "actual_tokens_seen": None, "training_tokens_B": None, "metrics": None,
               "historical_fixed_100k": None, "errors": [], "warnings": [], "sources": {}}
        rows.append(row)
        run_dir = root / entry["name"]
        # A symlink could silently import another budget or an archived run.
        if run_dir.is_symlink():
            row.update(status="invalid", stage="目录校验失败", errors=["Run directory must not be a symlink"])
            continue
        if not run_dir.is_dir():
            continue
        records, warnings = _read_train(run_dir / "train.jsonl")
        row["warnings"].extend(warnings)
        row["stage"] = "已创建／等待训练记录"
        for record in records:
            step, tokens = record.get("step"), record.get("tokens_seen")
            if type(step) is int and step >= 0:
                row["actual_steps"] = step
            if type(tokens) is int and tokens >= 0:
                row["actual_tokens_seen"] = tokens
            row["stage"] = str(record.get("event", "训练中"))
        if row["actual_tokens_seen"] is not None:
            row["training_tokens_B"] = row["actual_tokens_seen"] / 1e9
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            if row["actual_tokens_seen"] is not None:
                row["training_tokens_B"] = row["actual_tokens_seen"] / 1e9
            continue
        row["sources"]["summary"] = str(summary_path)
        try:
            summary = _read_json(summary_path)
            if not isinstance(summary, dict):
                raise ValueError("summary.json must be an object")
            if summary.get("status") not in {"complete", "stopped_over_budget"}:
                row["stage"] = str(summary.get("status", "等待有效最终摘要"))
                continue
            config = _read_json(run_dir / "config.json")
            if not isinstance(config, dict) or not isinstance(config.get("arguments"), dict):
                raise ValueError("Run config must contain arguments")
            for key in CONDITION_KEYS:
                if config["arguments"].get(key) != entry[key]:
                    raise ValueError(f"Run config condition {key} differs from manifest")
            for key, value in entry["expected_arguments"].items():
                if config["arguments"].get(key) != value:
                    raise ValueError(f"Run config scientific argument {key} differs from manifest")
            for key in ("status", "actual_steps", "actual_tokens_seen", "effective_target_steps", "effective_target_tokens"):
                if key in config and config[key] != summary.get(key):
                    raise ValueError(f"Run config {key} disagrees with summary")
            if _target(config) != (summary.get("effective_target_steps"), summary.get("effective_target_tokens")):
                raise ValueError("Run config effective target computed from arguments disagrees with summary")
            sparsity = build_final_sparsity(summary, config, run_name=entry["name"])
            metrics = _quality(summary.get("final_full_validation"))
            final = summary["final_full_validation"]
            _validate_full_scope(final, config)
            for key in ("step", "tokens_seen"):
                if key in final and final[key] != summary[key]:
                    raise ValueError(f"Final validation {key} differs from summary")
            metrics.update(_diagnostic_metrics(final["diagnostics"], sparsity, entry["loops"]))
            metrics["diagnostic_sample_tokens"] = sparsity["protocol"]["sampled_tokens"]
            row.update(actual_steps=sparsity["actual_steps"], actual_tokens_seen=sparsity["actual_tokens_seen"],
                       training_tokens_B=sparsity["training_tokens_B"], metrics=metrics)
            # Missing target declarations are rejected rather than inferred from
            # successful looking loss curves or an old checkpoint's filename.
            declared = (_integer(summary.get("effective_target_steps"), "summary.effective_target_steps", 1),
                        _integer(summary.get("effective_target_tokens"), "summary.effective_target_tokens", 1))
            current = (entry["effective_target_steps"], entry["effective_target_tokens"])
            actual = (row["actual_steps"], row["actual_tokens_seen"])
            if declared != current:
                row.update(status="budget_mismatch", stage="最终验证完成；预算不匹配")
                row["errors"].append(f"Summary target {declared} differs from manifest target {current}")
            elif actual[0] > current[0] or actual[1] > current[1] or summary["status"] == "stopped_over_budget":
                row.update(status="overbudget", stage="最终全量验证完成；超额隔离")
            elif actual != current:
                raise ValueError(f"Completed run actual step/token counts {actual} do not equal target {current}")
            else:
                row.update(status="complete", stage="最终全量验证及稀疏度完整")
            throughput = summary.get("mean_train_tokens_per_second")
            memory = summary.get("peak_gpu_memory_bytes")
            metrics["mean_train_B_per_second"] = None if throughput is None else _finite(throughput, "mean_train_tokens_per_second", minimum=0) / 1e9
            metrics["peak_memory_GiB"] = None if memory is None else _finite(memory, "peak_gpu_memory_bytes", minimum=0) / (1024 ** 3)
            train = [r for r in records if r.get("event") == "train" and type(r.get("step")) is int
                     and r["step"] <= row["actual_steps"] and r.get("grad_norm_preclip") is not None]
            metrics.update(gradient_norm_last_window=None, gradient_window_end_step=None)
            if train:
                last = train[-1]
                metrics["gradient_norm_last_window"] = _finite(last["grad_norm_preclip"], "train.grad_norm_preclip", minimum=0)
                metrics["gradient_window_end_step"] = last["step"]
                row["sources"]["train"] = str(run_dir / "train.jsonl")
            for key, filename in (("sparsity", "sparsity_final.json"), ("sparsity_readme", "README_sparsity_final.md")):
                path = run_dir / filename
                if path.is_file():
                    row["sources"][key] = str(path)
            if row["status"] == "overbudget":
                # Historical fixed validation is explicitly separate from both
                # the main final-full table and the actual terminal evaluation.
                evaluations, eval_warnings = _read_train(run_dir / "eval.jsonl")
                row["warnings"].extend(eval_warnings)
                matches = [e for e in evaluations if e.get("event") == "validation_fixed"
                           and e.get("step") == 100_000 and e.get("tokens_seen") == 1_638_400_000]
                if matches:
                    try:
                        historical = _quality(matches[-1])
                        if any(_quality(e) != historical for e in matches):
                            raise ValueError("Conflicting historical 100k fixed-validation records")
                        row["historical_fixed_100k"] = historical
                        row["sources"]["historical_fixed"] = str(run_dir / "eval.jsonl")
                    except ValueError as error:
                        row["warnings"].append(str(error))
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            row.update(status="invalid", stage="最终结果校验失败", metrics=None)
            row["errors"].append(str(error))
        if row["actual_tokens_seen"] is not None:
            row["training_tokens_B"] = row["actual_tokens_seen"] / 1e9
    return rows


def _number(value: Any, decimals: int = 6) -> str:
    return "—" if value is None else f"{value:.{decimals}f}"


def _status(row: dict[str, Any]) -> str:
    return STATUS_LABELS[row["status"]]


def _progress(row: dict[str, Any]) -> list[str]:
    return [_status(row), "—" if row["actual_steps"] is None else f"{row['actual_steps']:,}",
            _number(row["training_tokens_B"])]


def _metrics(row: dict[str, Any], *, allow_excluded: bool = False) -> dict[str, Any]:
    return (row["metrics"] or {}) if allow_excluded or row["status"] == "complete" else {}


def _metric_cells(row: dict[str, Any], fields: list[str], *, allow_excluded: bool = False) -> list[str]:
    metrics = _metrics(row, allow_excluded=allow_excluded)
    return ["N/A" if field == "mean_update_cosine" and row["loops"] == 1 and metrics
            else _number(metrics.get(field), 9 if field == "mean_train_B_per_second" else 6) for field in fields]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |" for row in rows])


def _sort(row: dict[str, Any]) -> tuple:
    return row["loops"], row["p"], MODE_ORDER[row["mode"]], row["schedule"], row["delta"]


def _link(path: str, label: str, document: Path) -> str:
    return f"[{label}](<{_relative(Path(path), document)}>)"


def _source_links(row: dict[str, Any], document: Path) -> str:
    sources = row["sources"]
    links = []
    if row["metrics"] and "summary" in sources:
        links.append(_link(sources["summary"], "最终 quality / diagnostics", document))
        if "sparsity" in sources:
            links.append(_link(sources["sparsity"], "sparsity JSON", document))
        if "sparsity_readme" in sources:
            links.append(_link(sources["sparsity_readme"], "逐 loop / 层", document))
    return " · ".join(links) or "—"


def render_results_section(rows: list[dict[str, Any]], document: Path) -> str:
    fixed = sorted([r for r in rows if r["schedule"] == "constant"], key=_sort)
    progress = ["状态", "实际步数", "训练量 B"]
    valid_metrics = [r["metrics"] for r in rows if r["metrics"]]
    validation_sizes = sorted({m["validation_tokens"] for m in valid_metrics})
    diagnostic_sizes = sorted({m["diagnostic_sample_tokens"] for m in valid_metrics})
    sample_note = ("已验证的完整验证量：" + "、".join(f"{x / 1e9:.9f} B" for x in validation_sizes)
                   + "；激活/多样性诊断样本量：" + "、".join(f"{x / 1e9:.9f} B" for x in diagnostic_sizes)
                   + "。诊断样本量与完整验证量不同。") if valid_metrics else "验证量与诊断样本量将在最终结果通过校验后列出。"
    lines = ["## 5. 实验结果表（自动更新）", "", BEGIN, "",
             "本区按 manifest 固定实验条件，完成一组即填入一组；未完成或校验失败的指标为 **—**。"
             "四张主表仅纳入 `status=complete` 且实际步数、实际 token 数均精确等于当前 manifest 有效目标的最终结果。",
             "当前目标为 " + "、".join(f"{s:,} 步 / {t / 1e9:.6f} B" for s, t in sorted({(r['effective_target_steps'], r['effective_target_tokens']) for r in rows})) + "；1 B = 10⁹ tokens。",
             "Loss / PPL 来自 `summary.json.final_full_validation` 的完整验证集；不填入训练 loss、途中 fixed validation 或 best checkpoint 指标。"
             "超额和预算不匹配结果保留实际步数与训练量，质量指标见独立表，不能与 100k 主表混合比较。", sample_note, "",
             "Gradient norm 是 `train.jsonl` 最后一个不晚于最终步数的 `train` 记录中 `grad_norm_preclip`（该日志窗口的裁剪前均值）；"
             "其窗口截止步数单独列出，不是最终验证指标，也不是 `grad_norm_preclip_last`。"
             "吞吐是 summary 的全程 `mean_train_tokens_per_second / 10⁹`，单位 B/s；显存是 `peak_gpu_memory_bytes / 2³⁰`，单位 GiB。", "",
             "近零主列只统计 **FFN residual 分支、ε=0.01**，Attention 单列："
             "在 eval 模式下，对每个 loop × 物理层的比例按 `tokens × hidden_size` 加权平均；"
             "位置为 sublayer RMSNorm 后、Dropout 与残差相加前，不是 SwiGLU 中间神经元稀疏度。"
             "完整性要求所有 loop × 层 × Attention/FFN 均存在。",
             "Update cosine 为诊断的跨 loop 非对角均值 `cross_loop.mean_update_cosine`；H=1 记 N/A。"
             "Update norm 为各 loop 的 `update_mean_l2` 等权均值（先逐 token 求更新向量 L2 范数再平均），不使用 update RMS。", "",
             "### Table 1. 主实验：Loop Depth × Dropout Rate × Dropout 方案（40 个唯一固定概率条件）", ""]
    fields = ["loss", "perplexity", "gradient_norm_last_window", "mean_train_B_per_second"]
    body = []
    for row in fixed:
        end = _metrics(row).get("gradient_window_end_step")
        body.append([str(row["loops"]), f"{row['p']:.2f}", LABELS[row["mode"]], *_progress(row),
                     *_metric_cells(row, fields), "—" if end is None else f"{end:,}"])
    lines += [_table(["H", "p", "方案", *progress, "Validation loss ↓", "PPL ↓", "Gradient norm（末日志窗口）", "吞吐 B/s", "梯度窗口截止步"], body), "",
              "### Table 2. Dropout 位置消融（H=4，baseline + p=0.10 三种位置）", ""]
    subset = [r for r in fixed if r["loops"] == 4 and r["p"] in (0, .1)]
    fields = ["loss", "perplexity", "mean_update_cosine", "ffn_near_zero_eps0.01", "attention_near_zero_eps0.01", "peak_memory_GiB"]
    lines += [_table(["位置", "p", *progress, "Loss ↓", "PPL ↓", "Update cosine", "FFN 近零 ε=.01", "Attention 近零 ε=.01", "Peak memory GiB"],
                     [[LABELS[r["mode"]], f"{r['p']:.2f}", *_progress(r), *_metric_cells(r, fields)] for r in subset]), "",
              "### Table 3. 循环概率调度（H=8，p=0.10；三方案 × 五调度，共15行）", ""]
    subset = sorted([r for r in rows if r["loops"] == 8 and r["p"] == .1], key=_sort)
    fields = ["loss", "perplexity", "mean_update_cosine", "ffn_near_zero_eps0.01", "attention_near_zero_eps0.01"]
    lines += [_table(["方案", "调度", "平均 p", "δ", *progress, "Loss ↓", "PPL ↓", "Update cosine", "FFN 近零 ε=.01", "Attention 近零 ε=.01"],
                     [[LABELS[r["mode"]], r["schedule"], f"{r['p']:.2f}", f"{r['delta']:.2f}", *_progress(r), *_metric_cells(r, fields)] for r in subset]), "",
              "### Table 4. Diversity / Sparsity 与质量关联（H=4/8 固定概率，20个条件）", ""]
    fields = ["mean_update_cosine", "mean_update_l2", "ffn_near_zero_eps0.01", "attention_near_zero_eps0.01", "perplexity"]
    lines += [_table(["H", "p", "方案", *progress, "Mean update cosine", "Mean update L2 norm", "FFN 近零 ε=.01", "Attention 近零 ε=.01", "Validation PPL ↓"],
                     [[str(r["loops"]), f"{r['p']:.2f}", LABELS[r["mode"]], *_progress(r), *_metric_cells(r, fields)] for r in fixed if r["loops"] in (4, 8)]), "",
              "### 独立表：超额／其他预算的实际终点（不纳入以上主表）", ""]
    excluded = [r for r in rows if r["status"] in {"overbudget", "budget_mismatch"} and r["metrics"]]
    fields = ["loss", "perplexity", "ffn_near_zero_eps0.01", "attention_near_zero_eps0.01"]
    lines += [_table(["实验", *progress, "实际终点 full loss", "实际终点 full PPL", "FFN 近零 ε=.01", "Attention 近零 ε=.01", "来源"],
                     [[r["name"], *_progress(r), *_metric_cells(r, fields, allow_excluded=True), _source_links(r, document)] for r in excluded]) if excluded else "暂无通过校验的超额或其他预算最终结果。", "",
              "### 独立参考：超额实验的历史 100k fixed validation", "",
              "以下只供历史参考，训练量为1.638400 B；验证样本数另列。它不是实际终点 full validation，也不回填主表。", ""]
    history = [r for r in rows if r["historical_fixed_100k"]]
    lines += [_table(["实验", "历史步数", "Fixed 验证量 B", "历史 fixed loss", "历史 fixed PPL", "实际最终步数", "来源"],
                     [[r["name"], "100,000", _number(r["historical_fixed_100k"]["validation_tokens"] / 1e9, 9),
                       _number(r["historical_fixed_100k"]["loss"]), _number(r["historical_fixed_100k"]["perplexity"]),
                       f"{r['actual_steps']:,}", _link(r["sources"]["historical_fixed"], "eval.jsonl（validation_fixed）", document)] for r in history]) if history else "暂无可用历史 fixed 参考。", "", END, ""]
    return "\n".join(lines)


def replace_results_section(original: str, rendered: str) -> str:
    """Replace only section 5; preserve all bytes outside its heading boundaries."""
    starts = list(re.finditer(r"^## 5[.．]\s.*$", original, re.MULTILINE))
    ends = list(re.finditer(r"^## 6[.．]\s.*$", original, re.MULTILINE))
    if len(starts) != 1 or len(ends) != 1 or starts[0].start() >= ends[0].start():
        raise ValueError("README must contain exactly one section 5 before exactly one section 6")
    block = original[starts[0].start():ends[0].start()]
    if block.count(BEGIN) != block.count(END) or block.count(BEGIN) > 1:
        raise ValueError("README automatic-result markers are unbalanced or duplicated")
    if BEGIN in block and block.index(BEGIN) >= block.index(END):
        raise ValueError("README automatic-result markers are reversed")
    return original[:starts[0].start()] + rendered.rstrip() + "\n\n" + original[ends[0].start():]


def render_live_readme(rows: list[dict[str, Any]], document: Path) -> str:
    counts = Counter(r["status"] for r in rows)
    lines = ["# 实验结果实时表（manifest 全部条件）", "",
             f"共 **{len(rows)}** 组：精确预算完成 **{counts['complete']}**，超额 **{counts['overbudget']}**，"
             f"未完成 **{counts['unfinished']}**，校验失败 **{counts['invalid']}**，预算不匹配 **{counts['budget_mismatch']}**。", "",
             "按 manifest 固定行，每个实验完成并通过校验后填入最终数据。未完成行不展示中途质量或稀疏度。"
             "超额行的最终指标对应其真实终点，不能用于精确100k配对。1 B = 10⁹ tokens；近零比例为 ε=.01 的全部 loop × 层加权均值。", ""]
    fields = ["loss", "perplexity", "ffn_near_zero_eps0.01", "attention_near_zero_eps0.01"]
    lines += [_table(["#", "实验", "阶段", "状态", "实际步数", "训练量 B", "最终 full loss", "最终 full PPL", "FFN 近零", "Attention 近零", "最终质量 / 稀疏度文件"],
                     [[str(r["index"]), r["name"], r["stage"], *_progress(r), *_metric_cells(r, fields, allow_excluded=True), _source_links(r, document)] for r in rows]), ""]
    problems = [(r["name"], issue) for r in rows for issue in r["errors"] + r["warnings"]]
    if problems:
        lines += ["## 校验错误与读取提示", ""]
        lines += [f"- `{name}`：{issue}" for name, issue in problems]
        lines.append("")
    return "\n".join(lines)


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def update_readme_results(*, manifest_path: Path, results_root: Path, readme_path: Path,
                          live_readme_path: Path | None = None, status_path: Path | None = None) -> dict[str, Any]:
    """Lock, rescan, validate, and atomically replace all derived result files.

    The lock covers source scanning AND rereading the README. Concurrent finish
    callbacks therefore cannot publish an earlier pre-lock snapshot over a newer
    one. Every writer for a given report must use the same original README path.
    """
    readme_path = Path(readme_path).resolve()
    live_readme_path = Path(live_readme_path).resolve() if live_readme_path else readme_path.with_name("README_实验结果实时表_CN.md")
    status_path = Path(status_path).resolve() if status_path else readme_path.with_name("sparsedrop_readme_status.json")
    if len({readme_path, live_readme_path, status_path}) != 3:
        raise ValueError("The README, live table and status JSON must be distinct files")
    lock_path = readme_path.with_name(readme_path.name + ".sparsedrop.lock")
    with _exclusive_lock(lock_path):
        rows = scan_results(Path(manifest_path), Path(results_root))
        # newline='' preserves CRLF and all non-section-5 source bytes.
        with readme_path.open(encoding="utf-8", newline="") as stream:
            original = stream.read()
        updated = replace_results_section(original, render_results_section(rows, readme_path))
        live = render_live_readme(rows, live_readme_path)
        counts = Counter(r["status"] for r in rows)
        counts = {"total": len(rows), "finished_valid_exact": counts["complete"], "overbudget": counts["overbudget"],
                  "unfinished": counts["unfinished"], "invalid": counts["invalid"], "budget_mismatch": counts["budget_mismatch"]}
        serialized_rows = [{**r, "sources": {key: _relative(Path(path), status_path) for key, path in r["sources"].items()}} for r in rows]
        payload = {"schema_version": "sparsedrop_readme_results_v1", "counts": counts,
                   "manifest": _relative(Path(manifest_path).resolve(), status_path),
                   "results_root": _relative(Path(results_root).resolve(), status_path),
                   "readme": _relative(readme_path, status_path), "live_readme": _relative(live_readme_path, status_path),
                   "rows": serialized_rows}
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        _write_atomic(readme_path, updated)
        _write_atomic(live_readme_path, live)
        _write_atomic(status_path, encoded)
    return {**payload, "readme_path": str(readme_path), "live_readme_path": str(live_readme_path), "status_path": str(status_path)}
