"""Read-only training counters and a dependency-free live terminal display."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import deque
import csv
import io
import math
import os
from pathlib import Path
import re
import shutil
import select
import subprocess
import sys
import threading
import textwrap
import time
import termios
from typing import Any, TextIO
import unicodedata

import numpy as np

from flylatro.env.upstream_contract import BLIND_REQ_OFF, BLIND_SCORED_OFF, GLOBAL_ANTE_OFF, GLOBAL_PHASE_OFF, GLOBAL_ROUND, UpstreamActionType
from flylatro.telemetry.actions import describe_action


@dataclass
class FlyProgress:
    fly: int
    decisions: int = 0
    completed_runs: int = 0
    wins: int = 0
    game_overs: int = 0
    run_decisions: int | None = 0
    run_return: float | None = 0.0
    best_return: float | None = None
    best_ante: int = 0
    ante: int = 1
    phase: str = "starting"
    chips: float | None = None
    target_chips: float | None = None
    peak_observed_chips: float | None = None
    best_terminal_score: float | None = None
    round: int | None = None
    highest_round: int | None = None
    run_highest_round: int | None = None
    run_skipped_rounds: int | None = 0
    round_sum: int = 0
    round_samples: int = 0
    round_history_start_run: int = 0
    round_history: list[dict] = field(default_factory=list)
    recent_decisions: list[dict] = field(default_factory=list)
    recent_activity: list[dict] = field(default_factory=list)
    last_beaten_ante: int = 0
    last_beaten_round: int | None = None


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


class TrainingMonitor:
    """Observe environment results without changing RNG, learning or actions."""
    def __init__(self, trainer: Any, saved: dict | None = None):
        self.flies = [FlyProgress(fly=index + 1, decisions=trainer.state.vector_steps)
            for index in range(trainer.env.num_envs)]
        self.environment_decisions = trainer.state.environment_decisions
        self.history_start_decision = self.environment_decisions
        self.episode_history_complete = self.environment_decisions == 0
        saved_matches = bool(saved and saved.get("version") == "training-monitor-v1"
                and saved.get("environment_decisions") == self.environment_decisions
                and len(saved.get("flies", [])) == len(self.flies))
        if saved_matches:
            self.flies = [FlyProgress(**row) for row in saved["flies"]]
            for fly, row in zip(self.flies, saved["flies"], strict=True):
                if "round_samples" not in row:
                    fly.round_history_start_run = fly.completed_runs
                if "run_skipped_rounds" not in row:
                    fly.run_skipped_rounds = 0 if fly.run_decisions == 0 else None
            for fly in self.flies:
                if not fly.recent_activity:
                    fly.recent_activity = _activity_history(asdict(fly))
            self.history_start_decision = saved["history_start_decision"]
            self.episode_history_complete = saved["episode_history_complete"]
        elif len(self.flies) == 1 and self.environment_decisions:
            fly = self.flies[0]
            fly.completed_runs = trainer.state.completed_episodes
            fly.wins = trainer.state.wins
            fly.game_overs = fly.completed_runs - fly.wins
            fly.best_return = getattr(trainer.state, 'best_episode_return',
                max(trainer.state.episode_returns, default=None))
            fly.best_ante = getattr(trainer.state, 'best_episode_ante',
                max(trainer.state.episode_antes, default=0))
            fly.run_decisions = fly.run_return = None
            self.episode_history_complete = True
        elif self.environment_decisions:
            for fly in self.flies:
                fly.run_decisions = fly.run_return = None
        if not saved_matches:
            for fly in self.flies:
                fly.round_history_start_run = fly.completed_runs
                if self.environment_decisions:
                    fly.run_skipped_rounds = None
        self.observe(trainer.observations)

    def observe(self, observations: dict) -> None:
        phases = ("blind select", "hand", "cash out", "shop", "pack", "terminal")
        for row, fly in enumerate(self.flies):
            global_values = observations["global"][row]
            hot = global_values[GLOBAL_ANTE_OFF:GLOBAL_ANTE_OFF + 9]
            fly.ante = int(np.argmax(hot)) + 1 if np.any(hot) else 0
            fly.best_ante = max(fly.best_ante, fly.ante)
            phase = global_values[GLOBAL_PHASE_OFF:GLOBAL_PHASE_OFF + 6]
            fly.phase = phases[int(np.argmax(phase))] if np.any(phase) else "unknown"
            fly.round = _round_number(global_values[GLOBAL_ROUND])
            if fly.round is not None:
                fly.highest_round = max(fly.highest_round or 0, fly.round)
                fly.run_highest_round = max(fly.run_highest_round or 0, fly.round)
            encoded = observations["blind"][row]
            with np.errstate(over="ignore", invalid="ignore"):
                fly.chips = finite_number(np.expm1(float(encoded[BLIND_SCORED_OFF])))
                fly.target_chips = finite_number(np.expm1(float(encoded[BLIND_REQ_OFF])))
            if fly.chips is not None:
                fly.peak_observed_chips = max(fly.peak_observed_chips or 0.0, fly.chips)
        # Small CPU observation tables only; copy before the next step can
        # replace/reorder cards or mutate a reusable environment buffer.
        self._action_context = {key: observations[key].copy() for key in
            ("hand", "joker_ids", "consumable_ids", "shop_ids", "blind", "global") if key in observations}

    def update(self, trainer: Any) -> list[dict]:
        step = trainer.last_step
        if step is None:
            return []
        episodes = []
        for row, fly in enumerate(self.flies):
            fly.decisions += 1
            actions = getattr(trainer, "last_executed_actions", None)
            if actions is not None:
                record = {"kind": "decision", "decision": fly.decisions, "run": fly.completed_runs + 1,
                    "action_type": int(actions["action_type"][row]),
                    "description": describe_action(actions, self._action_context, row),
                    "reward": finite_number(step.rewards[row])}
                fly.recent_decisions.append(record)
                fly.recent_activity.append(record)
                del fly.recent_decisions[:-8]
                # Accepted SKIP_BLIND actions are observable decisions. The
                # simulator's round counter excludes them; track separately.
                if record['action_type'] == UpstreamActionType.SKIP_BLIND and self._action_context['global'][row, GLOBAL_PHASE_OFF]:
                    if fly.run_skipped_rounds is not None:
                        fly.run_skipped_rounds += 1
            self._record_game_events(fly, step, row)
            del fly.recent_activity[:-256]
            if fly.run_decisions is not None:
                fly.run_decisions += 1
            if fly.run_return is not None:
                fly.run_return += float(step.rewards[row])
            if not step.dones[row]:
                continue
            episode = step.infos[row].get("episode", {})
            won = bool(episode.get("won", False))
            fly.completed_runs += 1
            fly.wins += int(won)
            fly.game_overs += int(not won)
            reward = finite_number(episode.get("r", fly.run_return))
            if reward is not None:
                fly.best_return = reward if fly.best_return is None else max(fly.best_return, reward)
            # Terminal observations have already auto-reset. The pre-step
            # observation belongs to the completed run; explicit terminal
            # round info, when supplied, may additionally describe that run.
            terminal_round = finite_number(episode.get("round"))
            rounds_reached = fly.run_highest_round
            if terminal_round is not None and terminal_round >= 0:
                rounds_reached = max(rounds_reached or 0, int(round(terminal_round)))
            if rounds_reached is not None:
                fly.highest_round = max(fly.highest_round or 0, rounds_reached)
                fly.round_sum += rounds_reached
                fly.round_samples += 1
                fly.round_history.append({"run": fly.completed_runs, "round": rounds_reached,
                    "skipped_rounds": fly.run_skipped_rounds, "return": reward, "won": won})
                del fly.round_history[:-64]
            fly.best_ante = max(fly.best_ante, int(episode.get("ante", 0)))
            score = finite_number(episode.get("score"))
            if score is not None:
                fly.best_terminal_score = score if fly.best_terminal_score is None else max(fly.best_terminal_score, score)
            episodes.append({"fly": fly.fly, "environment_decisions": trainer.state.environment_decisions,
                "fly_decisions": fly.decisions, "run": fly.completed_runs, "won": won,
                "ante": episode.get("ante"), "return": reward, "terminal_score": score,
                "round": rounds_reached,
                "skipped_rounds": fly.run_skipped_rounds,
                "decisions": episode.get("l", fly.run_decisions)})
            fly.run_decisions = 0
            fly.run_return = 0.0
            fly.last_beaten_ante = 0
            fly.last_beaten_round = None
            fly.run_highest_round = None
            fly.run_skipped_rounds = 0
        self.environment_decisions = trainer.state.environment_decisions
        # Auto-reset observations belong to the next run; account for the
        # completed episode first, then display the current environment state.
        self.observe(trainer.observations)
        return episodes

    def _record_game_events(self, fly: FlyProgress, step: Any, row: int) -> None:
        """Milestones are timeline events, never additional decisions."""
        info = step.infos[row]
        done = bool(step.dones[row])
        won = done and bool(info.get("episode", {}).get("won", False))
        before_global = self._action_context["global"][row]
        after_global = step.observations["global"][row]
        after_ante_hot = after_global[GLOBAL_ANTE_OFF:GLOBAL_ANTE_OFF + 9]
        after_ante = int(np.argmax(after_ante_hot)) + 1 if np.any(after_ante_hot) else 0
        entered_cash_out = not done and bool(before_global[GLOBAL_PHASE_OFF + 1]) and bool(after_global[GLOBAL_PHASE_OFF + 2])
        components = info.get("reward_components", {})
        round_beaten = won or entered_cash_out or bool(info.get("blind_cleared") or info.get("round_cleared")) or float(components.get("blind_clear", 0)) > 0
        blind = self._action_context["blind"][row, :3]
        blind_name = ("Small blind", "Big blind", "Boss blind")[int(np.argmax(blind))] if np.any(blind) else None
        ante_beaten = won or bool(info.get("ante_cleared")) or float(components.get("ante_clear", 0)) > 0 or (
            not done and after_ante > fly.ante > 0) or (round_beaten and blind_name == "Boss blind")

        def event(kind: str, description: str) -> None:
            fly.recent_activity.append({"kind": "event", "event": kind, "description": description,
                "run": fly.completed_runs + 1, "ante": fly.ante, "round": fly.round,
                "after_decision": fly.decisions})

        if round_beaten:
            detail = f"{blind_name} / " if blind_name else ""
            event("round_beaten", f"ROUND BEATEN / {_number(fly.round)} / {detail}Ante {fly.ante}")
        if ante_beaten and (fly.ante, fly.round) != (fly.last_beaten_ante, fly.last_beaten_round):
            event("ante_beaten", f"ANTE BEATEN / Ante {fly.ante}")
            fly.last_beaten_ante = fly.ante
            fly.last_beaten_round = fly.round
        if done:
            event("run_won" if won else "game_over", "RUN WON / target reached" if won else
                f"GAME OVER / Round {_number(fly.round)} / Ante {fly.ante}")

    def state_dict(self) -> dict:
        return {"version": "training-monitor-v1", "environment_decisions": self.environment_decisions,
            "history_start_decision": self.history_start_decision,
            "episode_history_complete": self.episode_history_complete,
            "flies": [asdict(fly) for fly in self.flies]}


GPU_QUERY = ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
    "--format=csv,noheader,nounits"]


def sample_gpus() -> dict:
    try:
        result = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=0.8, check=False)
        if result.returncode:
            return {"status": "unavailable", "gpus": []}
        devices = []
        for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            if len(row) != 9:
                continue
            devices.append({"index": row[0], "uuid": row[1], "name": row[2],
                **{key: finite_number(value) for key, value in zip(
                    ("utilization_percent", "memory_utilization_percent", "used_mib", "total_mib", "temperature_c", "power_w"), row[3:], strict=True)}})
        return {"status": "available" if devices else "unavailable", "gpus": devices,
            "sampled_at": time.time(), "scope": "whole GPU, including other processes"}
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "unavailable", "gpus": []}


def _number(value: Any, digits: int = 0) -> str:
    number = finite_number(value)
    return "--" if number is None else f"{number:,.{digits}f}"


def _round_number(encoded: Any) -> int | None:
    """Pinned GLOBAL_ROUND is the actual round counter divided by 24."""
    value = finite_number(encoded)
    return int(round(value * 24)) if value is not None and value >= 0 else None


def _average_round(fly: dict) -> float | None:
    count = fly.get("round_samples", 0)
    return fly.get("round_sum", 0) / count if count else None


def _duration(seconds: float | None) -> str:
    value = finite_number(seconds)
    if value is None:
        return "--"
    seconds = max(0, int(value))
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def _activity_history(fly: dict) -> list[dict]:
    """Read older monitoring checkpoints without hiding their terminal events."""
    if fly.get("recent_activity"):
        return fly["recent_activity"]
    activity = []
    for entry in fly.get("recent_decisions", []):
        activity.append({**entry, "kind": "decision"})
        if entry.get("outcome"):
            won = entry["outcome"] == "win"
            activity.append({"kind": "event", "event": "run_won" if won else "game_over",
                "description": "RUN WON / target reached" if won else "GAME OVER",
                "run": entry["run"], "after_decision": entry["decision"]})
    return activity


def _render_plain_dashboard(snapshot: dict, *, width: int = 110, height: int = 30, now: float = 0) -> str:
    width = max(30, width)
    metrics = snapshot.get("metrics", {})
    monitoring = snapshot.get("monitoring", {})
    flies = monitoring.get("flies", [])
    decisions = snapshot.get("decisions", 0)
    budget = snapshot.get("target_decisions", 0)
    rate = finite_number(metrics.get("runtime/environment_decisions_per_second")) or 0.0
    eta = (budget - decisions) / rate if rate and budget > decisions else 0 if decisions >= budget else None
    bar_width = min(35, width - 18)
    fill = int(bar_width * min(decisions / max(budget, 1), 1))
    lines = [f"FLYLATRO TRAINING | {snapshot.get('status', 'starting').upper()}",
        str(snapshot.get("run_name", "")),
        f"[{('#' * fill).ljust(bar_width, '-')}] {_number(decisions)}/{_number(budget)} decisions",
        f"Speed {_number(rate, 2)} dec/s | Elapsed {_duration(metrics.get('runtime/wall_seconds'))} | ETA {_duration(eta)}",
        f"Completed runs {_number(snapshot.get('completed_runs', 0))} | Wins {_number(snapshot.get('wins', 0))} | Game overs {_number(snapshot.get('game_overs', 0))}",
        "-" * width]
    setup = snapshot.get("game_setup", {})
    if setup:
        goal = f"Ante {setup['target_ante']}" if setup.get("target_ante") is not None else "clear mock blind"
        lines.insert(2, f"{setup['deck']} / {setup['stake']} | Win target: {goal} | New seed each run")
    gpu = snapshot.get("gpu", {})
    for device in gpu.get("gpus", []):
        lines += [f"GPU {device['index']} {device['name']} | Util {_number(device['utilization_percent'])}% | VRAM {_number(device['used_mib'])}/{_number(device['total_mib'])} MiB",
            f"Memory activity {_number(device['memory_utilization_percent'])}% | {_number(device['temperature_c'])} C | {_number(device['power_w'], 1)} W (whole GPU)"]
    if not gpu.get("gpus"):
        lines.append("GPU: CPU execution" if snapshot.get("device", "cpu").startswith("cpu") else "GPU: telemetry unavailable / waiting for sample")
    lines.append(f"This process: Torch {_number(metrics.get('runtime/gpu_allocated_mb'))} MiB | Peak {_number(metrics.get('runtime/gpu_peak_allocated_mb'))} MiB | Peak RSS {_number(metrics.get('runtime/process_peak_rss_mb'))} MiB")
    bound = finite_number(metrics.get("plasticity/max_learner_bound_fraction"))
    lines.append(f"Plasticity: drift {_number(metrics.get('plasticity/mean_absolute_drift'), 4)} | bound occupancy {_number(100 * bound if bound is not None else None, 2)}% | updates {_number(metrics.get('plasticity/changed_event_count'))}")
    lines.append("-" * width)
    page_size = max(1, (height - len(lines) - 5) // ((3 if width >= 100 else 5) + 3))
    pages = max(1, math.ceil(len(flies) / page_size))
    page = int(now // 8) % pages
    shown = flies[page * page_size:(page + 1) * page_size]
    for fly in shown:
        lines += [f"Fly {fly['fly']:02} | Run {fly['completed_runs'] + 1} | Done {fly['completed_runs']} | Wins {fly['wins']} | Game overs {fly['game_overs']} | Decisions {fly['decisions']}",
            f"  Ante {fly['ante']} (best {fly['best_ante']}) | {fly['phase']} | Blind chips ~{_number(fly['chips'])}/~{_number(fly['target_chips'])} | Highest blind chips ~{_number(fly['peak_observed_chips'])}",
            f"  Round {_number(fly.get('round'))} | Highest round {_number(fly.get('highest_round'))} | Average completed run {_number(_average_round(fly), 2)} ({fly.get('round_samples', 0)} runs)",
            f"  Run decisions {_number(fly['run_decisions'])} | Training reward {_number(fly['run_return'], 3)} | Best reward {_number(fly['best_return'], 3)} | Final game score {_number(fly['best_terminal_score'])}"]
        lines += ["  " + line for line in _round_bars(fly, width=width - 2, rows=4, colour=False, unicode=False)]
        for entry in reversed(_activity_history(fly)[-3:]):
            if entry["kind"] == "event":
                lines.append(f"  *** {entry['description']} [run {entry['run']}] ***")
            else:
                lines.append(f"  #{entry['decision']} [run {entry['run']}] {entry['description']}")
    if pages > 1:
        lines.append(f"Fly page {page + 1}/{pages}; rotates every 8 seconds")
    lines += ["Highest blind chips = best observed blind total (decoded log1p); unavailable final scores show --."]
    lines.append("Training reward = accumulated learning signal; round average/graph use completed runs.")
    if monitoring.get("history_start_decision", 0):
        lines.append(f"Chip history begins at decision {monitoring['history_start_decision']}." +
            (" Per-fly run counters begin there too." if not monitoring.get("episode_history_complete", True) else ""))
    lines.append(f"Run directory: {snapshot.get('run_dir', '')}")
    return "\n".join(part for line in lines for part in textwrap.wrap(
        "".join(char for char in line if char >= " "), width=width, subsequent_indent="  "))


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_COLOURS = {"cyan": 117, "violet": 141, "green": 150, "gold": 221, "red": 203,
    "muted": 245, "white": 255}


def _clean(value: Any) -> str:
    return "".join(char for char in _ANSI.sub("", str(value))
        if not unicodedata.category(char).startswith("C"))


def _char_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def _visible_width(value: str) -> int:
    return sum(_char_width(char) for char in _ANSI.sub("", value))


def _fit(value: str, width: int, *, unicode: bool) -> str:
    """Clip styled text by terminal cells, retaining complete ANSI sequences."""
    width = max(0, width)
    if not unicode:
        value = value.encode("ascii", errors="replace").decode("ascii")
    if _visible_width(value) <= width:
        return value + " " * (width - _visible_width(value))
    suffix = "…" if unicode else "." * min(3, width)
    available = width - len(suffix)
    parts = []
    used = 0
    for token in re.split(r"(\x1b\[[0-9;]*m)", value):
        if _ANSI.fullmatch(token):
            parts.append(token)
            continue
        for char in token:
            cells = _char_width(char)
            if used + cells > available:
                return "".join(parts) + ("\x1b[0m" if "\x1b" in value else "") + " " * (available - used) + suffix
            parts.append(char)
            used += cells
    return "".join(parts)


def _paint(value: str, tone: str, colour: bool, *, bold: bool = False) -> str:
    if not colour:
        return value
    return f"\x1b[{('1;' if bold else '')}38;5;{_COLOURS[tone]}m{value}\x1b[0m"


def _meter(value: Any, maximum: Any, cells: int, *, tone: str, colour: bool, unicode: bool) -> str:
    value, maximum = finite_number(value), finite_number(maximum)
    if value is None or maximum is None or maximum <= 0:
        return _paint(("┄" if unicode else "-") * cells, "muted", colour)
    filled = round(cells * max(0.0, min(value / maximum, 1.0)))
    return _paint(("━" if unicode else "#") * filled, tone, colour) + _paint(
        ("─" if unicode else "-") * (cells - filled), "muted", colour)


def _round_bars(fly: dict, *, width: int, rows: int, colour: bool, unicode: bool, scale: int | None = None) -> list[str]:
    """Directly labelled run categories, chronological order and zero baseline."""
    history = fly.get("round_history", [])
    if not history:
        return [_paint("Recent runs: waiting for a completed run", "muted", colour)]
    high = max(1, scale or 0, max(entry['round'] + (entry.get('skipped_rounds') or 0) for entry in history))
    shown = history[-max(1, rows):]
    label_width = max(len(f"Run {entry['run']}") for entry in shown)
    outcomes = [f"{entry['round']} played + {entry.get('skipped_rounds') if entry.get('skipped_rounds') is not None else '?'} skipped" for entry in shown]
    cells = max(2, min(22, width - label_width - max(map(len, outcomes)) - 4))
    played_glyph, skipped_glyph, empty_glyph = ("█", "▓", "░") if unicode else ("#", "+", ".")
    legend = _paint(played_glyph + " played", 'cyan', colour) + " " + _paint(skipped_glyph + " skipped", 'gold', colour)
    result = [f"RECENT RUNS / {legend} / scale 0 to {high}"]
    for entry, outcome in zip(shown, outcomes, strict=True):
        value = entry['round']
        filled = round(cells * value / high)
        skipped = round(cells * (value + (entry.get('skipped_rounds') or 0)) / high) - filled
        bar = _paint(played_glyph * filled, "cyan", colour) + _paint(skipped_glyph * skipped, 'gold', colour) + _paint(
            empty_glyph * (cells - filled - skipped), "muted", colour)
        result.append(f"{('Run ' + str(entry['run'])).ljust(label_width)}  {bar}  {outcome}")
    return result


_METRIC_HELP = [
    "Round: simulator round counter; starts at 0 before selecting the first blind.",
    "Highest round: furthest round observed, including the current run.",
    "Average round: mean highest round reached in completed, tracked runs.",
    "Bars: played rounds plus skipped blinds, oldest completed run at the top.",
    "Bar lengths start at zero and share a scale across flies for fair comparison.",
    "Skipped blinds do not count as played rounds; Antes and rounds differ.",
    "Skipped blinds use a separate bar segment and count, never rounds beaten.",
    "Legacy skip counts are unknown (?); fresh runs record them completely.",
    "Blind chips: observed chips scored so far / chips required to beat the blind.",
    "Highest blind chips: largest observed blind total across tracked runs.",
    "Training reward: accumulated environment learning signal, not game chips.",
    "Best reward: highest total training reward from a completed run.",
    "Final game score: unavailable when the simulator does not report it (--).",
    "Wins: runs that reached the configured Ante target (shown in game setup).",
    "Weight change: mean absolute synaptic change; guard shows the worst fly.",
    "Bounds: fraction of plastic weights at a limit; guard shows the worst fly.",
    "GPU readings cover the whole device; Torch memory belongs to this process.",
    "Peak RAM: maximum resident process memory observed by the operating system.",
    "Feed: submitted card/item choices; coloured milestones add no decisions.",
    "Up/Down or mouse wheel: browse feed. PgUp/PgDn: jump five entries.",
    "Tab or Left/Right: choose fly. Home: oldest. End or L: live feed.",
    "Press H again to return. Ctrl+C stops training and saves a checkpoint.",
]


def _box(title: str, body: list[str], width: int, *, tone: str, colour: bool, unicode: bool) -> list[str]:
    top_left, top_right, bottom_left, bottom_right, vertical, horizontal = (
        ("╭", "╮", "╰", "╯", "│", "─") if unicode else ("+", "+", "+", "+", "|", "-"))
    label = _fit(f"{horizontal} {_clean(title)} ", width - 2, unicode=unicode).rstrip()
    top = _paint(top_left + label + horizontal * (width - 2 - _visible_width(label)) + top_right, tone, colour)
    side = _paint(vertical, tone, colour)
    return [top, *[side + " " + _fit(row, width - 4, unicode=unicode) + " " + side for row in body],
        _paint(bottom_left + horizontal * (width - 2) + bottom_right, tone, colour)]


def _beside(left: list[str], right: list[str], width: int, *, unicode: bool) -> list[str]:
    return [_fit(left[index] if index < len(left) else "", width, unicode=unicode) + "  " +
        (right[index] if index < len(right) else "") for index in range(max(len(left), len(right)))]


def _health_tone(value: Any, maximum: Any) -> str:
    value, maximum = finite_number(value), finite_number(maximum)
    if value is None or maximum is None or maximum <= 0:
        return "muted"
    return "red" if value >= maximum else "gold" if value >= maximum * 0.75 else "green"


def _render_flight_deck(snapshot: dict, *, width: int, height: int, now: float,
                        colour: bool, unicode: bool) -> str:
    width, height = max(20, width), max(8, height)
    metrics = snapshot.get("metrics", {})
    monitoring = snapshot.get("monitoring", {})
    flies = monitoring.get("flies", [])
    dense = height < 24
    paint = lambda value, tone="white", bold=False: _paint(str(value), tone, colour, bold=bold)
    meter = lambda value, limit, size=12, tone="cyan": _meter(value, limit, size,
        tone=tone, colour=colour, unicode=unicode)
    separator = " · " if unicode else " | "
    status = _clean(snapshot.get("status", "starting")).upper()
    active = status in ("TRAINING", "INITIALISING", "STARTING")
    status_tone = "cyan" if active else "green" if status in ("COMPLETED", "STAGE COMPLETE") else "gold" if status == "INTERRUPTED" else "red"
    spinner = "|/-\\"[int(now * 2) % 4] if active else "●" if unicode else "*"
    badge = paint(f"{spinner} {status}", status_tone, True)
    heading = paint("FLYLATRO", "cyan", True) + paint(" / TRAINING DASHBOARD", "muted")
    if width < _visible_width(heading) + _visible_width(badge) + 2:
        heading = paint("FLYLATRO", "cyan", True)
    gap = width - _visible_width(heading) - _visible_width(badge)
    lines = [heading + " " * max(2, gap) + badge]
    setup = snapshot.get("game_setup", {})
    if setup:
        goal = f"Ante {setup['target_ante']}" if setup.get("target_ante") is not None else "clear mock blind"
        lines.append(paint(f"{_clean(setup.get('deck', ''))} / {_clean(setup.get('stake', ''))}" +
            f" / Win target: {goal} / new seed each run", "white"))
    name = paint("Experiment: " + _clean(snapshot.get("run_name", "")), "muted")
    if width >= 100 or height >= 32:
        lines.append(_fit(name, width, unicode=unicode))
    if snapshot.get("show_help"):
        help_rows = [part for message in _METRIC_HELP for part in textwrap.wrap(message, width=max(16, width - 4))]
        offset = min(snapshot.get("help_offset", 0), max(0, len(help_rows) - height + len(lines) + 3))
        rows = help_rows[offset:offset + max(1, height - len(lines) - 3)]
        lines += _box("METRICS / CONTROLS", rows, width, tone="violet", colour=colour, unicode=unicode)
        lines.append(paint("Up/Down: scroll help / H: return / Ctrl+C: stop training", "muted"))
        return "\n".join(_fit(line, width, unicode=unicode) for line in lines[:height])

    decisions = finite_number(snapshot.get("decisions")) or 0
    target = finite_number(snapshot.get("target_decisions")) or 0
    rate = finite_number(metrics.get("runtime/environment_decisions_per_second")) or 0.0
    eta = (target - decisions) / rate if rate and target > decisions else 0 if target > 0 and decisions >= target else None
    count = f"{_number(decisions)}/{_number(target)} decisions"
    ratio = decisions / target * 100 if target > 0 else 0
    progress_width = min(30, max(4, width - len(count) - 18))
    overview = [meter(decisions, target, progress_width) + "  " + paint(count, "white", True) + "  " + paint(f"{ratio:.1f}%", "cyan"),
        separator.join((paint(f"Speed {_number(rate, 2)} decisions/s", "cyan"),
            f"Elapsed {_duration(metrics.get('runtime/wall_seconds'))}", f"Time left {_duration(eta)}")),
        separator.join((f"Completed runs {_number(snapshot.get('completed_runs', 0))}",
            paint(f"Wins {_number(snapshot.get('wins', 0))}", "green"),
            paint(f"Game overs {_number(snapshot.get('game_overs', 0))}", "gold")))]
    if dense:
        # Keep room for the current fly in a short viewport.
        lines.append(_fit(overview.pop(), width, unicode=unicode))
    lines += _box("TRAINING PROGRESS", overview, width, tone="cyan", colour=colour, unicode=unicode)

    gpu_devices = snapshot.get("gpu", {}).get("gpus", [])
    device = gpu_devices[int(now // 8) % len(gpu_devices)] if gpu_devices else None
    panel_width = (width - 2) // 2 if width >= 100 else width
    if device:
        gpu_title = f"GPU {device['index']} / {_clean(device['name'])}"
        used, total = device['used_mib'], device['total_mib']
        gpu_rows = [f"GPU load {meter(device['utilization_percent'], 100, 10)} {paint(_number(device['utilization_percent']) + '%', 'cyan')}",
            f"GPU memory {_number(used)} of {_number(total)} MiB",
            separator.join((f"Temp {_number(device['temperature_c'])} C", f"Power {_number(device['power_w'], 1)} W")),
            f"Training memory {_number(metrics.get('runtime/gpu_allocated_mb'))} MiB (Torch)"]
        if height < 36:
            gpu_rows = gpu_rows[:3]
        if width < 100:
            gpu_rows = [separator.join((f"Compute {meter(device['utilization_percent'], 100, 8)} {_number(device['utilization_percent'])}%",
                f"VRAM {_number(used)}/{_number(total)} MiB")),
                separator.join((f"{_number(device['temperature_c'])} C", f"{_number(device['power_w'], 1)} W",
                    f"Mem {_number(device['memory_utilization_percent'])}%", f"Torch {_number(metrics.get('runtime/gpu_allocated_mb'))} MiB"))]
        if dense or (width < 100 and height < 32):
            gpu_rows = [f"Util {_number(device['utilization_percent'])}% / VRAM {_number(used)}/{_number(total)} MiB / {_number(device['temperature_c'])} C"]
    else:
        gpu_title = "COMPUTE / CPU" if str(snapshot.get("device", "cpu")).startswith("cpu") else "GPU TELEMETRY"
        gpu_rows = [paint("CPU execution" if "CPU" in gpu_title else "Waiting for telemetry / unavailable", "muted"),
            f"Peak RAM {_number(metrics.get('runtime/process_peak_rss_mb'))} MiB"]
        if dense or (width < 100 and height < 32):
            gpu_rows = gpu_rows[:1]
    thresholds = snapshot.get("safety_thresholds", {})
    drift = metrics.get("plasticity/mean_absolute_drift")
    drift_limit = thresholds.get("maximum_mean_absolute_drift")
    # The guard uses the worst learner. Show and colour the same quantity.
    worst_drift = metrics.get("plasticity/max_learner_mean_absolute_drift", drift)
    bound = finite_number(metrics.get("plasticity/max_learner_bound_fraction"))
    bound_limit = finite_number(thresholds.get("maximum_bound_fraction"))
    bound_pct = None if bound is None else bound * 100
    limit_pct = None if bound_limit is None else bound_limit * 100
    drift_tone, bound_tone = _health_tone(worst_drift, drift_limit), _health_tone(bound, bound_limit)
    health_rows = [f"Weight change {paint(_number(worst_drift, 4), drift_tone)} (limit {_number(drift_limit, 2)})",
        f"At weight bounds {paint(_number(bound_pct, 2) + '%', bound_tone)} (limit {_number(limit_pct, 0)}%)",
        f"Synapse update events {paint(_number(metrics.get('plasticity/changed_event_count')), 'white')}",
        f"Peak RAM {_number(metrics.get('runtime/process_peak_rss_mb'))} MiB"]
    if height < 36:
        health_rows = health_rows[:3]
    if width < 100:
        health_rows = [separator.join((f"Weight change {paint(_number(worst_drift, 4), drift_tone)} (limit {_number(drift_limit, 2)})",
            f"Bounds {paint(_number(bound_pct, 2) + '%', bound_tone)}/{_number(limit_pct, 0)}%",
            f"Updates {_number(metrics.get('plasticity/changed_event_count'))}")),
            f"Worst learner / configured limits {meter(worst_drift, drift_limit, 16, drift_tone)}"]
    if dense or (width < 100 and height < 32):
        health_rows = health_rows[:1]
    gpu_panel = _box(gpu_title, gpu_rows, panel_width, tone="violet", colour=colour, unicode=unicode)
    health_panel = _box("LEARNING LIMITS / WORST FLY", health_rows, panel_width if width >= 100 else width,
        tone="violet", colour=colour, unicode=unicode)
    if width < 100 and height < 30:
        brief_gpu_title = f"GPU {device['index']}" if device else gpu_title
        brief_rows = [f"{brief_gpu_title} / {gpu_rows[0]}", health_rows[0]]
        lines += brief_rows if height < 24 else _box("HARDWARE / LEARNING LIMITS", brief_rows, width,
            tone="violet", colour=colour, unicode=unicode)
    else:
        lines += _beside(gpu_panel, health_panel, panel_width, unicode=unicode) if width >= 100 else gpu_panel + health_panel

    columns = 2 if width >= 100 and len(flies) > 1 else 1
    fly_width = (width - 2) // 2 if columns == 2 else width
    expanded = height >= 36
    show_score = any(fly.get("best_terminal_score") is not None for fly in flies)
    stat_rows = (8 + int(show_score)) if expanded else 4
    base_card_height = stat_rows + 3  # stats, feed heading and two borders
    footer_height = 4 if monitoring.get("history_start_decision", 0) else 3
    available = height - len(lines) - footer_height
    desired_feed = 3 if width >= 100 else 1
    bar_rows = min(4 if expanded else 3, max(1, available - base_card_height - desired_feed - 1))
    graph_lines = bar_rows + 1 if any(fly.get('round_history') for fly in flies) else 1
    history_slots = min(5 if expanded else 3, max(1, available - base_card_height - graph_lines))
    card_height = base_card_height + history_slots + graph_lines
    round_scale = max([1, *[entry['round'] + (entry.get('skipped_rounds') or 0) for fly in flies for entry in fly.get('round_history', [])]])
    card_rows = max(1, (height - len(lines) - footer_height) // card_height)
    page_size = card_rows * columns
    pages = max(1, math.ceil(len(flies) / page_size))
    page = int(now // 8) % pages
    selected = snapshot.get("feed_fly")
    if selected is not None:
        index = next((index for index, fly in enumerate(flies) if fly["fly"] == selected), 0)
        page = index // page_size
    shown = flies[page * page_size:(page + 1) * page_size]
    cards = []
    for fly in shown:
        fly_rows = [f"Run {fly['completed_runs'] + 1}" + separator + f"Completed {fly['completed_runs']}" + separator +
                paint(f"Wins {fly['wins']}", 'green') + separator + f"Game overs {fly['game_overs']}",
            f"Round {_number(fly.get('round'))} / highest {_number(fly.get('highest_round'))}" +
                f" / average {_number(_average_round(fly), 2)}" + separator + f"Ante {fly['ante']}",
            f"Decisions {_number(fly['decisions'])} ({_number(fly['run_decisions'])} this run)" + separator +
                f"Reward {_number(fly['run_return'], 2)}",
            f"Blind chips ~{_number(fly['chips'])} of ~{_number(fly['target_chips'])}" +
                f" (highest ~{_number(fly['peak_observed_chips'])})"]
        if expanded:
            fly_rows = [f"Current run: {fly['completed_runs'] + 1}" + separator + f"Completed: {fly['completed_runs']}",
                paint(f"Wins: {fly['wins']}", 'green') + separator + f"Game overs: {fly['game_overs']}",
                f"Ante: {fly['ante']} (highest {fly['best_ante']})" + separator +
                    f"Round: {paint(_number(fly.get('round')), 'cyan', True)}" + separator +
                    f"Skipped: {_number(fly.get('run_skipped_rounds'))}",
                f"Highest round: {_number(fly.get('highest_round'))}" +
                    f"  Average: {_number(_average_round(fly), 2)} ({fly.get('round_samples', 0)} runs)",
                f"Decisions: {_number(fly['decisions'])} total" + separator + f"{_number(fly['run_decisions'])} this run",
                f"Blind chips: ~{_number(fly['chips'])} of ~{_number(fly['target_chips'])} " + meter(fly['chips'], fly['target_chips'], 8),
                f"Highest blind chips: ~{_number(fly['peak_observed_chips'])}",
                f"Training reward: {_number(fly['run_return'], 3)}" + separator + f"Best: {_number(fly['best_return'], 3)}"]
            if show_score:
                fly_rows.append(f"Highest final score: {_number(fly['best_terminal_score'])}")
        graph = _round_bars(fly, width=fly_width - 4, rows=bar_rows, colour=colour, unicode=unicode, scale=round_scale)
        fly_rows += graph + [""] * (graph_lines - len(graph))
        activity = snapshot.get("feed_history", []) if fly['fly'] == selected and snapshot.get("feed_offset", 0) else _activity_history(fly)
        offset = min(snapshot.get("feed_offset", 0), max(0, len(activity) - 1)) if fly['fly'] == selected else 0
        feed_mode = f"HISTORY / {offset} older" if offset else "LIVE / newest first"
        fly_rows.append(paint("ACTION FEED / " + feed_mode, "violet" if offset else "white"))
        recent = list(reversed(activity[:len(activity) - offset][-history_slots:]))
        for index, entry in enumerate(recent):
            if entry["kind"] == "event":
                marker = "◆" if unicode else "!"
                tone = "red" if entry["event"] == "game_over" else "gold" if entry["event"] == "ante_beaten" else "green"
                fly_rows.append(paint(f"{marker} {_clean(entry['description'])} [r{entry['run']}]", tone, True))
            else:
                fly_rows.append(paint(f"#{entry['decision']} [r{entry['run']}] {_clean(entry['description'])}", "white" if index == 0 else "muted"))
        if not recent:
            fly_rows.append(paint("Waiting for recorded actions...", "muted"))
        fly_rows += [""] * (history_slots - max(1, len(recent)))
        title = f"Fly {fly['fly']:02} / {_clean(fly['phase']).upper()}" + (" / SELECTED" if selected == fly['fly'] else "")
        cards.append(_box(title, fly_rows, fly_width,
            tone="violet" if selected == fly['fly'] else "cyan", colour=colour, unicode=unicode))
    for index in range(0, len(cards), columns):
        lines += _beside(cards[index], cards[index + 1] if index + 1 < len(cards) else [], fly_width,
            unicode=unicode) if columns == 2 else cards[index]
    if not flies:
        lines.append(paint("Preparing the fly / counters arrive with the first observation", "muted"))
    note = "Reward = learning signal; average round = completed runs; -- = unavailable; H = help"
    if not show_score:
        note = "GPU = whole device; reward = learning signal; final scores unavailable; H = help"
    if pages > 1:
        note = f"Fly page {page + 1}/{pages}" + (" / selected fly" if selected is not None else "; rotates every 8 seconds") + " / H: help / whole GPU"
    for fly in shown:
        if fly.get("round_history_start_run", 0):
            note = f"Fly {fly['fly']:02}: round history from run {fly['round_history_start_run'] + 1} / H: help"
            break
    lines.append(paint(note, "muted"))
    lines.append(paint("Feed: wheel or Up/Down / PgUp/PgDn / Tab: fly / End: live" if snapshot.get("interactive_controls") else
        "Interactive feed controls require terminal input; -- = unavailable", "muted"))
    if monitoring.get("history_start_decision", 0):
        note = f"Chip history from decision {monitoring['history_start_decision']}"
        if not monitoring.get("episode_history_complete", True):
            note += "; per-fly run history starts there too"
        lines.append(paint(note, "muted"))
    lines.append(paint("Run directory: " + _clean(snapshot.get("run_dir", "")), "muted"))
    # Terminal drawing reserves its final row, so a resize cannot scroll the
    # alternate screen. Very small viewports retain the header and footer.
    if len(lines) > height:
        lines = lines[:height - 2] + [paint("Enlarge terminal to see all fly details", "muted"), lines[-1]]
    return "\n".join(_fit(line, width, unicode=unicode) for line in lines)


def render_dashboard(snapshot: dict, *, width: int = 110, height: int = 30, now: float = 0,
                     colour: bool = False, unicode: bool = False) -> str:
    if colour or unicode or snapshot.get("terminal_layout") or snapshot.get("interactive_controls"):
        return _render_flight_deck(snapshot, width=width, height=height, now=now, colour=colour, unicode=unicode)
    return _render_plain_dashboard(snapshot, width=width, height=height, now=now)


class TerminalDashboard:
    def __init__(self, *, mode: str, refresh: float, device: str, run_name: str, run_dir: Path,
                 stream: TextIO | None = None, target_decisions: int = 0, input_stream: TextIO | None = None):
        self.stream = stream if stream is not None else sys.stderr
        self.input_stream = input_stream if input_stream is not None else sys.stdin
        self.tty = bool(self.stream.isatty()) and os.environ.get("TERM") != "dumb"
        self.colour = self.tty and "NO_COLOR" not in os.environ
        try:
            "╭─━▁".encode(self.stream.encoding or "utf-8")
            self.unicode = self.tty
        except (AttributeError, LookupError, UnicodeEncodeError):
            self.unicode = False
        self.enabled = mode == "on" or (mode == "auto" and self.tty)
        self.refresh = refresh
        self.snapshot = {"run_name": run_name, "run_dir": str(run_dir), "device": device,
            "status": "initialising", "target_decisions": target_decisions}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._gpu = {"status": "waiting", "gpus": []}
        self._rates = deque(maxlen=36)
        self._rate_decisions = None
        self._input_fd = None
        self._terminal_attributes = None
        self._pending_input = ""
        self._feed_fly = None
        self._feed_offset = 0
        self._feed_history = []
        self._show_help = False
        self._help_offset = 0
        self._last_frame = ""

    def __enter__(self):
        if self.enabled:
            if self.tty:
                self.stream.write("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H")
                self._start_input()
                self.stream.flush()
            self._thread = threading.Thread(target=self._loop, name="flylatro-dashboard", daemon=True)
            self._thread.start()
        return self

    def update(self, **values):
        with self._lock:
            self.snapshot.update(values)
            decisions = values.get("decisions")
            rate = finite_number(values.get("metrics", {}).get("runtime/environment_decisions_per_second"))
            if decisions is not None and decisions != self._rate_decisions and rate is not None:
                self._rates.append(rate)
                self._rate_decisions = decisions
            self.snapshot["throughput_history"] = list(self._rates)

    def _loop(self):
        next_gpu = 0.0
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if str(self.snapshot["device"]).startswith("cuda") and now >= next_gpu:
                    self._gpu = sample_gpus()
                    next_gpu = now + max(2.0, self.refresh)
                snapshot = self._view_snapshot()
                size = self._terminal_size()
                frame = render_dashboard(snapshot, width=size.columns, height=max(8, size.lines - 1), now=now,
                    colour=self.colour, unicode=self.unicode)
                if self.tty:
                    self.stream.write("\x1b[H" + "\x1b[K\n".join(frame.splitlines()) + "\x1b[K\x1b[J")
                else:
                    self.stream.write("\n" + frame + "\n")
                self.stream.flush()
                self._last_frame = _ANSI.sub("", frame)
                self._wait_for_input(self.refresh)
        except (OSError, ValueError, TypeError):
            # A detached terminal or unavailable driver must not change training.
            self._stop.set()

    def _start_input(self):
        """Use cbreak, preserving ISIG so Ctrl+C still reaches training."""
        try:
            if not self.input_stream.isatty():
                return
            fd = self.input_stream.fileno()
            if os.tcgetpgrp(fd) != os.getpgrp():
                return  # Background runs must not consume a shell's input.
            saved = termios.tcgetattr(fd)
            changed = [*saved[:6], list(saved[6])]
            changed[3] &= ~(termios.ICANON | termios.ECHO)
            changed[6][termios.VMIN], changed[6][termios.VTIME] = 1, 0
            termios.tcsetattr(fd, termios.TCSANOW, changed)
            self._input_fd, self._terminal_attributes = fd, saved
            self.stream.write("\x1b[?1000h\x1b[?1006h")
        except (AttributeError, OSError, ValueError, termios.error):
            return

    def _restore_input(self):
        if self._input_fd is not None:
            try:
                termios.tcsetattr(self._input_fd, termios.TCSANOW, self._terminal_attributes)
            except (OSError, ValueError, termios.error):
                pass
            self._input_fd = None
            self.stream.write("\x1b[?1000l\x1b[?1006l")

    def _view_snapshot(self):
        with self._lock:
            return {**self.snapshot, "gpu": self._gpu, "feed_fly": self._feed_fly,
                "feed_offset": self._feed_offset, "feed_history": self._feed_history,
                "show_help": self._show_help, "help_offset": self._help_offset,
                "interactive_controls": self._input_fd is not None, "terminal_layout": self.tty}

    def _wait_for_input(self, timeout):
        if self._input_fd is None:
            self._stop.wait(timeout)
            return
        deadline = time.monotonic() + timeout
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            ready, _, _ = select.select([self._input_fd], [], [], min(0.1, remaining))
            if ready:
                data = os.read(self._input_fd, 4096)
                if not data:
                    self._stop.wait(remaining)
                    return
                self._consume_input(data.decode("ascii", errors="ignore"))
                return

    def _consume_input(self, data):
        self._pending_input += data
        keys = {"A": "older", "B": "newer", "C": "next", "D": "previous", "H": "oldest", "F": "live",
            "1~": "oldest", "4~": "live", "5~": "page_older", "6~": "page_newer"}
        while self._pending_input:
            if self._pending_input.startswith("\x1b"):
                mouse = re.match(r"\x1b\[<(\d+);(\d+);(\d+)[Mm]", self._pending_input)
                escape = re.match(r"\x1b[\[O]([ABCDHF]|[1456]~)", self._pending_input)
                if mouse:
                    button, x, y = map(int, mouse.groups())
                    if button in (0, 64, 65):
                        self._select_mouse_fly(x, y)
                    if button in (64, 65):
                        self._handle_key("older" if button == 64 else "newer")
                    self._pending_input = self._pending_input[mouse.end():]
                elif escape:
                    self._handle_key(keys[escape[1]])
                    self._pending_input = self._pending_input[escape.end():]
                elif len(self._pending_input) < 32 and (self._pending_input == "\x1b" or
                        re.fullmatch(r"\x1b(?:\[<?[0-9;]*|O)", self._pending_input)):
                    return  # Keep an escape sequence split across reads.
                else:
                    self._pending_input = self._pending_input[1:]
            else:
                key = self._pending_input[0]
                self._pending_input = self._pending_input[1:]
                command = {"k": "older", "j": "newer", "\t": "next", "l": "live", "L": "live", "h": "help", "H": "help"}.get(key)
                if command:
                    self._handle_key(command)

    def _select_mouse_fly(self, x, y):
        titles = []
        for line_index, line in enumerate(self._last_frame.splitlines()[:y]):
            for match in re.finditer(r"Fly (\d+) /", line):
                titles.append((line_index, match.start(), int(match[1])))
        if not titles:
            return
        row = max(title[0] for title in titles)
        candidates = [title for title in titles if title[0] == row and title[1] <= x]
        if candidates:
            chosen = max(candidates, key=lambda title: title[1])[2]
            with self._lock:
                if chosen != self._feed_fly:
                    self._feed_fly, self._feed_offset, self._feed_history = chosen, 0, []

    def _handle_key(self, command):
        with self._lock:
            if command == "help":
                self._show_help = not self._show_help
                self._help_offset = 0
                return
            if self._show_help and command in ("older", "newer", "page_older", "page_newer", "oldest", "live"):
                size = self._terminal_size()
                help_rows = sum(len(textwrap.wrap(message, width=max(16, size.columns - 4))) for message in _METRIC_HELP)
                header_rows = 1 + bool(self.snapshot.get('game_setup')) + (size.columns >= 100 or size.lines >= 33)
                maximum = max(0, help_rows - max(1, size.lines - 1 - header_rows - 3))
                delta = -1 if command == "older" else 1 if command == "newer" else -5 if command == "page_older" else 5
                self._help_offset = 0 if command == "oldest" else maximum if command == "live" else max(0, min(maximum, self._help_offset + delta))
                return
            flies = self.snapshot.get("monitoring", {}).get("flies", [])
            if not flies:
                return
            visible = re.search(r"Fly (\d+) /", self._last_frame)
            selected = self._feed_fly if self._feed_fly is not None else int(visible[1]) if visible else flies[0]["fly"]
            index = next((index for index, fly in enumerate(flies) if fly['fly'] == selected), 0)
            if command in ("next", "previous"):
                index = (index + (1 if command == "next" else -1)) % len(flies)
                self._feed_fly, self._feed_offset, self._feed_history = flies[index]['fly'], 0, []
                return
            self._feed_fly = flies[index]['fly']
            if command == "live":
                self._feed_offset, self._feed_history = 0, []
                return
            if not self._feed_offset:
                self._feed_history = list(_activity_history(flies[index]))
            maximum = max(0, len(self._feed_history) - 1)
            delta = {"older": 1, "newer": -1, "page_older": 5, "page_newer": -5}.get(command, 0)
            self._feed_offset = maximum if command == "oldest" else max(0, min(maximum, self._feed_offset + delta))
            if not self._feed_offset:
                self._feed_history = []

    def _terminal_size(self):
        try:
            size = os.get_terminal_size(self.stream.fileno())
            if size.columns > 0 and size.lines > 0:
                return size
        except (AttributeError, OSError, ValueError):
            pass
        return shutil.get_terminal_size((110, 30))

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        with self._lock:
            self._show_help, self._feed_offset, self._feed_history = False, 0, []
        if self.enabled:
            try:
                if self.tty:
                    self._restore_input()
                    self.stream.write("\x1b[0m\x1b[?25h\x1b[?1049l")
                snapshot = self._view_snapshot()
                size = self._terminal_size()
                self.stream.write(render_dashboard(snapshot, width=size.columns, height=max(8, size.lines - 1),
                    colour=self.colour, unicode=self.unicode) + "\n")
                self.stream.flush()
            except (OSError, ValueError):
                pass

    def __exit__(self, kind, *_):
        if kind is not None:
            self.update(status="interrupted" if issubclass(kind, KeyboardInterrupt) else "failed")
        self.close()
