import io
import json
import os
import pickle
import re
import subprocess
import termios
from types import SimpleNamespace

import numpy as np
import pytest

from flylatro.env.balatro_sim import ArrayStep
from flylatro.env.upstream_contract import BLIND_REQ_OFF, BLIND_SCORED_OFF, GLOBAL_ANTE_OFF, GLOBAL_PHASE_OFF, GLOBAL_ROUND
from flylatro.env.upstream_contract import UpstreamActionType, empty_action_batch
from flylatro.telemetry.actions import describe_action
from flylatro.telemetry.item_names import JOKER_NAMES
from flylatro.learning.cli import main as train_main
from flylatro.telemetry.dashboard import TerminalDashboard, TrainingMonitor, render_dashboard, sample_gpus, _round_bars
from helpers import write_config


def observations(chips=(32, 62)):
    count = len(chips)
    result = {"global": np.zeros((count, 64)), "blind": np.zeros((count, 38))}
    result["global"][:, GLOBAL_ANTE_OFF] = 1
    result["global"][:, GLOBAL_PHASE_OFF + 1] = 1
    result["global"][:, GLOBAL_ROUND] = 1 / 24
    result["blind"][:, BLIND_REQ_OFF] = np.log1p(300)
    result["blind"][:, BLIND_SCORED_OFF] = np.log1p(chips)
    return result


def trainer():
    return SimpleNamespace(env=SimpleNamespace(num_envs=2), observations=observations(), last_step=None,
        state=SimpleNamespace(vector_steps=0, environment_decisions=0))


def test_per_fly_outcomes_auto_reset_high_scores_and_resume_are_accounted_separately():
    source = trainer()
    monitor = TrainingMonitor(source)
    source.observations = observations((0, 0))
    source.state.vector_steps = 1; source.state.environment_decisions = 2
    source.last_step = ArrayStep(source.observations, {}, np.array([2.5, -0.6]), np.array([True, True]),
        ({"episode": {"won": True, "ante": 2, "r": 2.5, "l": 1}},
         {"episode": {"won": False, "ante": 1, "r": -0.6, "l": 1, "score": 71}}))
    episodes = monitor.update(source)
    first, second = monitor.flies
    assert (first.completed_runs, first.wins, first.game_overs, first.decisions) == (1, 1, 0, 1)
    assert (second.completed_runs, second.wins, second.game_overs, second.decisions) == (1, 0, 1, 1)
    assert first.run_return == second.run_return == 0  # next, auto-reset run
    assert first.best_return == 2.5 and second.best_return == -0.6
    assert first.best_terminal_score is None and second.best_terminal_score == 71
    assert first.peak_observed_chips == pytest.approx(32)
    assert second.peak_observed_chips == pytest.approx(62)
    assert [episode["fly"] for episode in episodes] == [1, 2]
    restored = TrainingMonitor(source, monitor.state_dict())
    assert restored.state_dict() == monitor.state_dict()


def test_gpu_sampler_handles_unavailable_driver_and_unsupported_fields(monkeypatch):
    from flylatro.telemetry import dashboard
    monkeypatch.setattr(dashboard.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0,
        stdout='0, GPU-test, RTX test, 95, 12, 3000, 16376, N/A, 111.5\n'))
    result = sample_gpus()
    assert result["gpus"][0]["utilization_percent"] == 95
    assert result["gpus"][0]["temperature_c"] is None
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 0.8)
    monkeypatch.setattr(dashboard.subprocess, "run", timeout)
    assert sample_gpus()["status"] == "unavailable"


def test_display_handles_resize_unknown_scores_and_nonfinite_metrics():
    monitor = TrainingMonitor(trainer())
    snapshot = {"run_name": "test", "status": "training", "device": "cpu", "decisions": 0,
        "target_decisions": 100, "monitoring": monitor.state_dict(),
        "metrics": {"runtime/environment_decisions_per_second": None, "plasticity/max_learner_bound_fraction": None}}
    frame = render_dashboard(snapshot, width=80, height=30)
    assert "FLYLATRO" in frame and "Fly 01" in frame and "Final game score --" in frame
    assert max(map(len, frame.splitlines())) <= 80
    many = {**snapshot, "monitoring": {**monitor.state_dict(), "flies": monitor.state_dict()["flies"] * 20}}
    assert "rotates every 8 seconds" in render_dashboard(many, width=100, height=25)
    for width, height in ((110, 30), (80, 24), (50, 20)):
        fancy = render_dashboard(many, width=width, height=height, colour=True, unicode=True)
        plain = re.sub(r"\x1b\[[0-9;]*m", "", fancy)
        assert "Fly 01" in plain and len(plain.splitlines()) <= height
        assert max(map(len, plain.splitlines())) <= width
        assert "\x1b[" in fancy


def test_terminal_restores_cursor_after_exception_and_redirected_output_has_no_ansi(monkeypatch):
    from flylatro.telemetry import dashboard
    monkeypatch.setattr(dashboard.shutil, "get_terminal_size", lambda *_: SimpleNamespace(columns=80, lines=30))
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    class TTY(io.StringIO):
        def isatty(self):
            return True
    stream = TTY()
    with pytest.raises(KeyboardInterrupt):
        with TerminalDashboard(mode="on", refresh=0.2, device="cpu", run_name="test", run_dir="test", stream=stream,
                target_decisions=100):
            raise KeyboardInterrupt
    assert "\x1b[?25l" in stream.getvalue() and "\x1b[?25h\x1b[?1049l" in stream.getvalue()
    final_frame = stream.getvalue().split("\x1b[?25h\x1b[?1049l")[-1]
    plain = re.sub(r"\x1b\[[0-9;]*m", "", final_frame)
    assert "0/100 decisions" in plain and max(map(len, plain.splitlines())) <= 80
    monkeypatch.setenv("NO_COLOR", "1")
    with TerminalDashboard(mode="on", refresh=0.2, device="cpu", run_name="test", run_dir="test", stream=TTY()) as display:
        assert display.colour is False and display.unicode is True
    class ASCII(TTY):
        @property
        def encoding(self):
            return "ascii"
    monkeypatch.delenv("NO_COLOR")
    stream = ASCII()
    with TerminalDashboard(mode="on", refresh=0.2, device="cpu", run_name="test", run_dir="test", stream=stream) as display:
        assert display.unicode is False
    stream.getvalue().encode("ascii")
    stream = io.StringIO()
    with TerminalDashboard(mode="on", refresh=0.2, device="cpu", run_name="test", run_dir="test", stream=stream):
        pass
    assert "FLYLATRO" in stream.getvalue() and "\x1b" not in stream.getvalue()
    stream = io.StringIO()
    with TerminalDashboard(mode="auto", refresh=0.2, device="cpu", run_name="test", run_dir="test", stream=stream):
        pass
    assert stream.getvalue() == ""


def test_recent_actions_use_pre_step_cards_and_items_and_survive_resume():
    source = trainer()
    source.observations["hand"] = np.zeros((2, 10, 38))
    for slot, rank, suit in ((0, 12, 0), (1, 5, 1), (2, 8, 2)):
        source.observations["hand"][0, slot, rank] = 1
        source.observations["hand"][0, slot, 13 + suit] = 1
    source.observations["shop_ids"] = np.zeros((2, 6), dtype=np.int64)
    source.observations["shop_ids"][1, 1] = JOKER_NAMES.index("Blueprint") + 1
    monitor = TrainingMonitor(source)
    actions = empty_action_batch(2)
    actions["action_type"][:] = [UpstreamActionType.PLAY_HAND, UpstreamActionType.BUY_SHOP]
    actions["n_cards"][:] = [3, 0]
    actions["cards"][0] = [0, 1, 2, -1, -1]
    actions["shop_target"][1] = 1
    source.last_executed_actions = actions
    # Mimic an environment overwriting the old buffers after play/shop removal.
    source.observations["hand"][:] = 0
    source.observations["shop_ids"][:] = 0
    source.state.environment_decisions = 2
    source.state.vector_steps = 1
    source.last_step = ArrayStep(source.observations, {}, np.array([1.0, 0.0]), np.array([True, False]),
        ({"episode": {"won": True, "ante": 2, "r": 1.0, "l": 1}}, {}))
    monitor.update(source)
    assert monitor.flies[0].recent_decisions[0]["description"] == "Played Ace S, 7 H, 10 C"
    assert [entry["event"] for entry in monitor.flies[0].recent_activity if entry["kind"] == "event"] == [
        "round_beaten", "ante_beaten", "run_won"]
    assert monitor.flies[1].recent_decisions[0]["description"] == "Bought Joker: Blueprint"
    assert all(entry["kind"] == "decision" for entry in monitor.flies[1].recent_activity)
    for step in range(2, 12):
        source.state.vector_steps = step
        source.state.environment_decisions = step * 2
        monitor.update(source)
    assert len(monitor.flies[0].recent_decisions) == 8
    assert [entry["decision"] for entry in monitor.flies[0].recent_decisions] == list(range(4, 12))
    assert TrainingMonitor(source, monitor.state_dict()).state_dict() == monitor.state_dict()
    legacy = monitor.state_dict()
    for fly in legacy["flies"]:
        fly.pop("recent_decisions")
    assert all(not fly.recent_decisions for fly in TrainingMonitor(source, legacy).flies)


def test_action_labels_respect_hidden_cards_and_pinned_item_vocabulary():
    obs = observations((0,))
    obs["hand"] = np.zeros((1, 10, 38))
    obs["hand"][0, 0, 12] = 1  # identity is hidden even if a stale rank were present
    obs["hand"][0, 0, 37] = 1
    obs["hand"][0, 1, 23] = 1
    action = empty_action_batch(1)
    action["action_type"][0] = UpstreamActionType.DISCARD
    action["n_cards"][0] = 2
    action["cards"][0] = [0, 1, -1, -1, -1]
    assert describe_action(action, obs, 0) == "Discarded face-down card [1], Stone card"
    obs["shop_ids"] = np.zeros((1, 6), dtype=np.int64)
    action["action_type"][0] = UpstreamActionType.BUY_SHOP
    action["shop_target"][0] = 0
    action["n_cards"][0] = 0
    for token, name in ((160, "Tarot: The Fool"), (182, "Planet: Mercury"), (194, "Spectral: Familiar"),
            (268, "card: Ace S"), (280, "Voucher: Overstock"), (320, "Arcana Pack"), (999, "item ID 999")):
        obs["shop_ids"][0, 0] = token
        assert name in describe_action(action, obs, 0)


def test_game_milestones_are_separate_events_without_extra_decisions_or_duplicate_ante_clears():
    source = trainer()
    source.observations["blind"][0, 2] = 1  # boss blind
    monitor = TrainingMonitor(source)
    source.last_executed_actions = empty_action_batch(2)
    source.last_executed_actions["n_cards"][:] = 0
    source.last_executed_actions["action_type"][:] = UpstreamActionType.PLAY_HAND
    cash_out = observations((0, 0))
    cash_out["global"][0, GLOBAL_PHASE_OFF + 1] = 0
    cash_out["global"][0, GLOBAL_PHASE_OFF + 2] = 1
    source.observations = cash_out
    source.state.environment_decisions = 2
    source.state.vector_steps = 1
    source.last_step = ArrayStep(cash_out, {}, np.array([0.0, 1.0]), np.array([False, False]),
        ({"reward_components": {"blind_clear": 0.0}}, {"reward_components": {"blind_progress": 1.0}}))
    monitor.update(source)
    first = monitor.flies[0]
    assert first.decisions == 1 and len(first.recent_decisions) == 1
    events = [entry for entry in first.recent_activity if entry["kind"] == "event"]
    assert [entry["event"] for entry in events] == ["round_beaten", "ante_beaten"]
    assert all("decision" not in entry for entry in events)
    assert all(entry["kind"] == "decision" for entry in monitor.flies[1].recent_activity)
    # Cashing out advances the Ante but does not emit the same clear twice.
    next_ante = observations((0, 0))
    next_ante["global"][0, GLOBAL_ANTE_OFF] = 0
    next_ante["global"][0, GLOBAL_ANTE_OFF + 1] = 1
    source.last_executed_actions["action_type"][:] = UpstreamActionType.CASH_OUT
    source.observations = next_ante
    source.state.environment_decisions = 4
    source.state.vector_steps = 2
    source.last_step = ArrayStep(next_ante, {}, np.zeros(2), np.array([False, True]),
        ({"ante_cleared": True}, {"episode": {"won": False, "ante": 1, "r": 1, "l": 2}}))
    monitor.update(source)
    assert first.decisions == 2
    assert sum(entry.get("event") == "ante_beaten" for entry in first.recent_activity) == 1
    assert monitor.flies[1].recent_activity[-1]["event"] == "game_over"
    assert monitor.flies[1].decisions == 2
    # Hieroglyph/Petroglyph can lower the Ante. A later boss at the same Ante
    # is another real clear; only the same round's cashout must be deduplicated.
    replayed_ante = observations((0, 0))
    replayed_ante["global"][0, GLOBAL_ROUND] = 4 / 24
    replayed_ante["blind"][0, 2] = 1
    monitor.observe(replayed_ante)
    replayed_ante["global"][0, GLOBAL_PHASE_OFF + 1] = 0
    replayed_ante["global"][0, GLOBAL_PHASE_OFF + 2] = 1
    source.observations = replayed_ante
    source.last_executed_actions["action_type"][:] = UpstreamActionType.PLAY_HAND
    source.state.environment_decisions = 6
    source.last_step = ArrayStep(replayed_ante, {}, np.zeros(2), np.array([False, False]), ({}, {}))
    monitor.update(source)
    assert sum(entry.get("event") == "ante_beaten" for entry in first.recent_activity) == 2
    assert first.decisions == 3
    snapshot = {"monitoring": monitor.state_dict(), "device": "cpu", "decisions": 4, "target_decisions": 100}
    for frame in (render_dashboard(snapshot, width=110, height=40),
            render_dashboard(snapshot, width=110, height=40, unicode=True)):
        plain = re.sub(r"\x1b\[[0-9;]*m", "", frame)
        for line in plain.splitlines():
            for cell in line.split("│"):
                if any(label in cell for label in ("GAME OVER", "ANTE BEATEN", "ROUND BEATEN")):
                    assert "#" not in cell
        assert "GAME OVER" in plain and "ANTE BEATEN" in plain and "ROUND BEATEN" in plain


def test_round_statistics_use_completed_runs_before_auto_reset_and_keep_bounded_graph_history():
    source = trainer()
    source.observations["global"][:, GLOBAL_ROUND] = [7 / 24, 2 / 24]
    monitor = TrainingMonitor(source)
    reset = observations((0, 0))
    reset["global"][:, GLOBAL_ROUND] = [0, 1 / 24]
    source.observations = reset
    source.state.environment_decisions = 2
    source.last_step = ArrayStep(reset, {}, np.zeros(2), np.array([True, False]),
        ({"episode": {"won": False}}, {}))
    monitor.update(source)
    first, second = monitor.flies
    assert first.round == 0 and first.highest_round == 7
    assert first.round_sum == 7 and first.round_samples == 1
    assert first.round_history[0]["round"] == 7
    assert second.round_samples == 0 and second.round_history == []
    assert second.run_highest_round == 2  # retain the maximum reached in this run
    # Explicit terminal round info refers to the finished run, never next obs.
    for index in range(1, 71):
        source.state.environment_decisions += 2
        source.last_step = ArrayStep(reset, {}, np.zeros(2), np.array([True, False]),
            ({"episode": {"won": False, "round": index}}, {}))
        monitor.update(source)
    assert first.round_samples == 71 and first.round_sum == 7 + sum(range(1, 71))
    assert len(first.round_history) == 64 and first.round_history[0]["run"] == 8
    assert first.highest_round == 70
    assert TrainingMonitor(source, monitor.state_dict()).state_dict() == monitor.state_dict()
    snapshot = {"monitoring": monitor.state_dict(), "device": "cpu", "decisions": 142, "target_decisions": 1000}
    frame = render_dashboard(snapshot, width=110, height=44, unicode=True)
    assert "Highest round: 70  Average: 35.10 (71 runs)" in frame
    assert "RECENT RUNS" in frame and "Run 71" in frame and "70 played" in frame
    assert "█" in frame and "scale 0 to 70" in frame
    assert "Training reward" in frame and "Highest blind chips" in frame
    # Old monitoring checkpoints cannot reconstruct round averages from Ante.
    legacy = monitor.state_dict()
    for fly in legacy["flies"]:
        for name in ("round", "highest_round", "run_highest_round", "round_sum", "round_samples", "round_history_start_run", "round_history"):
            fly.pop(name)
    restored = TrainingMonitor(source, legacy)
    assert restored.flies[0].round_samples == 0 and restored.flies[0].round_history == []
    assert restored.flies[0].round_history_start_run == 71
    source.observations["global"][0, GLOBAL_ROUND] = np.nan
    restored.observe(source.observations)
    assert restored.flies[0].round is None


def test_performance_bars_have_direct_values_zero_baseline_shared_scale_and_chronological_order():
    fly = {"round_history": [{"run": 7, "round": 0, "skipped_rounds": 0},
        {"run": 8, "round": 2, "skipped_rounds": 1}, {"run": 9, "round": 4, "skipped_rounds": 2}]}
    lines = _round_bars(fly, width=46, rows=4, colour=False, unicode=False, scale=8)
    assert "scale 0 to 8" in lines[0]
    assert "Run 7" in lines[1] and "0 played + 0 skipped" in lines[1] and "#" not in lines[1]
    assert "Run 8" in lines[2] and "2 played + 1 skipped" in lines[2]
    assert "Run 9" in lines[3] and "4 played + 2 skipped" in lines[3]
    assert abs(lines[3].count("#") - 2 * lines[2].count("#")) <= 1  # terminal-cell quantisation
    assert lines[2].count("+") > 1 and lines[3].count("+") > lines[2].count("+")
    frame = render_dashboard({"monitoring": {"flies": []}, "throughput_history": [1, 2, 3]}, unicode=True)
    assert "RATE" not in frame and not any(character in frame for character in "▁▂▃▄▅▆▇")


def test_skipped_blinds_are_separate_from_played_rounds_clears_and_resume_with_unknown_legacy_counts():
    source = trainer()
    source.observations['global'][:, GLOBAL_ROUND] = 0
    source.observations['global'][:, GLOBAL_PHASE_OFF + 1] = 0
    source.observations['global'][:, GLOBAL_PHASE_OFF] = 1
    monitor = TrainingMonitor(source)
    actions = empty_action_batch(2)
    actions['n_cards'][:] = 0
    source.last_executed_actions = actions
    for index in range(1, 5):
        actions['action_type'][:] = [UpstreamActionType.SKIP_BLIND if index <= 2 else
            UpstreamActionType.SELECT_BLIND if index == 3 else UpstreamActionType.PLAY_HAND,
            UpstreamActionType.SELECT_BLIND if index == 1 else UpstreamActionType.PLAY_HAND]
        after = observations((0, 0))
        if index <= 2:
            after['global'][0, GLOBAL_ROUND] = 0
            after['global'][0, GLOBAL_PHASE_OFF + 1] = 0
            after['global'][0, GLOBAL_PHASE_OFF] = 1
        done = index == 4
        if done:
            after['global'][:, GLOBAL_ROUND] = 0  # next run auto-reset
        source.observations = after
        source.state.environment_decisions = index * 2
        source.last_step = ArrayStep(after, {}, np.zeros(2), np.array([done, done]),
            ({'episode': {'won': True}}, {'episode': {'won': False}}) if done else ({}, {}))
        episodes = monitor.update(source)
        if index <= 2:
            assert not any(entry['kind'] == 'event' for entry in monitor.flies[0].recent_activity)
    first, second = monitor.flies
    assert first.round_history[-1]['round'] == 1 and first.round_history[-1]['skipped_rounds'] == 2
    assert second.round_history[-1]['skipped_rounds'] == 0
    assert first.round_sum == first.highest_round == 1  # skips do not inflate averages
    assert first.run_skipped_rounds == second.run_skipped_rounds == 0
    assert [episode['skipped_rounds'] for episode in episodes] == [2, 0]
    assert sum(entry.get('event') == 'round_beaten' for entry in first.recent_activity) == 1
    assert first.decisions == 4
    saved = monitor.state_dict()
    assert TrainingMonitor(source, saved).state_dict() == saved
    for fly in saved['flies']:
        fly.pop('run_skipped_rounds')
        fly['run_decisions'] = 3
        for entry in fly['round_history']:
            entry.pop('skipped_rounds')
    restored = TrainingMonitor(source, saved)
    assert restored.flies[0].run_skipped_rounds is None
    bars = _round_bars(saved['flies'][0], width=50, rows=4, colour=False, unicode=False)
    assert '1 played + ? skipped' in bars[-1]


def test_feed_keyboard_mouse_and_help_navigation_freeze_only_the_browsed_history():
    display = TerminalDashboard(mode="off", refresh=0.2, device="cpu", run_name="test", run_dir="test", stream=io.StringIO())
    monitor = TrainingMonitor(trainer()).state_dict()
    for fly in monitor["flies"]:
        fly["recent_activity"] = [{"kind": "decision", "decision": index, "run": 1,
            "description": f"Played card {index}"} for index in range(1, 13)]
    display.update(monitoring=monitor)
    display._consume_input("\x1b[")
    assert display._feed_offset == 0
    display._consume_input("A")
    assert display._feed_offset == 1 and display._feed_fly == 1
    monitor["flies"][0]["recent_activity"].append({"kind": "decision", "decision": 13, "run": 1, "description": "Bought Joker: Blueprint"})
    display.update(monitoring=monitor, decisions=13)
    assert display._view_snapshot()["decisions"] == 13
    assert len(display._feed_history) == 12  # new actions don't move the view
    frame = render_dashboard(display._view_snapshot(), width=110, height=40, unicode=True)
    assert "HISTORY / 1 older" in frame and "#11 [r1] Played card 11" in frame
    assert "Bought Joker: Blueprint" not in frame
    display._consume_input("\x1b[5~")
    assert display._feed_offset == 6
    display._consume_input("\x1b[F")
    assert display._feed_offset == 0 and display._feed_history == []
    frame = render_dashboard(display._view_snapshot(), width=110, height=40, unicode=True)
    assert "Bought Joker: Blueprint" in frame
    display._consume_input("\t")
    assert display._feed_fly == 2
    display._last_frame = frame
    title_line = next(index for index, line in enumerate(frame.splitlines(), 1) if "Fly 01 /" in line)
    display._consume_input(f"\x1b[<64;90;{title_line + 4}M")
    assert display._feed_offset == 1 and display._feed_fly == 2
    display._consume_input("h")
    help_frame = render_dashboard(display._view_snapshot(), width=110, height=40, unicode=True)
    assert "METRICS / CONTROLS" in help_frame and "not game chips" in help_frame
    display._consume_input("hL")
    assert not display._show_help and display._feed_offset == 0


def test_interactive_input_preserves_ctrl_c_and_restores_terminal_settings_on_interrupt(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(os, "tcgetpgrp", lambda _: os.getpgrp())
    class TTY(io.StringIO):
        def isatty(self):
            return True
    master, slave = os.openpty()
    try:
        original = termios.tcgetattr(slave)
        with os.fdopen(os.dup(slave), "r") as input_stream:
            stream = TTY()
            with pytest.raises(KeyboardInterrupt):
                with TerminalDashboard(mode="on", refresh=0.2, device="cpu", run_name="test", run_dir="test",
                        stream=stream, input_stream=input_stream):
                    changed = termios.tcgetattr(slave)
                    assert not changed[3] & (termios.ICANON | termios.ECHO)
                    assert changed[3] & termios.ISIG == original[3] & termios.ISIG
                    raise KeyboardInterrupt
            assert termios.tcgetattr(slave) == original
            assert "\x1b[?1000h" in stream.getvalue() and "\x1b[?1000l" in stream.getvalue()
    finally:
        os.close(master)
        os.close(slave)


def test_dashboard_is_observational_and_checkpoint_counters_survive_multi_fly_resume(tmp_path, capsys):
    config = write_config(tmp_path, overrides={"environment": {"num_envs": 2},
        "training": {"max_environment_decisions": 20, "checkpoint_every_decisions": 4}})
    runs = []
    for mode in ("--no-dashboard", "--dashboard"):
        run = tmp_path / mode.removeprefix("--")
        assert train_main(["--config", str(config), "--run-dir", str(run), mode, "--no-tensorboard"]) == 0
        runs.append(run)
    first, second = [pickle.loads((run / "plastic-checkpoint-final.pkl").read_bytes()) for run in runs]
    np.testing.assert_array_equal(first["trainer"]["plastic_state"]["efficacy"], second["trainer"]["plastic_state"]["efficacy"])
    assert (runs[0] / "training-actions.jsonl").read_bytes() == (runs[1] / "training-actions.jsonl").read_bytes()
    assert first["monitoring_state"] == second["monitoring_state"]
    partial = tmp_path / "staged"
    common = ["--config", str(config), "--run-dir", str(partial), "--no-dashboard", "--no-tensorboard"]
    assert train_main([*common, "--stop-after-decisions", "8"]) == 0
    assert train_main([*common, "--resume", str(partial / "plastic-checkpoint-stage-000000000008.pkl")]) == 0
    final = pickle.loads((partial / "plastic-checkpoint-final.pkl").read_bytes())
    assert final["monitoring_state"] == first["monitoring_state"]
    snapshot = json.loads((partial / "training-dashboard.json").read_text())
    assert snapshot["status"] == "completed" and snapshot["decisions"] == 20
    assert sum(fly["decisions"] for fly in snapshot["monitoring"]["flies"]) == 20
    assert sum(fly["completed_runs"] for fly in snapshot["monitoring"]["flies"]) == snapshot["completed_runs"]
    episodes = [json.loads(line) for line in (partial / "training-episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == snapshot["completed_runs"]
    output = capsys.readouterr()
    assert "FLYLATRO" in output.err and "\x1b" not in output.out


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_dashboard_refresh_refuses_invalid_values(tmp_path, value):
    with pytest.raises(ValueError, match="dashboard refresh"):
        train_main(["--config", "unused.toml", "--dashboard-refresh", value])
