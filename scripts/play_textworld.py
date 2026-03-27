#!/usr/bin/env python3
"""
Play a TextWorld .ulx game interactively in the terminal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _unpack_reset(result: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return (obs if isinstance(obs, str) else str(obs), info if isinstance(info, dict) else {})
    return (result if isinstance(result, str) else str(result), {})


def _unpack_step(result: Any) -> tuple[str, float, bool, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated) or bool(truncated)
        return (
            obs if isinstance(obs, str) else str(obs),
            float(reward),
            done,
            info if isinstance(info, dict) else {},
        )
    if isinstance(result, tuple) and len(result) >= 4:
        obs, reward, done, info = result[0], result[1], result[2], result[3]
        return (
            obs if isinstance(obs, str) else str(obs),
            float(reward),
            bool(done),
            info if isinstance(info, dict) else {},
        )
    raise RuntimeError("Unexpected env.step return format")


def _print_status(info: dict[str, Any], fallback_score: float | None) -> float | None:
    score = info.get("score", fallback_score)
    max_score = info.get("max_score")
    try:
        score_val = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_val = fallback_score
    if max_score is not None:
        try:
            max_score = float(max_score)
        except (TypeError, ValueError):
            max_score = None
    if score_val is not None and max_score is not None:
        print(f"[score] {score_val:.1f}/{max_score:.1f}")
    elif score_val is not None:
        print(f"[score] {score_val:.1f}")
    else:
        print("[score] n/a")
    return score_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a TextWorld .ulx game interactively.")
    parser.add_argument("game_file", help="Path to .ulx file")
    parser.add_argument("--max-steps", type=int, default=50, help="Interactive cap for this debug session")
    args = parser.parse_args()

    game_file = Path(args.game_file).expanduser().resolve()
    if not game_file.exists():
        raise FileNotFoundError(f"Game file not found: {game_file}")

    try:
        import gym  # noqa: F401
        import textworld.gym
        from textworld import EnvInfos
    except Exception as e:
        raise RuntimeError("textworld + gym are required. Install `textworld` and `gym`.") from e

    infos = EnvInfos(admissible_commands=True, feedback=True, score=True)
    env_id = textworld.gym.register_game(
        str(game_file),
        max_episode_steps=int(args.max_steps),
        name=f"play_{game_file.stem}",
        request_infos=infos,
    )
    env = gym.make(env_id)

    obs, info = _unpack_reset(env.reset())
    print(obs)
    score_cache: float | None = None
    score_cache = _print_status(info, score_cache)
    steps = 0

    while True:
        try:
            cmd = input("> ").strip()
        except EOFError:
            print("\nExiting.")
            return
        if cmd.lower() in {"quit", "exit", ":q"}:
            print("Exiting.")
            return
        if not cmd:
            continue
        steps += 1
        obs, _, done, info = _unpack_step(env.step(cmd))
        print(obs)
        score_cache = _print_status(info, score_cache)
        if done:
            won = bool(info.get("won") or info.get("game_won"))
            lost = bool(info.get("lost") or info.get("game_lost"))
            if won:
                print("YOU WON")
            elif lost:
                print("YOU LOST")
            elif steps >= int(args.max_steps):
                print("MAX STEPS REACHED")
            else:
                print("YOU LOST")
            return
        if steps >= int(args.max_steps):
            print("MAX STEPS REACHED")
            return


if __name__ == "__main__":
    main()
