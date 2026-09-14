"""Profile the similarity checker on a short Chinese text sample."""

import cProfile
import pstats
import time
from pathlib import Path

from checker.algorithm import calculate_similarity


ORIGINAL_TEXT = "今天是星期天，天气晴，今天晚上我要去看电影。"
PLAGIARIZED_TEXT = "今天是周天，天气晴朗，我晚上要去看电影。"


def run_sample() -> float:
    """Run one short-text similarity calculation for profiling."""
    return calculate_similarity(ORIGINAL_TEXT, PLAGIARIZED_TEXT)


def main() -> None:
    output_dir = Path("profile_output")
    output_dir.mkdir(exist_ok=True)
    profile_path = output_dir / "short_text.prof"
    stats_path = output_dir / "short_text_stats.txt"

    # Exclude jieba's one-time dictionary initialization from the measured run.
    run_sample()
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    similarity = run_sample()
    profiler.disable()
    elapsed = time.perf_counter() - started
    profiler.dump_stats(str(profile_path))

    with stats_path.open("w", encoding="utf-8") as output:
        stats = pstats.Stats(profiler, stream=output)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)

    print(f"similarity={similarity:.2f}")
    print(f"elapsed={elapsed:.5f}s")
    print(f"profile={profile_path}")
    print(f"stats={stats_path}")


if __name__ == "__main__":
    main()