from pathlib import Path
from typing import Callable, Self, TypeVar

from line_profiler import LineProfiler

CallableReturnType = TypeVar("CallableReturnType")


class Profiler:
    def __init__(self, functions_to_profile: list[Callable]) -> None:
        self.line_profiler = LineProfiler(*functions_to_profile)

    def start_profile(
        self, function_to_run: Callable[..., CallableReturnType], *args, **kwargs
    ) -> CallableReturnType:
        return self.line_profiler.runcall(function_to_run, *args, **kwargs)

    def print_stats(self, file_path: Path | None = None) -> Self:
        if file_path is None:
            self.line_profiler.print_stats()

        else:
            with file_path.open(mode="w", encoding="utf-8") as file:
                self.line_profiler.print_stats(stream=file)
        return self

    def dump_stats(self, path=Path("assets/profiling/profiling_results.pkl")) -> Self:
        self.line_profiler.dump_stats(str(path))
        return self
