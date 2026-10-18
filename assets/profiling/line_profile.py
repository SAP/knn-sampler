from collections.abc import Callable
from pathlib import Path
from typing import Any, Self, TypeVar, cast

try:
    from line_profiler import LineProfiler as _LineProfiler
except ImportError:
    _LineProfiler = None  # type: ignore[assignment,misc]

CallableReturnType = TypeVar("CallableReturnType")

# Anchored to the project root so the default does not depend on the CWD
DEFAULT_DUMP_PATH = (
    Path(__file__).resolve().parents[2] / "assets/profiling/profiling_results.pkl"
)


class Profiler:
    def __init__(self, functions_to_profile: list[Callable]) -> None:
        if _LineProfiler is not None:
            self.line_profiler: _LineProfiler | None = _LineProfiler(
                *functions_to_profile
            )
        else:
            self.line_profiler = None

    def start_profile(
        self,
        function_to_run: Callable[..., CallableReturnType],
        *args: Any,
        **kwargs: Any,
    ) -> CallableReturnType:
        if self.line_profiler is not None:
            return cast(
                CallableReturnType,
                self.line_profiler.runcall(function_to_run, *args, **kwargs),
            )
        return function_to_run(*args, **kwargs)

    def print_stats(self, file_path: Path | None = None) -> Self:
        if self.line_profiler is not None:
            if file_path is None:
                self.line_profiler.print_stats()

            else:
                with file_path.open(mode="w", encoding="utf-8") as file:
                    self.line_profiler.print_stats(stream=file)
        return self

    def dump_stats(self, path: Path = DEFAULT_DUMP_PATH) -> Self:
        if self.line_profiler is not None:
            self.line_profiler.dump_stats(str(path))
        return self
