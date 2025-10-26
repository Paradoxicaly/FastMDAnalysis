from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Set


@dataclass
class ToolStats:
    modules: Set[str] = field(default_factory=set)
    functions: Set[str] = field(default_factory=set)
    files: Set[str] = field(default_factory=set)
    exceptions: int = 0
    attempts: int = 0
    successes: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "modules": sorted(self.modules),
            "functions": sorted(self.functions),
            "files": sorted(self.files),
            "exceptions": self.exceptions,
            "attempts": self.attempts,
            "successes": self.successes,
        }


class Instrumentation:
    def __init__(self, tools: Iterable[str]):
        self._stats: Dict[str, ToolStats] = {tool: ToolStats() for tool in tools}

    def record_modules(self, tool: str, modules: Iterable[str]) -> None:
        self._stats[tool].modules.update(modules)

    def record_functions(self, tool: str, functions: Iterable[str]) -> None:
        self._stats[tool].functions.update(functions)

    def record_files(self, tool: str, files: Iterable[Path]) -> None:
        stringified = [str(Path(f)) if not isinstance(f, Path) else str(f) for f in files]
        self._stats[tool].files.update(stringified)

    def record_attempt(self, tool: str) -> None:
        self._stats[tool].attempts += 1

    def record_success(self, tool: str) -> None:
        self._stats[tool].successes += 1

    def record_exception(self, tool: str) -> None:
        self._stats[tool].exceptions += 1

    def to_dict(self) -> Dict[str, Dict[str, object]]:
        return {tool: stats.to_dict() for tool, stats in self._stats.items()}

    def to_markdown(self) -> str:
        headers = [
            "Tool",
            "Modules",
            "Functions",
            "Files",
            "Attempts",
            "Successes",
            "Exceptions",
        ]
        lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
        for tool, stats in self._stats.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        tool,
                        str(len(stats.modules)),
                        str(len(stats.functions)),
                        str(len(stats.files)),
                        str(stats.attempts),
                        str(stats.successes),
                        str(stats.exceptions),
                    ]
                )
                + " |"
            )
        return "\n".join(lines) + "\n"

    def merge(self, other: "Instrumentation") -> None:
        for tool, other_stats in other._stats.items():
            stats = self._stats.setdefault(tool, ToolStats())
            stats.modules.update(other_stats.modules)
            stats.functions.update(other_stats.functions)
            stats.files.update(other_stats.files)
            stats.exceptions += other_stats.exceptions
            stats.attempts += other_stats.attempts
            stats.successes += other_stats.successes

