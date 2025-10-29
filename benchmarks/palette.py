from __future__ import annotations

from typing import Iterable, List

TOOL_COLORS = {
    "fastmdanalysis": {"primary": "#4472C4", "secondary": "#9DC3E6"},
    "mdtraj": {"primary": "#ED7D31", "secondary": "#F4B183"},
    "mdanalysis": {"primary": "#A5A5A5", "secondary": "#D9D9D9"},
}


def _normalize(tool: str) -> str:
    key = tool.lower()
    if key not in TOOL_COLORS:
        raise KeyError(f"Unknown tool '{tool}' for palette lookup")
    return key


def color_for(tool: str, *, variant: str = "primary") -> str:
    key = _normalize(tool)
    palette = TOOL_COLORS[key]
    return palette.get(variant, palette["primary"])


def colors_for(tools: Iterable[str], *, variant: str = "primary") -> List[str]:
    return [color_for(tool, variant=variant) for tool in tools]


def label_for_tool(tool: str) -> str:
    key = tool.lower()
    if key == "fastmdanalysis":
        return "FastMDAnalysis"
    if key == "mdtraj":
        return "MDTraj"
    if key == "mdanalysis":
        return "MDAnalysis"
    return tool
