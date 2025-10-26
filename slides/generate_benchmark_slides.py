from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

from pptx import Presentation
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "benchmarks" / "results"
OUTPUT_PPTX = ROOT / "slides" / "FastMDAnalysis_benchmarks.pptx"
OVERVIEW_DIR = RESULTS_ROOT / "overview"
INSTRUMENTATION_JSON = OVERVIEW_DIR / "instrumentation_summary.json"
INSTRUMENTATION_CHART = OVERVIEW_DIR / "instrumentation_overview.png"

PLOT_FILES = [
    ("Runtime Summary", "runtime_summary.png"),
    ("Peak Memory", "memory_summary.png"),
    ("Runtime Distribution", "runtime_distribution.png"),
    ("Line-of-Code Summary", "loc_summary.png"),
]

OVERVIEW_CHARTS = [
    {
        "title": "Modules touched per benchmark",
        "path": OVERVIEW_DIR / "modules_per_benchmark.png",
        "caption": "Distinct modules per workflow; FastMDAnalysis stays flat",
    },
    {
        "title": "Functions touched per benchmark",
        "path": OVERVIEW_DIR / "functions_per_benchmark.png",
        "caption": "Function touchpoints per workload",
    },
    {
        "title": "External module dependencies",
        "path": OVERVIEW_DIR / "external_modules.png",
        "caption": "Non-core imports each tool requires",
        "detail_field": "external_modules",
    },
    {
        "title": "Snippet lines of code (calc vs plot)",
        "path": OVERVIEW_DIR / "loc_totals.png",
        "caption": "Human effort split between calculation and plotting code",
        "summary_key": "loc_totals",
    },
]


def iter_benchmark_dirs(root: Path) -> Iterable[Path]:
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name != "overview":
            yield entry


def add_benchmark_slide(prs: Presentation, benchmark_dir: Path) -> None:
    title_text = benchmark_dir.name.replace("_", " ").title()
    readme = benchmark_dir / "README.md"
    summary_lines = readme.read_text(encoding="utf-8").splitlines() if readme.exists() else []

    slide = prs.slides.add_slide(prs.slide_layouts[6])

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(9.0), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.clear()
    title_run = title_frame.paragraphs[0].add_run()
    title_run.text = title_text
    title_run.font.bold = True
    title_run.font.size = Pt(28)

    if summary_lines:
        summary_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.9), Inches(9.0), Inches(1.4))
        summary_frame = summary_box.text_frame
        for idx, line in enumerate(summary_lines[:3]):
            paragraph = summary_frame.add_paragraph() if idx else summary_frame.paragraphs[0]
            paragraph.text = line.lstrip("- ")
            paragraph.font.size = Pt(14)

    positions = [
        (Inches(0.5), Inches(2.4)),
        (Inches(5.0), Inches(2.4)),
        (Inches(0.5), Inches(5.3)),
        (Inches(5.0), Inches(5.3)),
    ]
    image_width = Inches(4.0)

    for (label, filename), (left, top) in zip(PLOT_FILES, positions):
        image_path = benchmark_dir / filename
        if not image_path.exists():
            continue
        slide.shapes.add_picture(str(image_path), left, top, width=image_width)
        caption_box = slide.shapes.add_textbox(left, top - Inches(0.3), image_width, Inches(0.3))
        caption_frame = caption_box.text_frame
        caption_frame.text = label
        caption_frame.paragraphs[0].font.size = Pt(12)
        caption_frame.paragraphs[0].font.bold = True


def add_chart_slide(
    prs: Presentation,
    title: str,
    image_path: Path,
    caption: str,
    notes: Sequence[str] | None = None,
) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    if image_path.exists():
        slide.shapes.add_picture(
            str(image_path),
            Inches(0.5),
            Inches(1.0),
            width=Inches(9.0),
        )

    textbox = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9.0), Inches(0.7))
    frame = textbox.text_frame
    frame.clear()
    title_run = frame.paragraphs[0].add_run()
    title_run.text = title
    title_run.font.bold = True
    title_run.font.size = Pt(24)

    if caption or notes:
        caption_box = slide.shapes.add_textbox(Inches(0.5), Inches(6.6), Inches(9.0), Inches(1.6))
        caption_frame = caption_box.text_frame
        caption_frame.clear()

        if caption:
            first_para = caption_frame.paragraphs[0]
            first_para.text = caption
            first_para.font.size = Pt(14)
            first_para.font.bold = True
        if notes:
            for idx, note in enumerate(notes):
                paragraph = caption_frame.add_paragraph() if (caption or idx) else caption_frame.paragraphs[0]
                paragraph.text = note
                paragraph.level = 1 if caption else 0
                paragraph.font.size = Pt(12)
                paragraph.font.bold = False


def add_summary_slide(prs: Presentation, summary: dict) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9.0), Inches(6.5))
    frame = box.text_frame
    frame.word_wrap = True
    frame.clear()

    title_run = frame.paragraphs[0].add_run()
    title_run.text = "Instrumentation Overview"
    title_run.font.size = Pt(28)
    title_run.font.bold = True

    aggregated = summary.get("aggregated", {})
    loc_totals = summary.get("loc_totals", {})

    def _fmt_tool(name: str) -> str:
        return "FastMDAnalysis" if name == "fastmdanalysis" else name.capitalize()

    for tool, stats in aggregated.items():
        paragraph = frame.add_paragraph()
        paragraph.level = 1
        paragraph.text = (
            f"{_fmt_tool(tool)}: {stats['modules']} modules, {stats['functions']} functions, "
            f"{stats['external_modules']} external imports, "
            f"{stats['attempts']} runs, {stats['successes']} successful, {stats['exceptions']} exceptions."
        )
        paragraph.font.size = Pt(16)

    if loc_totals:
        snippet_para = frame.add_paragraph()
        snippet_para.level = 1
        snippet_para.font.size = Pt(16)
        chunks = [
            f"{_fmt_tool(tool)} {values.get('calc', 0)} calc / {values.get('plot', 0)} plot lines"
            for tool, values in loc_totals.items()
        ]
        snippet_para.text = "Snippet footprint: " + ", ".join(chunks) + "."

    closing = frame.add_paragraph()
    closing.level = 1
    closing.font.size = Pt(16)
    closing.text = "Charts on the following slides illustrate these touchpoint gaps."


def build_presentation(output_path: Path) -> None:
    prs = Presentation()
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = "FastMDAnalysis Benchmarks"
    subtitle = title_slide.placeholders[1]
    subtitle.text = "Runtime, memory, and usability comparisons versus MDTraj and MDAnalysis"

    benchmark_dirs = list(iter_benchmark_dirs(RESULTS_ROOT))
    summary: dict | None = None
    if INSTRUMENTATION_JSON.exists():
        summary = json.loads(INSTRUMENTATION_JSON.read_text(encoding="utf-8"))

    tools: List[str] = []
    aggregated: dict = {}
    aggregated_details: dict = {}
    per_benchmark = {}

    def _fmt_tool(name: str) -> str:
        return "FastMDAnalysis" if name == "fastmdanalysis" else name.capitalize()

    def _fmt_range(values: List[int]) -> str:
        if not values:
            return "0"
        low = min(values)
        high = max(values)
        return str(low) if low == high else f"{low}-{high}"

    if summary:
        tools = summary.get("metadata", {}).get("tools", [])
        aggregated = summary.get("aggregated", {})
        aggregated_details = summary.get("aggregated_details", {})
        per_benchmark = summary.get("per_benchmark", {})

    for benchmark_dir in benchmark_dirs:
        add_benchmark_slide(prs, benchmark_dir)
        if not summary:
            continue

        benchmark_key = benchmark_dir.name.split("_", 1)[0]
        chart_path = OVERVIEW_DIR / f"instrumentation_{benchmark_key}.png"
        entry = per_benchmark.get(benchmark_key, {})
        if not chart_path.exists() or not entry:
            continue

        notes: List[str] = []
        ordered_tools = tools or list(entry.keys())
        for tool in ordered_tools:
            stats = entry.get(tool, {})
            notes.append(
                f"{_fmt_tool(tool)}: {stats.get('modules', 0)} modules, {stats.get('functions', 0)} functions, "
                f"{stats.get('attempts', 0)} runs, {stats.get('successes', 0)} successful, {stats.get('exceptions', 0)} exceptions"
            )

        add_chart_slide(
            prs,
            f"{benchmark_key.upper()} instrumentation overview",
            chart_path,
            "Workflow touchpoints captured per tool",
            notes=notes,
        )

    if summary:
        add_summary_slide(prs, summary)

        instrumentation_notes: List[str] | None = None
        if INSTRUMENTATION_CHART.exists() and aggregated:
            instrumentation_notes = []
            ordered_tools = tools or list(aggregated.keys())
            for tool in ordered_tools:
                stats = aggregated.get(tool, {})
                instrumentation_notes.append(
                    f"{_fmt_tool(tool)}: {stats.get('modules', 0)} modules, {stats.get('functions', 0)} functions, "
                    f"{stats.get('external_modules', 0)} external imports, "
                    f"{stats.get('attempts', 0)} runs, {stats.get('successes', 0)} successful, {stats.get('exceptions', 0)} exceptions"
                )
            loc_totals = summary.get("loc_totals", {})
            if loc_totals:
                footprint = ", ".join(
                    f"{_fmt_tool(tool)} {vals.get('calc', 0)} calc / {vals.get('plot', 0)} plot lines"
                    for tool, vals in loc_totals.items()
                )
                instrumentation_notes.append(f"Snippet footprint: {footprint}")
        if INSTRUMENTATION_CHART.exists():
            add_chart_slide(
                prs,
                "Instrumentation overview",
                INSTRUMENTATION_CHART,
                "Touchpoint counts captured per tool",
                notes=instrumentation_notes,
            )

        for chart in OVERVIEW_CHARTS:
            chart_path = chart["path"]
            if not chart_path.exists():
                continue

            notes: List[str] | None = None
            detail_field = chart.get("detail_field")
            if detail_field and tools and aggregated and aggregated_details:
                notes = []
                for tool in tools:
                    stats = aggregated.get(tool, {})
                    detail_items = aggregated_details.get(tool, {}).get(detail_field, [])
                    amount = stats.get(detail_field, len(detail_items))
                    joined = ", ".join(detail_items) if detail_items else "(none)"
                    notes.append(f"{_fmt_tool(tool)} ({amount}): {joined}")

            summary_key = chart.get("summary_key")
            if summary_key == "loc_totals":
                loc_totals = summary.get("loc_totals", {})
                if loc_totals:
                    notes = notes or []
                    fast = loc_totals.get("fastmdanalysis", {})
                    if fast:
                        fast_calc = fast.get("calc", 0)
                        fast_plot = fast.get("plot", 0)
                        other_plot = [vals.get("plot", 0) for name, vals in loc_totals.items() if name != "fastmdanalysis"]
                        other_calc = [vals.get("calc", 0) for name, vals in loc_totals.items() if name != "fastmdanalysis"]
                        if other_plot:
                            notes.append(
                                "FastMDAnalysis keeps plotting helpers lean "
                                f"({fast_plot} lines vs {_fmt_range(other_plot)} elsewhere)."
                            )
                        if other_calc:
                            notes.append(
                                "Calculation snippets stay compact too "
                                f"({fast_calc} lines vs {_fmt_range(other_calc)} for the others)."
                            )
                    for tool in tools:
                        values = loc_totals.get(tool, {})
                        if values:
                            notes.append(
                                f"{_fmt_tool(tool)}: {values.get('calc', 0)} calc lines, {values.get('plot', 0)} plotting lines"
                            )

            add_chart_slide(
                prs,
                chart["title"],
                chart_path,
                chart.get("caption", ""),
                notes=notes,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output_path)
    print(f"Saved benchmark slideshow to {output_path}")


if __name__ == "__main__":
    build_presentation(OUTPUT_PPTX)
