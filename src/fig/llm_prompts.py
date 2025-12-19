"""Prompt and context builders for LLM integration in Financial Insight Generator.

This module is responsible for turning the structured `metrics_bundle` and
related context into safe, compact text prompts that can be passed to an LLM.

Design goals
------------
- Keep all prompt construction in one place so it is easy to audit and test.
- Use the existing analytics output (`metrics_bundle`) as the *only* source of truth.
- Explicitly encode language requirements (English / Bahasa Indonesia).
- Avoid leaking large raw DataFrames by summarising and truncating context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import math

import pandas as pd


@dataclass
class MetricsSummaryOptions:
    """Options controlling how the metrics bundle is summarised for prompts."""

    max_context_chars: int = 12_000
    max_time_series_rows: int = 6
    max_segments_per_type: int = 5


def _normalise_language(language: str | None) -> str:
    """Normalise language codes to a small set we actually handle."""
    if not language:
        return "en"
    lang = str(language).strip().lower()
    if lang.startswith("id"):
        return "id"
    if lang.startswith("en"):
        return "en"
    # Default to English for unknown codes.
    return "en"


# ---------------------------------------------------------------------------
# Metrics bundle summarisation
# ---------------------------------------------------------------------------


def _format_overall_section(overall: Mapping[str, Any]) -> str:
    if not overall:
        return ""
    lines: List[str] = ["[Overall metrics]"]

    # We favour robustness over exact field names; missing keys are skipped.
    def _fmt_number(key: str) -> str | None:
        value = overall.get(key)
        if value is None:
            return None
        # Avoid scientific notation for typical business numbers.
        try:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return f"{value:,.2f}"
            return str(value)
        except Exception:
            return str(value)

    for label, key in [
        ("Total revenue", "total_revenue"),
        ("Total cost", "total_cost"),
        ("Gross profit", "gross_profit"),
        ("Gross margin %", "gross_margin_pct"),
        ("Number of transactions", "n_transactions"),
        ("Average order value", "avg_order_value"),
    ]:
        formatted = _fmt_number(key)
        if formatted is not None:
            lines.append(f"- {label}: {formatted}")

    date_min = overall.get("date_min")
    date_max = overall.get("date_max")
    if date_min is not None or date_max is not None:
        lines.append(f"- Date range: {date_min} to {date_max}")

    return "\n".join(lines)


def _format_time_series_section(ts: Any, max_rows: int) -> str:
    if not isinstance(ts, pd.DataFrame) or ts.empty:
        return ""
    # We only show the last `max_rows` rows to keep things compact.
    tail = ts.tail(max_rows)
    # Expect a "period" column; if missing, try to use the index.
    if "period" in tail.columns:
        period_series = tail["period"]
    else:
        period_series = tail.index

    cols: List[str] = [
        c for c in ["revenue", "cost", "gross_profit", "gross_margin_pct"] if c in tail.columns
    ]

    lines: List[str] = ["[Revenue time series (last rows)]"]
    header = ["period"] + cols
    lines.append(" | ".join(header))
    lines.append(" | ".join(["-" * len(h) for h in header]))

    for idx, row in tail.iterrows():
        period = period_series.loc[idx]
        try:
            if hasattr(period, "date"):
                period_str = str(period.date())
            else:
                period_str = str(period)
        except Exception:
            period_str = str(period)
        row_values: List[str] = [period_str]
        for c in cols:
            value = row.get(c)
            if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                row_values.append("-")
            else:
                try:
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        row_values.append(f"{value:,.2f}")
                    else:
                        row_values.append(str(value))
                except Exception:
                    row_values.append(str(value))
        lines.append(" | ".join(row_values))

    return "\n".join(lines)


def _format_segments_section(segments: Mapping[str, Any], max_per_type: int) -> str:
    if not segments:
        return ""
    lines: List[str] = ["[Top segments]"]
    for seg_name, seg_df in segments.items():
        if not isinstance(seg_df, pd.DataFrame) or seg_df.empty:
            continue
        # Limit to top N rows as returned by analytics.compute_segment_revenue
        head = seg_df.head(max_per_type)
        # First column is the segment label (e.g. category, product, customer_id, channel)
        # We derive it from the DataFrame columns.
        segment_cols = [
            c
            for c in head.columns
            if c
            not in {
                "revenue",
                "cost",
                "gross_profit",
                "gross_margin_pct",
                "order_count",
                "revenue_share_pct",
            }
        ]
        if not segment_cols:
            # Fallback: use the index name if we have to.
            segment_cols = [head.index.name or seg_name]
        segment_col = segment_cols[0]

        lines.append(f"- Segment type: {seg_name}")
        for idx, row in head.iterrows():
            label = row.get(segment_col) if segment_col in row else getattr(idx, "name", idx)
            revenue = row.get("revenue")
            share = row.get("revenue_share_pct")
            pieces: List[str] = [f"  • {segment_col}={label!r}"]
            if revenue is not None:
                try:
                    pieces.append(f"revenue={float(revenue):,.2f}")
                except Exception:
                    pieces.append(f"revenue={revenue}")
            if share is not None:
                try:
                    pieces.append(f"share={float(share):.1f}%")
                except Exception:
                    pieces.append(f"share={share}%")
            lines.append(", ".join(pieces))
    return "\n".join(lines)


def _format_trend_section(trend: Mapping[str, Any]) -> str:
    if not trend:
        return ""
    lines: List[str] = ["[Monthly trend summary]"]
    direction = trend.get("direction")
    if direction:
        lines.append(f"- Direction: {direction}")
    for label, key in [
        ("Current period", "current_period"),
        ("Current revenue", "current_revenue"),
        ("Previous period", "previous_period"),
        ("Previous revenue", "previous_revenue"),
        ("Absolute change", "absolute_change"),
        ("Percent change", "percent_change"),
    ]:
        value = trend.get(key)
        if value is not None:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def _format_anomaly_section(anomaly: Mapping[str, Any]) -> str:
    if not anomaly:
        return ""
    lines: List[str] = ["[Recent anomaly check]"]
    for label, key in [
        ("Has anomaly", "has_anomaly"),
        ("Last date", "last_date"),
        ("Last revenue", "last_revenue"),
        ("Expected mean", "expected_mean"),
        ("Expected std", "expected_std"),
        ("Z-score", "z_score"),
    ]:
        value = anomaly.get(key)
        if value is not None:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def summarize_metrics_bundle(
    metrics_bundle: Mapping[str, Any],
    options: MetricsSummaryOptions | None = None,
) -> str:
    """Convert a metrics bundle into a compact text summary.

    This is intended to be passed into LLM prompts as *structured context*.
    It deliberately avoids free-form narrative so that the LLM remains in
    charge of phrasing, while we stay in control of the underlying numbers.

    Parameters
    ----------
    metrics_bundle:
        The dictionary produced by analytics.build_metrics_bundle.
    options:
        Optional MetricsSummaryOptions controlling truncation limits.

    Returns
    -------
    str
        Plain-text summary of the metrics, truncated to options.max_context_chars.
    """
    opts = options or MetricsSummaryOptions()

    overall = metrics_bundle.get("overall") or {}
    time_series = metrics_bundle.get("time_series")
    segments = metrics_bundle.get("segments") or {}
    monthly_trend = metrics_bundle.get("monthly_trend") or {}
    anomaly = metrics_bundle.get("anomaly") or {}

    parts: List[str] = []
    overall_txt = _format_overall_section(overall)
    if overall_txt:
        parts.append(overall_txt)

    ts_txt = _format_time_series_section(time_series, max_rows=opts.max_time_series_rows)
    if ts_txt:
        parts.append(ts_txt)

    seg_txt = _format_segments_section(segments, max_per_type=opts.max_segments_per_type)
    if seg_txt:
        parts.append(seg_txt)

    trend_txt = _format_trend_section(monthly_trend)
    if trend_txt:
        parts.append(trend_txt)

    anomaly_txt = _format_anomaly_section(anomaly)
    if anomaly_txt:
        parts.append(anomaly_txt)

    combined = "\n\n".join(parts)
    if len(combined) <= opts.max_context_chars:
        return combined

    truncated = combined[: opts.max_context_chars].rstrip()
    return truncated + "\n\n[Context truncated for length]"


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


def build_system_prompt_for_report(language: str = "en") -> str:
    """Build a system prompt describing the report-writing role and constraints."""
    lang = _normalise_language(language)
    if lang == "id":
        return (
            "Anda adalah asisten analis keuangan yang membantu menjelaskan kinerja bisnis.\n"
            "Semua nilai uang menggunakan Rupiah Indonesia (IDR) dan harus ditulis dengan format 'Rp'.\n"
            "Gunakan hanya metrik dan ringkasan yang diberikan dalam konteks. "
            "Jangan mengarang transaksi baru, tanggal yang tidak ada, atau detail pelanggan.\n"
            "Tuliskan jawaban dalam Bahasa Indonesia yang jelas, terstruktur, dan mudah dipahami "
            "oleh manajer bisnis non-teknis."
        )
    # Default: English
    return (
        "You are a financial analyst assistant helping to explain business performance.\n"
        "All monetary values are in Indonesian Rupiah (IDR) and must be shown using the 'Rp' format.\n"
        "Use only the metrics and summaries provided in the context. "
        "Do not invent new transactions, dates, or customer details.\n"
        "Write your response in clear, structured English suitable for a non-technical business manager."
    )


def build_system_prompt_for_chat(language: str = "en") -> str:
    """Build a system prompt for interactive Q&A over the metrics bundle."""
    lang = _normalise_language(language)
    if lang == "id":
        return (
            "Anda adalah asisten analis keuangan yang menjawab pertanyaan tentang data penjualan.\n"
            "Gunakan hanya metrik yang diberikan dalam konteks. "
            "Jika informasi tidak tersedia, jelaskan keterbatasannya dengan jujur.\n"
            "Jawaban harus dalam Bahasa Indonesia yang singkat, jelas, dan membantu."
        )
    return (
        "You are a financial analyst assistant answering questions about sales data.\n"
        "Use only the metrics provided in the context. "
        "If some information is not available, explain the limitation honestly instead of guessing.\n"
        "Answer in concise, clear English."
    )


# ---------------------------------------------------------------------------
# User prompts
# ---------------------------------------------------------------------------


def build_user_prompt_for_report(
    metrics_bundle: Mapping[str, Any],
    language: str = "en",
    *,
    template_report: str | None = None,
    options: MetricsSummaryOptions | None = None,
) -> str:
    """Build the user prompt used when asking the LLM to write a full report.

    Parameters
    ----------
    metrics_bundle:
        The structured metrics output from analytics.build_metrics_bundle.
    language:
        Target output language ("en" or "id"). This is also encoded in the
        instructions so the model knows how to respond.
    template_report:
        Optional existing template-based report text. In 'hybrid' mode this is
        passed as additional context that the model may refine or expand.
    options:
        Optional MetricsSummaryOptions controlling truncation limits.

    Returns
    -------
    str
        A user prompt string ready to be sent alongside the system prompt.
    """
    lang = _normalise_language(language)
    opts = options or MetricsSummaryOptions()
    metrics_summary = summarize_metrics_bundle(metrics_bundle, options=opts)

    if lang == "id":
        instructions = (
            "Gunakan ringkasan metrik berikut untuk menulis laporan naratif tentang kinerja keuangan.\n"
            "Fokus pada: gambaran umum, segmen yang paling berkontribusi, tren dari waktu ke waktu, "
            "dan adanya anomali terbaru.\n"
            "Jelaskan angka-angka utama, berikan interpretasi singkat, dan berikan insight yang bisa "
            "dipakai manajemen untuk mengambil keputusan.\n"
            "Pastikan seluruh laporan ditulis dalam Bahasa Indonesia."
        )
    else:
        instructions = (
            "Use the following metrics summary to write a narrative report about business performance.\n"
            "Focus on: overall overview, the most important segments, trends over time, "
            "and any recent anomalies.\n"
            "Explain the key numbers, provide brief interpretation, and highlight insights "
            "that could help management make decisions.\n"
            "Make sure the entire report is written in English."
        )

    lines: List[str] = [instructions, "", "Structured metrics summary:", metrics_summary]

    if template_report:
        lines.extend(
            [
                "",
                "Existing template-based report (you may refine and expand this, but stay faithful to the numbers):",
                template_report.strip(),
            ]
        )

    return "\n".join(lines)


def _format_available_commands(available_commands: Sequence[str], language: str) -> str:
    if not available_commands:
        return ""
    lang = _normalise_language(language)
    if lang == "id":
        header = (
            "Perintah CLI yang tersedia (untuk referensi, Anda tidak perlu mengulang sintaks persisnya):"
        )
        descriptions = {
            "summary": "ringkasan singkat performa keseluruhan",
            "overview": "gambaran umum metrik utama",
            "top categories": "kategori dengan kontribusi pendapatan terbesar",
            "top products": "produk dengan kontribusi pendapatan terbesar",
            "top customers": "pelanggan dengan kontribusi pendapatan terbesar",
            "top channels": "saluran penjualan dengan kontribusi pendapatan terbesar",
            "trend": "tren pendapatan dari waktu ke waktu",
            "anomaly": "pemeriksaan anomali pendapatan harian terakhir",
            "time series": "deret waktu pendapatan agregat",
        }
    else:
        header = "Available CLI-style commands (for reference; you do not need to repeat the exact syntax):"
        descriptions = {
            "summary": "short summary of overall performance",
            "overview": "overview of key metrics",
            "top categories": "categories contributing most revenue",
            "top products": "products contributing most revenue",
            "top customers": "customers contributing most revenue",
            "top channels": "sales channels contributing most revenue",
            "trend": "trend in revenue over time",
            "anomaly": "recent daily revenue anomaly check",
            "time series": "aggregated revenue time series",
        }

    lines: List[str] = [header]
    for cmd in available_commands:
        desc = descriptions.get(cmd, "")
        if desc:
            lines.append(f"- {cmd}: {desc}")
        else:
            lines.append(f"- {cmd}")
    return "\n".join(lines)


def build_user_prompt_for_question(
    question: str,
    metrics_bundle: Mapping[str, Any],
    language: str = "en",
    *,
    available_commands: Sequence[str] | None = None,
    options: MetricsSummaryOptions | None = None,
) -> str:
    """Build the user prompt used for ad-hoc Q&A over the metrics bundle.

    Parameters
    ----------
    question:
        The user's natural-language question.
    metrics_bundle:
        Structured metrics context.
    language:
        Target output language ("en" or "id").
    available_commands:
        Optional list of CLI-style commands that the model may reference in its
        explanation (e.g. to suggest that the user runs "trend" or "top categories").
    options:
        Optional MetricsSummaryOptions controlling truncation limits.

    Returns
    -------
    str
        A user prompt string ready to be sent alongside the chat system prompt.
    """
    lang = _normalise_language(language)
    opts = options or MetricsSummaryOptions()
    metrics_summary = summarize_metrics_bundle(metrics_bundle, options=opts)

    if lang == "id":
        instructions = (
            "Jawablah pertanyaan pengguna berdasarkan ringkasan metrik berikut.\n"
            "Gunakan hanya informasi yang ada di ringkasan ini. Jika data yang diminta tidak tersedia, "
            "jelaskan keterbatasannya secara jujur.\n"
            "Jika relevan, Anda boleh menyebutkan jenis informasi yang kira-kira akan ditampilkan "
            "oleh perintah CLI tertentu (misalnya 'trend', 'top categories'), namun jangan mengarang angka baru."
        )
        question_label = "Pertanyaan pengguna"
    else:
        instructions = (
            "Answer the user's question based on the following metrics summary.\n"
            "Use only the information present in this summary. If the requested detail is not available, "
            "explain that limitation honestly instead of guessing.\n"
            "If helpful, you may refer to what a CLI-style command (such as 'trend' or 'top categories') "
            "would show, but do not invent new numbers."
        )
        question_label = "User question"

    lines: List[str] = [instructions]

    if available_commands:
        lines.extend(["", _format_available_commands(available_commands, language)])

    lines.extend(
        [
            "",
            "Structured metrics summary:",
            metrics_summary,
            "",
            f"{question_label}:",
            question.strip(),
        ]
    )

    return "\n".join(lines)