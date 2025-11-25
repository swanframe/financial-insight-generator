from fig.insights import generate_full_report


def test_generate_full_report_contains_sections(metrics_bundle):
    """English report should contain the main section headings.

    This relies on the default language='en' behavior so that existing
    callers that do not pass a language keep working.
    """
    report_text = generate_full_report(metrics_bundle)

    assert "FINANCIAL INSIGHT REPORT" in report_text
    assert "1. Overview" in report_text
    assert "2. Segment Highlights" in report_text
    assert "3. Trend Analysis" in report_text
    assert "4. Daily Anomaly Check" in report_text


def test_generate_full_report_indonesian_headings(metrics_bundle):
    """Indonesian report should use localized headings.

    This is a light smoke test to ensure that the i18n layer is wired
    correctly and that 'id' does not silently fall back to English.
    """
    report_text_id = generate_full_report(metrics_bundle, language="id")

    assert "LAPORAN INSIGHT KEUANGAN" in report_text_id
    assert "1. Gambaran Umum" in report_text_id
    assert "2. Sorotan Segmen" in report_text_id
    assert "3. Analisis Tren" in report_text_id
    assert "4. Pemeriksaan Anomali Harian" in report_text_id