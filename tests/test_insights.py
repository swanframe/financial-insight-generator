from fig.insights import generate_full_report


def test_generate_full_report_contains_sections(metrics_bundle):
    report_text = generate_full_report(metrics_bundle)

    assert "FINANCIAL INSIGHT REPORT" in report_text
    assert "1. Overview" in report_text
    assert "2. Segment Highlights" in report_text
    assert "3. Trend Analysis" in report_text
    assert "4. Daily Anomaly Check" in report_text