from clincot.evaluation.vqa_metrics import compute_vqa_metrics


def test_vqa_metric_basic():
    rows = [
        {"pred": "yes", "gt": "yes", "answer_type": "closed"},
        {"pred": "no", "gt": "yes", "answer_type": "closed"},
    ]
    m = compute_vqa_metrics(rows)
    assert "exact_match" in m
    assert 0.0 <= m["exact_match"] <= 1.0
