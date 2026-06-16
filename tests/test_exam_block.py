"""ExamBlock dataclass."""
from __future__ import annotations
from records import ExamBlock


def test_exam_block_construction():
    eb = ExamBlock(
        name="resilience",
        config={"mod_type": "FLICKER", "s_max": 0.9, "n_episodes": 5, "max_iters": 6},
        raw={"aurc": 0.7, "s_half": 0.4, "s_max": 0.9, "cliff_slope": 8.0, "points": []},
        formula="success := aurc; adapt_score := s_half / s_max",
    )
    assert eb.name == "resilience"
    assert eb.config["s_max"] == 0.9
    assert eb.raw["aurc"] == 0.7
    assert "aurc" in eb.formula
