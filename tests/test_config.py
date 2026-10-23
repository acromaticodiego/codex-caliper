from banana_vision.config import AppConfig


def test_effective_height_subtracts_marker_thickness():
    cfg = AppConfig(marker_thickness_mm=3.0, working_distance_mm=270.0)
    assert cfg.effective_height_mm == 267.0


def test_roundtrip_preserves_marker_layout(tmp_path):
    cfg = AppConfig(marker_thickness_mm=2.5, marker_layout={0: (0.0, 0.0), 1: (300.0, 0.0)})
    cfg.save(tmp_path / "c.json")
    assert AppConfig.load(tmp_path / "c.json").marker_layout == cfg.marker_layout
