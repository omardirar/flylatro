from pathlib import Path

import numpy as np
import pytest


def test_primary_builder_keeps_weak_anatomical_pairs_and_preserves_existing_artifacts(tmp_path: Path, monkeypatch):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from flylatro.fly import build_flywire as builder
    from flylatro.fly.flywire_artifact import sha256_file

    # Retain the published sensory/descending census checks in the builder.
    classes = ["ALPN"] * 16938 + ["Kenyon_Cell", "MBON", "MBON", "DAN", "DAN"] + ["DN"] * 1305
    supers = ["sensory"] * 16938 + ["central"] * 5 + ["descending"] * 1305
    roots = np.arange(1000, 1000 + len(classes))
    kc, mbon1, mbon2 = roots[16938:16941]
    sources = tmp_path / "source"
    sources.mkdir()
    tables = {
        "neurons.csv.gz": pd.DataFrame({"root_id": roots, "nt_type": "ACH"}),
        "classification.csv.gz": pd.DataFrame({"root_id": roots, "flow": "intrinsic", "super_class": supers,
            "class": classes, "sub_class": "", "side": "left"}),
        "consolidated_cell_types.csv.gz": pd.DataFrame({"root_id": roots,
            "primary_type": ["ALPN"] * 16938 + ["KC", "MBON01", "MBON02", "PAM01", "PPL101"] + ["DN"] * 1305}),
        "coordinates.csv.gz": pd.DataFrame({"root_id": roots, "position": "[0 0 0]"}),
        "connections_princeton_no_threshold.csv.gz": pd.DataFrame({"pre_root_id": [kc, kc],
            "post_root_id": [mbon1, mbon2], "syn_count": [5, 2], "nt_type": "ACH"}),
        "connections_princeton.csv.gz": pd.DataFrame({"pre_root_id": [kc],
            "post_root_id": [mbon1], "syn_count": [5], "nt_type": "ACH"}),
    }
    for name, frame in tables.items():
        frame.to_csv(sources / name, index=False, compression="gzip")
    monkeypatch.setattr(builder, "REFERENCE_SHA256", {name: sha256_file(sources / name) for name in tables})
    monkeypatch.setattr(builder, "EXPECTED_NEURONS", len(roots))
    monkeypatch.setattr(builder, "EXPECTED_KENYON_CELLS", 1)
    monkeypatch.setattr(builder, "EXPECTED_CONNECTION_PAIRS", 1)
    complete, _ = builder.build(sources, tmp_path / "complete")
    historical, _ = builder.build(sources, tmp_path / "historical", connection_source="threshold5")
    with np.load(complete) as data:
        assert len(data["kc_mbon_edge_indices"]) == 2
        assert sorted(data["signed_synapse_counts"].tolist()) == [2, 5]
    with np.load(historical) as data:
        assert len(data["kc_mbon_edge_indices"]) == 1
    before = sha256_file(historical)
    with pytest.raises(FileExistsError, match="not empty"):
        builder.build(sources, tmp_path / "historical")
    assert sha256_file(historical) == before
