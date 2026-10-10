"""scripts/snapshot_qdrant.py, loaded from its file (scripts/ is not a
package)."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
from qdrant_client import models

SCRIPT = Path(__file__).parents[2] / "scripts" / "snapshot_qdrant.py"
spec = importlib.util.spec_from_file_location("snapshot_qdrant", SCRIPT)
assert spec is not None and spec.loader is not None
snapshot_qdrant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot_qdrant)


@pytest.fixture
def prefixed(settings, name, tmp_path, monkeypatch):
    """The script's settings, with the throwaway `name` as the collection
    prefix and the data dir in tmp_path."""
    forced = replace(settings, collection_prefix=name, data_dir=tmp_path)
    monkeypatch.setattr(
        snapshot_qdrant.Settings, "load", staticmethod(lambda **kwargs: forced)
    )
    return forced


def make_collection(client, name: str, points: int = 3) -> None:
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        name,
        points=[models.PointStruct(id=i, vector=[1.0, i]) for i in range(points)],
        wait=True,
    )


@pytest.mark.network
def test_every_collection_under_the_prefix_is_saved_to_data(
    prefixed, client, name, restore, capsys
):
    make_collection(client, f"{name}_a_2", points=3)
    make_collection(client, f"{name}x")  # not `<prefix>_`: left alone
    assert snapshot_qdrant.main([]) == 0
    [path] = prefixed.snapshots_dir.iterdir()
    assert path.name.startswith(f"{name}_a_2")
    assert f"saved {path}" in capsys.readouterr().out
    assert client.list_snapshots(f"{name}_a_2") == []
    assert restore(path, f"{name}_restored") == 3


@pytest.mark.network
def test_named_collections_go_to_out(prefixed, client, name, tmp_path):
    make_collection(client, f"{name}_one")
    make_collection(client, f"{name}_two")
    out = tmp_path / "backups"
    assert snapshot_qdrant.main(["--collection", f"{name}_two", "--out", str(out)]) == 0
    assert [p.name.startswith(f"{name}_two") for p in out.iterdir()] == [True]


@pytest.mark.network
def test_an_unknown_collection_is_an_error(prefixed, capsys):
    assert snapshot_qdrant.main(["--collection", "no_such_collection"]) == 1
    assert "no such collection: no_such_collection" in capsys.readouterr().err


@pytest.mark.network
def test_no_collections_under_the_prefix_is_an_error(prefixed, name, capsys):
    assert snapshot_qdrant.main([]) == 1
    assert f"no {name}_* collections" in capsys.readouterr().err
    assert not prefixed.snapshots_dir.exists()


def test_qdrant_unreachable_is_an_error(settings, tmp_path, monkeypatch, capsys):
    closed = replace(settings, url="http://localhost:1", data_dir=tmp_path)
    monkeypatch.setattr(
        snapshot_qdrant.Settings, "load", staticmethod(lambda **kwargs: closed)
    )
    assert snapshot_qdrant.main([]) == 1
    assert (
        "error: Qdrant at http://localhost:1: ConnectError" in capsys.readouterr().err
    )
