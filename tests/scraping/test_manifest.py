import re
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from databricks_docs_vectordb.scraping.manifest import (
    SCHEMA_VERSION,
    AddResult,
    Manifest,
)

AWS = "https://docs.databricks.com/aws/en/"
API = "https://docs.databricks.com/api/"
A, B, C = AWS + "delta/", AWS + "delta/merge", AWS + "sql/"
T1, T2, T3 = "2026-09-20T01:00:00Z", "2026-09-20T02:00:00Z", "2026-09-20T03:00:00Z"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "nested" / "manifest.sqlite"


@pytest.fixture
def manifest(db_path):
    with Manifest(db_path) as m:
        yield m


def fetch_ok(manifest: Manifest, url: str, *, etag="e1", now=T1, **extra) -> None:
    manifest.record_ok(
        url,
        http_status=200,
        sha256="ab" * 32,
        size_bytes=1234,
        etag=etag,
        last_modified="Fri, 18 Sep 2026 21:33:20 GMT",
        now=now,
        **extra,
    )


# -- schema and persistence ---------------------------------------------


def test_opening_creates_the_database_and_its_parent_directory(db_path):
    assert not db_path.parent.exists()
    with Manifest(db_path) as m:
        assert m.counts() == Counter()
    assert db_path.exists()


def test_reopening_from_disk_returns_the_same_counts(db_path):
    with Manifest(db_path) as m:
        m.add_pending("aws-en", [A, B, C])
        fetch_ok(m, A)
        m.record_gone(B)
        before = m.counts()
    with Manifest(db_path) as m:
        assert m.counts() == before == Counter(ok=1, gone=1, pending=1)
        assert m.get(A).etag == "e1"


def test_schema_version_is_recorded_and_checked(db_path):
    with Manifest(db_path):
        pass
    conn = sqlite3.connect(db_path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="schema version 99"):
        Manifest(db_path)


def test_status_column_rejects_unknown_values(manifest):
    manifest.add_pending("aws-en", [A])
    with pytest.raises(sqlite3.IntegrityError):
        manifest._conn.execute("UPDATE pages SET status = 'bogus'")


def test_database_uses_wal_so_status_can_read_during_a_crawl(db_path):
    with Manifest(db_path) as writer, Manifest(db_path) as reader:
        assert writer._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        writer.add_pending("aws-en", [A, B])
        assert reader.counts() == Counter(pending=2)


# -- add_pending ---------------------------------------------------------


def test_add_pending_creates_pending_rows_with_derived_fields(manifest):
    result = manifest.add_pending("aws-en", [A, B])
    assert result == AddResult(added=2, revived=0)

    page = manifest.get(A)
    assert page.source == "aws-en"
    assert page.rel_path == "aws/en/delta/index.html"
    assert page.status == "pending"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", page.first_seen_at)
    assert (page.etag, page.sha256, page.fetched_at, page.checked_at) == (None,) * 4
    assert (page.attempts, page.error) == (0, None)
    assert manifest.get(B).rel_path == "aws/en/delta/merge.html"


def test_add_pending_twice_is_idempotent_and_preserves_existing_state(manifest):
    manifest.add_pending("aws-en", [A, B])
    fetch_ok(manifest, A, etag="keep-me")
    first_seen = manifest.get(A).first_seen_at
    before = manifest.counts()

    result = manifest.add_pending("aws-en", [A, B])

    assert result == AddResult(added=0, revived=0)
    assert manifest.counts() == before == Counter(ok=1, pending=1)
    page = manifest.get(A)
    assert (page.status, page.etag, page.first_seen_at) == ("ok", "keep-me", first_seen)


def test_add_pending_ignores_duplicates_within_one_call(manifest):
    assert manifest.add_pending("aws-en", [A, A, A]) == AddResult(1, 0)
    assert manifest.counts() == Counter(pending=1)


def test_add_pending_rejects_urls_outside_the_sources_and_adds_nothing(manifest):
    bad = "https://docs.databricks.com/gcp/en/delta/"
    with pytest.raises(ValueError, match="outside every configured source"):
        manifest.add_pending("aws-en", [A, bad])
    assert manifest.counts() == Counter()


def test_add_pending_is_all_or_nothing_when_two_urls_share_a_file(manifest):
    # `.../a/index` and `.../a/` both map to aws/en/a/index.html
    clash = [AWS + "a/", AWS + "a/index"]
    with pytest.raises(sqlite3.IntegrityError):
        manifest.add_pending("aws-en", [A, *clash])
    assert manifest.counts() == Counter()


# -- mark_removed --------------------------------------------------------


def test_mark_removed_flags_urls_missing_from_the_sitemap(manifest):
    manifest.add_pending("aws-en", [A, B, C])
    fetch_ok(manifest, B, etag="kept")

    assert manifest.mark_removed("aws-en", [A, C]) == 1

    assert manifest.get(B).status == "removed"
    assert manifest.get(B).etag == "kept"  # validators survive removal
    assert manifest.get(A).status == manifest.get(C).status == "pending"
    assert manifest.counts() == Counter(pending=2, removed=1)


def test_mark_removed_is_idempotent(manifest):
    manifest.add_pending("aws-en", [A, B])
    assert manifest.mark_removed("aws-en", [A]) == 1
    assert manifest.mark_removed("aws-en", [A]) == 0


def test_mark_removed_only_touches_the_given_source(manifest):
    manifest.add_pending("aws-en", [A, B])
    manifest.add_pending("api", [API])
    manifest.mark_removed("aws-en", [A])
    assert manifest.get(API).status == "pending"
    assert manifest.get(B).status == "removed"


def test_mark_removed_refuses_an_empty_sitemap(manifest):
    manifest.add_pending("aws-en", [A])
    with pytest.raises(ValueError, match="empty"):
        manifest.mark_removed("aws-en", [])
    assert manifest.get(A).status == "pending"


def test_a_removed_url_that_reappears_returns_to_pending_with_its_validators(manifest):
    manifest.add_pending("aws-en", [A, B])
    fetch_ok(manifest, B, etag="still-valid")
    manifest.mark_removed("aws-en", [A])

    result = manifest.add_pending("aws-en", [A, B])

    assert result == AddResult(added=0, revived=1)
    page = manifest.get(B)
    assert (page.status, page.etag) == ("pending", "still-valid")


def test_revalidating_a_revived_page_with_a_304_makes_it_ok_again(manifest):
    manifest.add_pending("aws-en", [A, B])
    fetch_ok(manifest, B)
    manifest.mark_removed("aws-en", [A])
    manifest.add_pending("aws-en", [A, B])
    manifest.record_not_modified(B, now=T2)
    assert manifest.get(B).status == "ok"


# -- next_batch ----------------------------------------------------------


def make_mixed(manifest: Manifest) -> None:
    urls = [AWS + n for n in ("p1", "p2", "ok1", "ok2", "gone1", "failed1", "removed1")]
    manifest.add_pending("aws-en", urls)
    fetch_ok(manifest, AWS + "ok1", now=T2)
    fetch_ok(manifest, AWS + "ok2", now=T1)
    manifest.record_gone(AWS + "gone1")
    manifest.record_failed(AWS + "failed1", error="boom", attempts=5)
    manifest.mark_removed("aws-en", [u for u in urls if not u.endswith("removed1")])


def test_next_batch_returns_only_pending_by_default(manifest):
    make_mixed(manifest)
    assert [p.url for p in manifest.next_batch()] == [AWS + "p1", AWS + "p2"]


def test_next_batch_with_refresh_also_returns_ok_rows(manifest):
    make_mixed(manifest)
    urls = [p.url for p in manifest.next_batch(refresh=True)]
    # pending first, then ok rows oldest-checked first; gone/failed/removed never
    assert urls == [AWS + "p1", AWS + "p2", AWS + "ok2", AWS + "ok1"]


def test_next_batch_limit(manifest):
    make_mixed(manifest)
    assert len(manifest.next_batch(limit=1)) == 1
    assert len(manifest.next_batch(refresh=True, limit=3)) == 3


def test_next_batch_stale_before_skips_ok_rows_already_checked_this_run(manifest):
    make_mixed(manifest)  # ok2 checked at T1, ok1 at T2
    stale = manifest.next_batch(refresh=True, stale_before=T2)
    assert [p.url for p in stale] == [AWS + "p1", AWS + "p2", AWS + "ok2"]


def test_next_batch_on_an_empty_manifest_is_empty(manifest):
    assert manifest.next_batch() == []
    assert manifest.next_batch(refresh=True) == []


def test_next_batch_returns_full_rows(manifest):
    manifest.add_pending("aws-en", [A])
    fetch_ok(manifest, A)
    manifest.add_pending("aws-en", [B])
    (page,) = manifest.next_batch()
    assert page.url == B
    (ok_page,) = manifest.next_batch(refresh=True, limit=10)[1:]
    assert (ok_page.url, ok_page.etag) == (A, "e1")


# -- record_* ------------------------------------------------------------


def test_record_ok_stores_the_result_and_clears_errors(manifest):
    manifest.add_pending("aws-en", [A])
    manifest.record_failed(A, error="timeout", attempts=3)
    manifest.requeue_failed()

    fetch_ok(manifest, A, now=T1)

    page = manifest.get(A)
    assert (page.status, page.http_status) == ("ok", 200)
    assert (page.sha256, page.size_bytes) == ("ab" * 32, 1234)
    assert (page.etag, page.last_modified) == ("e1", "Fri, 18 Sep 2026 21:33:20 GMT")
    assert (page.fetched_at, page.checked_at) == (T1, T1)
    assert (page.attempts, page.error) == (0, None)


def test_record_ok_keeps_final_url_only_when_it_differs(manifest):
    manifest.add_pending("aws-en", [A, B])
    fetch_ok(manifest, A, final_url=A)
    fetch_ok(manifest, B, final_url=AWS + "elsewhere")
    assert manifest.get(A).final_url is None
    assert manifest.get(B).final_url == AWS + "elsewhere"


def test_record_ok_replaces_validators_on_a_changed_page(manifest):
    manifest.add_pending("aws-en", [A])
    fetch_ok(manifest, A, etag="old", now=T1)
    manifest.record_ok(
        A, http_status=200, sha256="cd" * 32, size_bytes=99, etag="new", now=T2
    )
    page = manifest.get(A)
    assert (page.etag, page.sha256, page.fetched_at) == ("new", "cd" * 32, T2)


def test_record_not_modified_only_advances_checked_at(manifest):
    manifest.add_pending("aws-en", [A])
    fetch_ok(manifest, A, now=T1)
    manifest.record_not_modified(A, now=T3)
    page = manifest.get(A)
    assert (page.status, page.http_status) == ("ok", 304)
    assert (page.fetched_at, page.checked_at) == (T1, T3)
    assert (page.etag, page.sha256) == ("e1", "ab" * 32)


def test_record_not_modified_adopts_new_validators_and_keeps_omitted_ones(manifest):
    manifest.add_pending("aws-en", [A])
    fetch_ok(manifest, A, etag="old-etag", now=T1)
    manifest.record_not_modified(A, etag="new-etag", now=T2)
    page = manifest.get(A)
    assert page.etag == "new-etag"
    assert page.last_modified == "Fri, 18 Sep 2026 21:33:20 GMT"  # omitted: kept
    manifest.record_not_modified(
        A, last_modified="Sat, 19 Sep 2026 00:00:00 GMT", now=T3
    )
    page = manifest.get(A)
    assert (page.etag, page.last_modified) == (
        "new-etag",
        "Sat, 19 Sep 2026 00:00:00 GMT",
    )


def test_record_gone_and_it_is_never_fetched_again(manifest):
    manifest.add_pending("aws-en", [A])
    manifest.record_gone(A, http_status=410, now=T1)
    page = manifest.get(A)
    assert (page.status, page.http_status, page.checked_at) == ("gone", 410, T1)
    assert manifest.next_batch(refresh=True) == []


def test_record_failed_marks_a_new_page_failed(manifest):
    manifest.add_pending("aws-en", [A])
    manifest.record_failed(A, error="HTTP 500", attempts=5, http_status=500)
    page = manifest.get(A)
    assert (page.status, page.error, page.attempts, page.http_status) == (
        "failed",
        "HTTP 500",
        5,
        500,
    )
    assert manifest.next_batch(refresh=True) == []


def test_record_failed_keeps_a_page_with_a_good_body_ok(manifest):
    manifest.add_pending("aws-en", [A])
    fetch_ok(manifest, A, now=T1)
    manifest.record_failed(A, error="timeout", attempts=5)
    page = manifest.get(A)
    assert page.status == "ok"
    assert (page.error, page.attempts) == ("timeout", 5)
    assert page.sha256 == "ab" * 32
    assert page.checked_at == T1  # not confirmed, so not advanced
    assert [p.url for p in manifest.next_batch(refresh=True)] == [A]


def test_requeue_failed_returns_failed_rows_to_pending(manifest):
    manifest.add_pending("aws-en", [A, B, C])
    manifest.record_failed(A, error="x", attempts=5)
    manifest.record_failed(B, error="y", attempts=5)
    manifest.record_gone(C)

    assert manifest.requeue_failed() == 2

    assert manifest.counts() == Counter(pending=2, gone=1)
    page = manifest.get(A)
    assert (page.status, page.error, page.attempts) == ("pending", None, 0)
    assert manifest.requeue_failed() == 0


@pytest.mark.parametrize(
    "call",
    [
        lambda m: fetch_ok(m, A),
        lambda m: m.record_not_modified(A),
        lambda m: m.record_gone(A),
        lambda m: m.record_failed(A, error="x", attempts=1),
    ],
    ids=["ok", "not_modified", "gone", "failed"],
)
def test_recording_an_unknown_url_raises_keyerror(manifest, call):
    with pytest.raises(KeyError):
        call(manifest)


def test_default_timestamps_are_utc_iso8601(manifest):
    manifest.add_pending("aws-en", [A])
    manifest.record_ok(A, http_status=200, sha256="ab" * 32, size_bytes=1)
    page = manifest.get(A)
    for stamp in (page.first_seen_at, page.fetched_at, page.checked_at):
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", stamp)


# -- inspection ----------------------------------------------------------


def test_get_returns_none_for_an_unknown_url(manifest):
    assert manifest.get(A) is None


def test_counts_and_counts_by_source(manifest):
    manifest.add_pending("aws-en", [A, B, C])
    manifest.add_pending("api", [API])
    fetch_ok(manifest, A)
    manifest.record_gone(API)

    assert manifest.counts() == Counter(ok=1, pending=2, gone=1)
    assert manifest.counts_by_source() == {
        "aws-en": Counter(ok=1, pending=2),
        "api": Counter(gone=1),
    }


def test_pages_with_status_lists_rows_ordered_by_url_with_an_optional_limit(manifest):
    manifest.add_pending("aws-en", [C, A, B])
    manifest.record_failed(C, error="x", attempts=1)
    manifest.record_failed(A, error="y", attempts=1)
    assert [p.url for p in manifest.pages_with_status("failed")] == [A, C]
    assert [p.url for p in manifest.pages_with_status("failed", limit=1)] == [A]
    assert [p.url for p in manifest.pages_with_status("pending")] == [B]
    assert manifest.pages_with_status("ok") == []
