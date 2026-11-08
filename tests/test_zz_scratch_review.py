import pytest
from tests.test_claims import two_computers, on  # noqa: F401
from tests.fakes import annotate
from omero_annotate_ai import claims


def test_note_on_other_hosts_session_is_kept(tmp_path, on):
    a, b, source, omero = two_computers(tmp_path, on)
    on("a")
    s = a.start_session(batch_size=1)
    sid = s.session
    s.finish("from a")
    on("b")
    b._merge_remote_sessions()
    b.set_session_note(sid, "edited on b")
    print([ (r.id, r.host, r.note) for r in b.config.sessions])
    remote = b.omero_writes.read_config(b.conn, b.table_id)
    print([(r.id, r.host, r.note) for r in remote.sessions])
    assert b._session(sid).note == "edited on b"


def test_session_id_collision(tmp_path, on):
    a, b, source, omero = two_computers(tmp_path, on)
    on("a")
    sa = a.start_session(batch_size=1)
    on("b")
    sb = b.start_session(batch_size=1)
    print(sa.session, sb.session)
    assert sa.session != sb.session


from tests.test_reconcile import run_first
from tests.fakes import reopen, image_mask
from tests.test_session import edited
from omero_annotate_ai.annotators import ScriptedAnnotator


@pytest.mark.parametrize("fail", ["roi", "label"])
def test_review_fail_then_retry_same_process(tmp_path, fail):
    first, source, omero = run_first(tmp_path, n_images=1)
    annotate(first)
    omero.fail = {fail: omero.calls.get(fail, 0) + 1}
    n = first.review(ScriptedAnnotator(image_mask, review_fn=edited))
    omero.fail = {}
    print("revised", n, first.rows[0].error, first.rows[0].claimed_by)
    print(omero.tables[first.table_id][["claimed_by", "claimed_at", "label_id"]])
    n2 = first.review(ScriptedAnnotator(image_mask, review_fn=edited))
    print("second review revised", n2)
    re = reopen(first, source, omero)
    print(omero.uploads_per_tag(), re.rows[0].label_id, re.rows[0].roi_id)
