import threading
from types import SimpleNamespace

import pytest

# curobo_service pulls in torch + the cuRobo runtime at import time; skip the
# whole module (without breaking collection) where those are unavailable.
# These tests run inside the container.
try:
    from team_8.curobo_service import CuRoboService
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - any missing runtime dep should skip
    CuRoboService = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    CuRoboService is None,
    reason=f"curobo_service import unavailable: {_IMPORT_ERROR}",
)


def _service_skeleton():
    """A CuRoboService with only the init-gating state the unit tests need."""
    svc = CuRoboService.__new__(CuRoboService)
    svc._curobo = None
    svc._init_error = ''
    svc._init_done = False
    svc._init_cv = threading.Condition()
    # _wait_for_init logs while gating; give it a no-op logger.
    svc.get_logger = lambda: SimpleNamespace(info=lambda *a, **k: None)
    return svc


def _complete(svc, curobo=None, error=''):
    with svc._init_cv:
        svc._curobo = curobo
        svc._init_error = error
        svc._init_done = True
        svc._init_cv.notify_all()


def test_wait_for_init_times_out_while_initializing():
    svc = _service_skeleton()
    curobo, error = svc._wait_for_init(0.1)
    assert curobo is None
    assert error == ''


def test_wait_for_init_returns_immediately_when_already_ready():
    svc = _service_skeleton()
    sentinel = object()
    _complete(svc, curobo=sentinel)
    curobo, error = svc._wait_for_init(5.0)
    assert curobo is sentinel
    assert error == ''


def test_wait_for_init_unblocks_when_initialization_completes():
    svc = _service_skeleton()
    sentinel = object()
    timer = threading.Timer(0.05, _complete, kwargs={'svc': svc, 'curobo': sentinel})
    timer.start()
    try:
        curobo, error = svc._wait_for_init(2.0)
    finally:
        timer.cancel()
    assert curobo is sentinel
    assert error == ''


def test_wait_for_init_unblocks_when_initialization_fails():
    svc = _service_skeleton()
    timer = threading.Timer(0.05, _complete, kwargs={'svc': svc, 'error': 'boom'})
    timer.start()
    try:
        curobo, error = svc._wait_for_init(2.0)
    finally:
        timer.cancel()
    assert curobo is None
    assert error == 'boom'
