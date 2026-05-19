"""Fixtures for end-to-end browser tests.

Provides:

* ``docs_server``  - session-scoped http.server serving the ``docs/``
                     directory on a free localhost port. Yields the
                     base URL.
* ``analysis_url`` - convenience function-scoped URL pointing at the
                     analysis page.
* ``design_url``   - same, for the design page.

Tests in this directory use Playwright's ``page`` fixture (provided
by pytest-playwright) to drive a real Chromium instance.
"""

from __future__ import annotations

import http.server
import socket
import socketserver
import threading
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_DIR = _REPO_ROOT / "docs"


def _pick_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _SilentHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that does not spam stderr on each request."""

    def log_message(self, format: str, *args: object) -> None:
        pass


class _DocsServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, port: int) -> None:
        # SimpleHTTPRequestHandler resolves relative paths against
        # the current working directory. Pass the docs root via the
        # `directory` parameter so it's hermetic.
        handler_cls = type(
            "DocsHandler",
            (_SilentHandler,),
            {"directory": str(_DOCS_DIR)},
        )

        def factory(*args: object, **kwargs: object) -> http.server.SimpleHTTPRequestHandler:
            return handler_cls(*args, directory=str(_DOCS_DIR), **kwargs)

        super().__init__(("127.0.0.1", port), factory)


@pytest.fixture(scope="session")
def docs_server() -> Iterator[str]:
    """Serve ``docs/`` on a random localhost port for the test session."""
    port = _pick_free_port()
    server = _DocsServer(port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def analysis_url(docs_server: str) -> str:
    return f"{docs_server}/analysis/"


@pytest.fixture
def design_url(docs_server: str) -> str:
    return f"{docs_server}/design/"
