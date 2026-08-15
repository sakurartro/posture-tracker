"""Silencing native (non-Python) stderr noise.

MediaPipe/TFLite print a handful of C++ log lines on every model load (GPU
delegate setup, EGL/GL context info, an XNNPACK notice, an absl banner about
logging before its own init), and OpenCV's Qt-based preview window prints a
missing-font warning on every open. None of it is configurable from Python --
GLOG_minloglevel and TF_CPP_MIN_LOG_LEVEL do nothing against this build, and
some of the lines come from a plain fprintf rather than any logging framework
at all. What all of it shares is the file descriptor it writes to.
"""

from __future__ import annotations

import contextlib
import os


@contextlib.contextmanager
def native_stderr_silenced():
    """Redirects fd 2 to /dev/null for the duration of the block.

    Confirmed a genuine failure inside the block (e.g. a missing model file)
    still raises a normal, readable Python exception -- the message travels
    with the exception object, not through stderr -- so real errors are never
    swallowed by this.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)
