"""Mirrored camera preview for aiming the webcam before tracking starts.

Exists because bad framing is silent otherwise: the tracker simply reports
that nobody is there, and it is not obvious whether the camera is pointed
wrong, the lighting is bad, or the app is broken. This shows exactly what the
detector sees.

Mirrored on purpose -- an un-mirrored feed makes every correction feel
inverted. The only requirement is that the whole head fits inside the frame
with a margin; sitting off to one side is fine, since posture is measured
relative to a calibrated baseline rather than to the centre of the picture.
"""

from __future__ import annotations

import time

import cv2

GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

# Clear space the head needs from each edge, as a fraction of the frame. Small
# on purpose: the detector copes fine with the head off to one side, what
# breaks it is the head running off an edge.
EDGE_MARGIN = 0.02
# How long the framing must hold before the preview declares success.
STEADY_SECONDS = 1.5
WINDOW = "posture-tracker - aim the camera - ESC to close"


def run_preview(cap, face_source, timeout_seconds: float = 300.0) -> bool:
    """Shows the live preview until the framing is good, ESC, or timeout.

    Returns True if the head was seen fully in frame and steady.
    """
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 720)

    started = time.monotonic()
    good_since: float | None = None
    framed = False
    try:
        while time.monotonic() - started < timeout_seconds:
            frame = face_source.read_frame(cap)
            if frame is None:
                continue

            height, width = frame.shape[:2]
            bounds = face_source.face_bounds(frame)
            view = cv2.flip(frame, 1)

            if bounds is None:
                hint = "no face detected - point the camera at yourself"
                ok = False
            else:
                x0, y0, x1, y1 = bounds
                ok = (x0 > EDGE_MARGIN and x1 < 1 - EDGE_MARGIN
                      and y0 > EDGE_MARGIN and y1 < 1 - EDGE_MARGIN)
                # x is mirrored for display, so the box corners swap sides.
                cv2.rectangle(view,
                              (int((1 - x1) * width), int(y0 * height)),
                              (int((1 - x0) * width), int(y1 * height)),
                              GREEN if ok else RED, 3)
                if ok:
                    hint = "head fully in frame - you are good to go"
                elif x1 >= 1 - EDGE_MARGIN or x0 <= EDGE_MARGIN:
                    hint = "head clipped at the side - turn the camera towards you"
                elif y1 >= 1 - EDGE_MARGIN:
                    hint = "chin clipped - tilt the camera up"
                else:
                    hint = "forehead clipped - tilt the camera down"

            cv2.putText(view, hint, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        GREEN if ok else RED, 2)
            cv2.putText(view, "sit how you normally work - move the CAMERA, not yourself",
                        (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
            cv2.putText(view, "ESC to close", (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

            if ok:
                good_since = good_since or time.monotonic()
                if time.monotonic() - good_since > STEADY_SECONDS:
                    framed = True
                    break
            else:
                good_since = None

            cv2.imshow(WINDOW, view)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cv2.destroyAllWindows()
        # A couple of extra waitKey calls let the window actually disappear;
        # destroyAllWindows alone often leaves a ghost on X11.
        for _ in range(4):
            cv2.waitKey(1)

    return framed
