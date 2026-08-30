"""Camera device discovery and selection.

Device *indices are not stable*: unplug the external webcam, reboot, or plug it into
another port and 2 can become 3. So a device may be given three ways, and everything
in this project accepts all three:

  * an index          2                    fast, but shifts around
  * a device path     /dev/video2          Linux only, still shifts on re-enumeration
  * a name fragment   logitech             case-insensitive, survives re-enumeration

Prefer the name fragment for anything you write down.
"""
from __future__ import annotations

import glob
import os
import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DeviceInfo:
    index: int
    path: str                 # "/dev/video2" on Linux; "" elsewhere
    name: str                 # "Integrated Camera" / "C920"
    by_id: str | None         # stable /dev/v4l/by-id/... symlink, if any
    sysfs_index: int | None   # 0 = the capture node; 1+ are usually metadata nodes
    opens: bool | None = None # filled in by probe(): did it actually yield a frame?
    frame: tuple | None = None


def _linux_devices() -> list[DeviceInfo]:
    out = []
    by_id = {}
    for link in glob.glob("/dev/v4l/by-id/*"):
        try:
            by_id[os.path.realpath(link)] = link
        except OSError:
            pass
    for path in sorted(glob.glob("/dev/video*"),
                       key=lambda p: int("".join(filter(str.isdigit, p)) or -1)):
        idx = int("".join(filter(str.isdigit, path)) or -1)
        sysfs = Path(f"/sys/class/video4linux/video{idx}")
        name = (sysfs / "name").read_text().strip() if (sysfs / "name").exists() else "?"
        try:
            si = int((sysfs / "index").read_text().strip())
        except (OSError, ValueError):
            si = None
        out.append(DeviceInfo(idx, path, name, by_id.get(path), si))
    return out


def list_devices(max_probe_index: int = 6) -> list[DeviceInfo]:
    """Enumerate cameras. Linux reads sysfs; elsewhere we can only guess indices."""
    if platform.system() == "Linux":
        return _linux_devices()
    return [DeviceInfo(i, "", f"index {i}", None, None) for i in range(max_probe_index)]


def probe(devices: list[DeviceInfo], width=640, height=480) -> list[DeviceInfo]:
    """Actually open each device and try to read one frame.

    This is the only reliable way to tell a capture node from a metadata node: many
    UVC webcams expose two /dev/video* nodes and only the first delivers images.
    """
    import cv2
    from .camera import _api_for_backend
    for d in devices:
        cap = None
        try:
            target = d.path if d.path else d.index
            cap = cv2.VideoCapture(target, _api_for_backend("auto"))
            if not cap.isOpened():
                d.opens = False
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, img = cap.read()
            d.opens = bool(ok and img is not None)
            if d.opens:
                d.frame = (img.shape[1], img.shape[0])
        except Exception:
            d.opens = False
        finally:
            if cap is not None:
                cap.release()
    return devices


def resolve(spec, devices: list[DeviceInfo] | None = None):
    """Turn an index / path / name fragment into something cv2.VideoCapture accepts.

    Raises ValueError with the available devices listed, rather than letting OpenCV
    fail with a bare 'can't open camera by index'.
    """
    if isinstance(spec, int):
        return spec
    s = str(spec).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    if s.startswith("/"):
        return s
    devs = devices if devices is not None else list_devices()
    hits = [d for d in devs if s.lower() in d.name.lower()
            or (d.by_id and s.lower() in d.by_id.lower())]
    # prefer the capture node when a name matches several of one camera's nodes
    capture_hits = [d for d in hits if d.sysfs_index in (0, None)]
    if len(capture_hits) == 1:
        return capture_hits[0].path or capture_hits[0].index
    if len(hits) == 1:
        return hits[0].path or hits[0].index
    if not hits:
        raise ValueError(f"no camera matching {spec!r}. Available:\n{format_table(devs)}")
    raise ValueError(f"{spec!r} is ambiguous, it matches {len(hits)} nodes:\n"
                     f"{format_table(hits)}\nUse a device path or an index instead.")


def format_table(devices: list[DeviceInfo]) -> str:
    rows = ["  IDX  PATH             NODE      PROBE          NAME",
            "  ---  ---------------  --------  -------------  ----------------------"]
    for d in devices:
        node = ("capture" if d.sysfs_index == 0
                else f"meta({d.sysfs_index})" if d.sysfs_index else "?")
        if d.opens is None:
            pr = "(not probed)"
        elif d.opens:
            pr = f"OK {d.frame[0]}x{d.frame[1]}" if d.frame else "OK"
        else:
            pr = "no frames"
        rows.append(f"  {d.index:<3}  {d.path or '-':<15}  {node:<8}  {pr:<13}  {d.name}")
        if d.by_id:
            rows.append(f"       stable: {d.by_id}")
    return "\n".join(rows)
