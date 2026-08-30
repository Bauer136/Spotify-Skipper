"""Load config.toml + .env.

Phase 8.2 extends this with build_pose_fn / build_detector / build_engine. The
loading half lives here early because camera selection needs a persisted default.
"""
from __future__ import annotations

import re
import tomllib                       # stdlib on Python 3.11+
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    def section(self, name: str) -> dict:
        return dict(self.raw.get(name, {}))


def load_config(path: str | Path | None = None) -> Config:
    try:
        from dotenv import load_dotenv
        load_dotenv()                              # pulls SPOTIFY_CLIENT_ID
    except ImportError:
        pass
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with p.open("rb") as fh:
        return Config(tomllib.load(fh), p)


def camera_kwargs(cfg: Config, override_device=None) -> dict:
    """[camera] as Camera(**kwargs). `override_device` wins — that is the CLI flag."""
    c = cfg.section("camera")
    return dict(
        device=override_device if override_device is not None else c.get("device", 0),
        width=int(c.get("width", 640)),
        height=int(c.get("height", 480)),
        fps=int(c.get("fps", 30)),
        mirror=bool(c.get("mirror", True)),
        backend=c.get("backend", "auto"),
        threaded=bool(c.get("threaded", True)),
    )


def set_camera_device(device, path: str | Path | None = None) -> str:
    """Rewrite `device = ...` inside [camera], preserving comments and layout.

    tomllib is read-only by design, and a full round-trip through a TOML writer would
    strip every comment in config.toml — which is where most of this project's
    documentation lives. So this edits the one line in place.
    """
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    value = str(device) if isinstance(device, int) or str(device).lstrip("-").isdigit() \
        else f'"{device}"'
    section, done = None, False
    for i, line in enumerate(lines):
        m = re.match(r"\s*\[([^\]]+)\]", line)
        if m:
            section = m.group(1)
            continue
        if section == "camera" and re.match(r"\s*device\s*=", line) and not done:
            comment = line.split("#", 1)[1].rstrip("\n") if "#" in line else ""
            lines[i] = f"device  = {value}" + (f"  #{comment}" if comment else "") + "\n"
            done = True
    if not done:
        raise RuntimeError(f"no `device =` line under [camera] in {p}")
    p.write_text("".join(lines), encoding="utf-8")
    return str(p)
