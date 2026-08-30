#!/usr/bin/env python3
"""List cameras and set the default one.

    python scripts/cameras.py                      # list + probe every device
    python scripts/cameras.py --set-default 2      # by index
    python scripts/cameras.py --set-default logitech   # by name fragment (preferred)
    python scripts/cameras.py --set-default /dev/video2
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.spotify_skipper import devices as D
from src.spotify_skipper.config import load_config, set_camera_device


def main() -> int:
    ap = argparse.ArgumentParser(description="List cameras / set the default camera.")
    ap.add_argument("--set-default", metavar="SPEC",
                    help="index, /dev path, or name fragment to write to config.toml")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip opening each device (faster, but cannot tell a capture "
                         "node from a metadata node)")
    args = ap.parse_args()

    devs = D.list_devices()
    if not args.no_probe:
        D.probe(devs)

    if args.set_default:
        try:
            target = D.resolve(args.set_default, devs)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        match = next((d for d in devs if d.path == target or d.index == target), None)
        if match and match.opens is False:
            print(f"refusing: {target!r} ({match.name}) opened but delivered no frames "
                  f"— it is almost certainly a metadata node, not a camera.",
                  file=sys.stderr)
            print(D.format_table(devs), file=sys.stderr)
            return 2
        # store the name fragment verbatim when given one: it survives re-enumeration
        stored = args.set_default if not str(args.set_default).lstrip("-").isdigit() \
            else int(args.set_default)
        path = set_camera_device(stored)
        print(f"default camera set to {stored!r} (resolves to {target!r}) in {path}")
        return 0

    print(D.format_table(devs))
    try:
        cur = load_config().section("camera").get("device", 0)
        print(f"\ncurrent default in config.toml: {cur!r}")
    except FileNotFoundError:
        print("\nno config.toml yet")
    print("\nPick a row with PROBE=OK. Prefer a name fragment over an index — indices "
          "shift when you replug or reboot:\n  python scripts/cameras.py --set-default "
          "<name-fragment>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
