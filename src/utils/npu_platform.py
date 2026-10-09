#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

import logging as Logger
import re
import shutil
import subprocess

_MDS_KEYWORDS = ("medusa", "mds")
_STX_KEYWORDS = ("stx", "krk", "gpt", "gpt2", "strix", "kracken", "gorgon")

MDS_DD_PLUGINS_SUBPATH = "lib/transaction/mds"
STX_DD_PLUGINS_SUBPATH = "lib/transaction/stx"


def get_dd_plugins_subpath(platform: str) -> str:
    """Return the project-relative DD_PLUGINS_ROOT path for the given NPU platform."""
    if platform == "mds":
        return MDS_DD_PLUGINS_SUBPATH
    return STX_DD_PLUGINS_SUBPATH


def parse_xrt_smi_examine_output(text: str) -> tuple[str | None, str | None]:
    """Extract (system_model, device_name) from xrt-smi examine text output."""
    system_model = None
    device_name = None
    match = re.search(r"^\s*Model\s*:\s*(.+)$", text, re.MULTILINE)
    if match:
        system_model = match.group(1).strip()
    for line in text.splitlines():
        if line.strip().startswith("|") and "00" in line:
            cols = [col.strip() for col in line.split("|") if col.strip()]
            if len(cols) >= 2:
                device_name = cols[1]
                break
    return system_model, device_name


def detect_npu_platform() -> str:
    """Return 'mds', 'stx', or 'unknown'. Falls back to 'unknown' when detection fails."""
    cmd = shutil.which("xrt-smi") or shutil.which("xrt-smi.exe")
    if not cmd:
        Logger.debug("xrt-smi not found in PATH; NPU platform detection skipped.")
        return "unknown"
    try:
        result = subprocess.run(
            [cmd, "examine"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        Logger.debug("xrt-smi examine failed: %s", exc)
        return "unknown"
    if result.returncode != 0:
        Logger.debug("xrt-smi examine exited with code %s", result.returncode)
        return "unknown"

    combined = f"{result.stdout}\n{result.stderr}"
    system_model, device_name = parse_xrt_smi_examine_output(combined)
    haystack = " ".join(filter(None, [system_model, device_name])).lower()
    Logger.debug(
        "xrt-smi examine: system_model=%r, device_name=%r",
        system_model,
        device_name,
    )

    if any(keyword in haystack for keyword in _MDS_KEYWORDS):
        return "mds"
    if any(keyword in haystack for keyword in _STX_KEYWORDS):
        return "stx"
    return "unknown"
