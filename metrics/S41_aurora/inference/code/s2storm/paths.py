"""Data and code locations. Everything resolves from S2STORM_DATA (no hard-coded /net/monsoon)."""
import os
import socket
from pathlib import Path

DEFAULTS = {
    "dsi": "/net/spaces/monsoon/jhayron/s2storm",
    "aurora": "/lus/flare/projects/S2Storm/s2storm",
}


def server() -> str:
    """Best guess of the machine we run on: 'aurora', 'dsi' or 'other'."""
    if os.environ.get("S2STORM_SERVER"):
        return os.environ["S2STORM_SERVER"]
    host = socket.getfqdn()
    if "aurora" in host or "alcf" in host or Path("/lus/flare").exists():
        return "aurora"
    if "ds.uchicago" in host or Path("/net/spaces/monsoon").exists():
        return "dsi"
    return "other"


def data_root() -> Path:
    root = os.environ.get("S2STORM_DATA") or DEFAULTS.get(server())
    if root is None:
        raise RuntimeError("Set S2STORM_DATA (no default for this machine)")
    return Path(root)


def runs() -> Path:
    return data_root() / "runs"


def data() -> Path:
    return data_root() / "data"


def logs() -> Path:
    return data_root() / "logs"


def envs() -> Path:
    return data_root() / "envs"


def ext() -> Path:
    """Upstream clones (never edited), e.g. ext/S2S = masak1112/S2S."""
    return data_root() / "ext"


def ucs2s_repo() -> Path:
    return Path(os.environ.get("UCS2S_REPO", ext() / "S2S"))
