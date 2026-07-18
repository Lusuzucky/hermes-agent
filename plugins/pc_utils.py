"""PC Wake-on-LAN utility — shared by ComfyUI and GPT-SoVITS plugins.

When a plugin gets a ConnectionError (PC is off), this module:
1.  Signals the WOL server to wake the PC via the router
2.  Waits (polls) until the PC is reachable
3.  Returns so the plugin can retry its original request

PC_IP is the PC's EasyTier IP — used for health-check after waking
(TCP probe on PC_HEALTH_PORT to confirm the PC is up and services ready).

Configuration via environment variables:
  PC_IP           — PC's EasyTier IP for health check (default: 10.10.10.5)
  PC_BOOT_TIMEOUT — max seconds to wait for PC to boot (default: 120)
  PC_HEALTH_PORT  — port to poll for health check (default: 8188, ComfyUI)
  WOL_MAC         — MAC address of the PC to wake (required)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── config ──────────────────────────────────────────────────────────────────

PC_IP = os.environ.get("PC_IP", "10.10.10.5").strip()
PC_BOOT_TIMEOUT = int(os.environ.get("PC_BOOT_TIMEOUT", "120"))
PC_HEALTH_PORT = int(os.environ.get("PC_HEALTH_PORT", "8188"))
WOL_MAC = os.environ.get("WOL_MAC", "").strip()
WOL_TRIGGER_FILE = "/tmp/wol_trigger"


def _trigger_wol(mac: str) -> None:
    """Write MAC address to the trigger file that wol-server.py watches."""
    try:
        with open(WOL_TRIGGER_FILE, "w") as f:
            f.write(mac + "\n")
        logger.info("WOL trigger written: %s", mac)
    except Exception as exc:
        logger.warning("Failed to write WOL trigger: %s", exc)


def _pc_is_up(host: str, port: int, timeout: int = 3) -> bool:
    """Try a raw TCP connect — fast and service-agnostic."""
    import socket

    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except OSError:
        return False


def wake_and_wait(
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
    mac: Optional[str] = None,
) -> bool:
    """Trigger WOL, then block until the PC is reachable or *timeout* expires.

    mac  — MAC address to wake. Uses WOL_MAC env var if not provided.
    Returns True if the PC came online, False otherwise.
    """
    host = host or PC_IP
    port = port or PC_HEALTH_PORT
    timeout = timeout if timeout is not None else PC_BOOT_TIMEOUT
    mac = mac or WOL_MAC
    if not mac:
        raise ValueError("MAC address required. Set WOL_MAC env var or pass mac= parameter")

    logger.info("PC %s:%d is unreachable — sending WOL (mac=%s)...", host, port, mac)
    _trigger_wol(mac)

    deadline = time.time() + timeout
    interval = 1  # seconds between checks

    # 先立刻检测一次（PC 可能本来就开着），之后每次间隔 interval
    while time.time() < deadline:
        if _pc_is_up(host, port, timeout=1):
            elapsed = time.time() - (deadline - timeout)
            logger.info("PC %s:%d is now reachable (waited %.0fs)", host, port, elapsed)
            return True
        time.sleep(interval)

    logger.warning("PC %s:%d did not come online within %ds", host, port, timeout)
    return False
