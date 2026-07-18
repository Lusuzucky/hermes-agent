"""GPT-SoVITS TTS backend plugin.

Connects to a local/remote GPT-SoVITS API server.
base_url is read from the ``GPTSOVITS_BASE_URL`` environment variable.
All other parameters are hardcoded — no config file needed.

Plugin directory: ``~/.hermes/plugins/tts/gptsovits/``
Enable: ``hermes plugins enable gptsovits``
Select: ``hermes config set tts.provider gptsovits``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)

# ── Hardcoded defaults ───────────────────────────────────────────────────
_ENDPOINT = "/tts"
_TIMEOUT = 120
_TEXT_LANG = "zh"

# Valid emotion values from the backend
_VALID_EMOTIONS = {
    "普通", "平淡", "开心", "撒娇", "夹子", "温柔",
    "低落", "委屈", "慵懒", "生气", "有气无力"
}


# ── Provider ──────────────────────────────────────────────────────────────

class GPTSoVITSTTSProvider(TTSProvider):
    """GPT-SoVITS TTS backend via REST API."""

    @property
    def name(self) -> str:
        return "gptsovits"

    @property
    def display_name(self) -> str:
        return "GPT-SoVITS"

    def _base_url(self) -> str:
        return (os.environ.get("GPTSOVITS_BASE_URL", "") or "").strip().rstrip("/")

    def is_available(self) -> bool:
        """Check if the GPT-SoVITS API server is reachable."""
        base = self._base_url()
        if not base:
            return False
        try:
            resp = requests.get(f"{base}/", timeout=5)
            return resp.status_code in (200, 404)
        except Exception:
            return False

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "GPT-SoVITS",
            "badge": "local",
            "tag": "GPT-SoVITS voice-cloning TTS via REST API",
            "env_vars": [
                {
                    "key": "GPTSOVITS_BASE_URL",
                    "prompt": "GPT-SoVITS API base URL (e.g. http://10.10.10.5:9880)",
                },
            ],
        }

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = "wav",
        **extra: Any,
    ) -> str:
        """Send text to GPT-SoVITS API and write the returned audio to *output_path*."""
        base = self._base_url()
        if not base:
            raise RuntimeError(
                "GPT-SoVITS base_url is not set. "
                "Set GPTSOVITS_BASE_URL in your .env file."
            )

        url = f"{base}{_ENDPOINT}"

        # Emotion is REQUIRED — agent must pass it explicitly.
        emotion = extra.get("emotion") if extra else None
        if not emotion or emotion not in _VALID_EMOTIONS:
            raise RuntimeError(
                f"GPT-SoVITS requires a valid emotion parameter. "
                f"Must be one of: {', '.join(sorted(_VALID_EMOTIONS))}. "
                f"Got: {emotion!r}"
            )

        payload = {
            "text": text,
            "text_lang": _TEXT_LANG,
            "emotion": emotion,
        }

        logger.info("GPT-SoVITS synthesizing: text_len=%d emotion=%s → %s", len(text), emotion, url)

        # WOL: wake PC before requesting
        import sys
        _parent = os.path.join(os.path.dirname(__file__), "..")
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        from pc_utils import wake_and_wait

        _parts = base.split("://")[-1].split(":")
        pc_host = _parts[0] if _parts[0] else os.environ.get("PC_IP", "")
        pc_port = int(_parts[1]) if len(_parts) > 1 else None
        wake_and_wait(host=pc_host, port=pc_port)

        try:
            resp = requests.post(url, json=payload, timeout=_TIMEOUT)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot connect to GPT-SoVITS at {url}. Is the server running?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(f"GPT-SoVITS request timed out after {_TIMEOUT}s") from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"GPT-SoVITS API error: {exc.response.status_code} — {exc.response.text[:500]}") from exc

        with open(output_path, "wb") as f:
            f.write(resp.content)

        logger.info("GPT-SoVITS audio saved: %s (%d bytes)", output_path, len(resp.content))
        return output_path


# ── Plugin entry point ────────────────────────────────────────────────────

def register(ctx) -> None:
    """Plugin entry point — wire GPT-SoVITS TTS provider into the registry."""
    ctx.register_tts_provider(GPTSoVITSTTSProvider())
