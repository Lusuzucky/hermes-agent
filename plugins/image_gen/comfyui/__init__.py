"""ComfyUI image generation backend.

Connects to any ComfyUI server (local or remote) via its REST API.
Configuration: COMFYUI_BASE_URL in .env. Workflow JSON auto-discovered
from the plugin directory (first .json file found)."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_workflow() -> Dict[str, Any]:
    """Load workflow JSON from the plugin directory.

    Looks for the first .json file in the same directory as this plugin.
    Falls back to COMFYUI_WORKFLOW_PATH env var if set.
    """
    # 1. Try env var (backward compat)
    env_path = (os.environ.get("COMFYUI_WORKFLOW_PATH") or "").strip()
    if env_path:
        with open(env_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # 2. Auto-discover from plugin directory
    plugin_dir = Path(__file__).parent
    json_files = sorted(plugin_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(
            f"No .json workflow file found in {plugin_dir} "
            "and COMFYUI_WORKFLOW_PATH not set."
        )
    path = json_files[0]
    logger.info("ComfyUI: auto-discovered workflow %s", path.name)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _inject_params(
    workflow: Dict[str, Any],
    prompt: str,
) -> Dict[str, Any]:
    """Inject prompt, DeepSeek API key, and random seed.

    Only replaces the CR Prompt Text node that feeds into DeepSeekANIMA3.
    Quality tags and artist strings are left untouched.
    Randomizes KSampler seed so every generation is unique.
    """
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")

        if class_type == "DeepSeekANIMA3":
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if api_key:
                node["inputs"]["api_key"] = api_key
            # Find the CR Prompt Text node feeding into this DeepSeekANIMA3
            prompt_input = node["inputs"].get("prompt", [])
            if isinstance(prompt_input, list) and len(prompt_input) == 2:
                src_id = str(prompt_input[0])
                src_node = workflow.get(src_id)
                if src_node and src_node.get("class_type") == "CR Prompt Text":
                    src_node["inputs"]["prompt"] = prompt

        elif class_type == "KSampler":
            import random
            node["inputs"]["seed"] = random.randint(0, 2**63 - 1)

    return workflow


def _cache_dir() -> Path:
    home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    path = Path(home) / "cache" / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ComfyUIImageGenProvider(ImageGenProvider):
    """ComfyUI image generation backend via REST API."""

    @property
    def name(self) -> str:
        return "comfyui"

    @property
    def display_name(self) -> str:
        return "ComfyUI"

    def is_available(self) -> bool:
        base = (os.environ.get("COMFYUI_BASE_URL") or "").strip()
        if not base:
            return False
        try:
            import requests  # noqa: F401
        except ImportError:
            return False
        try:
            resp = requests.get(f"{base.rstrip('/')}/system_stats", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        checkpoint = os.environ.get("COMFYUI_CHECKPOINT", "auto-detect")
        return [
            {
                "id": checkpoint,
                "display": f"ComfyUI ({checkpoint})",
                "speed": "~10-120s (depends on hardware/model)",
                "strengths": "Any SD/Flux model installed on your ComfyUI server",
                "price": "free (your GPU)",
            }
        ]

    def default_model(self) -> Optional[str]:
        return os.environ.get("COMFYUI_CHECKPOINT", "auto-detect")

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "ComfyUI",
            "badge": "free",
            "tag": "Connect to your local/remote ComfyUI server via EasyTier.",
            "env_vars": [
                {"key": "COMFYUI_BASE_URL", "prompt": "ComfyUI server URL (e.g. http://10.10.10.5:8188)"},
                {"key": "COMFYUI_WORKFLOW_PATH", "prompt": "Path to workflow API JSON"},
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required",
                error_type="invalid_argument",
                provider="comfyui",
                aspect_ratio=aspect,
            )

        base_url = (os.environ.get("COMFYUI_BASE_URL") or "").strip().rstrip("/")
        if not base_url:
            return error_response(
                error="COMFYUI_BASE_URL not set in .env",
                error_type="auth_required",
                provider="comfyui",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Load & inject workflow
        try:
            workflow = _load_workflow()
        except Exception as exc:
            return error_response(
                error=f"Failed to load workflow: {exc}",
                error_type="workflow_error",
                provider="comfyui",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            workflow = _inject_params(workflow, prompt)
        except Exception as exc:
            return error_response(
                error=f"Failed to inject parameters: {exc}",
                error_type="workflow_error",
                provider="comfyui",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Submit
        api_prompt_url = f"{base_url}/prompt"
        payload = {"prompt": workflow, "client_id": f"hermes-{uuid.uuid4().hex[:8]}"}

        # 先触发 WOL（已开机的 PC 收到无副作用），等服务就绪后再请求
        import sys, os as _os
        _parent = _os.path.join(_os.path.dirname(__file__), "..")
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        from pc_utils import wake_and_wait

        _parts = base_url.split("://")[-1].split(":")
        pc_host = _parts[0] if _parts[0] else os.environ.get("PC_IP", "")
        pc_port = int(_parts[1]) if len(_parts) > 1 else None
        wake_and_wait(host=pc_host, port=pc_port)

        try:
            resp = requests.post(api_prompt_url, json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            return error_response(
                error=f"Cannot connect to ComfyUI at {base_url}.",
                error_type="connection_error",
                provider="comfyui",
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            return error_response(
                error=f"ComfyUI submission failed: {exc}",
                error_type="api_error",
                provider="comfyui",
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.exceptions.Timeout:
            return error_response(
                error=f"ComfyUI at {base_url} timed out",
                error_type="timeout",
                provider="comfyui",
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            return error_response(
                error=f"ComfyUI submission failed: {exc}",
                error_type="api_error",
                provider="comfyui",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        prompt_id = result.get("prompt_id")
        if not prompt_id:
            error_msg = result.get("error", {})
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", json.dumps(error_msg))
            return error_response(
                error=f"ComfyUI returned no prompt_id: {error_msg}",
                error_type="api_error",
                provider="comfyui",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Poll for completion
        history_url = f"{base_url}/history/{prompt_id}"
        max_wait = int(os.environ.get("COMFYUI_TIMEOUT", "300"))
        poll_interval = 2.0
        elapsed = 0.0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            try:
                hist_resp = requests.get(history_url, timeout=10)
                hist_resp.raise_for_status()
                history = hist_resp.json()
            except Exception:
                continue

            prompt_data = history.get(prompt_id)
            if not prompt_data:
                continue

            status = prompt_data.get("status", {})
            if status.get("completed") is not True:
                if status.get("status_str") == "error":
                    return error_response(
                        error="ComfyUI job failed — check ComfyUI logs",
                        error_type="generation_error",
                        provider="comfyui",
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
                continue

            # Extract outputs
            outputs = prompt_data.get("outputs", {})
            images_found: List[Dict[str, Any]] = []

            for node_id, node_output in outputs.items():
                for img_info in node_output.get("images", []):
                    filename = img_info.get("filename", "")
                    subfolder = img_info.get("subfolder", "")
                    img_type = img_info.get("type", "output")
                    if filename:
                        images_found.append({
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": img_type,
                        })

            if not images_found:
                return error_response(
                    error="ComfyUI completed but produced no images",
                    error_type="empty_response",
                    provider="comfyui",
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            # Download first image
            first_img = images_found[0]
            params = {
                "filename": first_img["filename"],
                "subfolder": first_img["subfolder"],
                "type": first_img["type"],
            }
            view_url = f"{base_url}/view"
            try:
                img_resp = requests.get(view_url, params=params, timeout=60)
                img_resp.raise_for_status()
            except Exception as exc:
                return error_response(
                    error=f"Failed to download image from ComfyUI: {exc}",
                    error_type="download_error",
                    provider="comfyui",
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            ext = first_img["filename"].rsplit(".", 1)[-1] if "." in first_img["filename"] else "png"
            ts = time.strftime("%Y%m%d_%H%M%S")
            short = uuid.uuid4().hex[:8]
            save_path = _cache_dir() / f"comfyui_{ts}_{short}.{ext}"
            save_path.write_bytes(img_resp.content)

            logger.info(
                "ComfyUI done: prompt_id=%s seed=%s image=%s",
                prompt_id,
                workflow.get("362", {}).get("inputs", {}).get("seed", "?"),
                save_path,
            )

            return success_response(
                image=str(save_path),
                model=os.environ.get("COMFYUI_CHECKPOINT", "comfyui"),
                prompt=prompt,
                aspect_ratio=aspect,
                provider="comfyui",
                modality="text",
                extra={"prompt_id": prompt_id},
            )

        return error_response(
            error=f"ComfyUI job {prompt_id} did not complete within {max_wait}s",
            error_type="timeout",
            provider="comfyui",
            prompt=prompt,
            aspect_ratio=aspect,
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(ComfyUIImageGenProvider())
