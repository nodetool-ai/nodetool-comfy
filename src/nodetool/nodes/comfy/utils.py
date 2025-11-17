from __future__ import annotations

import base64
from contextlib import contextmanager
from typing import Generator

import comfy
import comfy.model_management
import comfy.utils

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress, PreviewUpdate


def unload_comfy_model(model) -> None:
    """
    Unload a ComfyUI model, preferring `unload_model_clones` when available.
    """
    unload_fn = getattr(comfy.model_management, "unload_model_clones", None)
    if callable(unload_fn):
        unload_fn(model)
    else:
        comfy.model_management.unload_all_models()


@contextmanager
def comfy_progress(
    context: ProcessingContext,
    node: BaseNode,
) -> Generator[None, None, None]:
    """
    Temporary ComfyUI progress hook that maps ProgressBar updates
    into Nodetool NodeProgress and PreviewUpdate messages.
    """
    previous_hook = getattr(comfy.utils, "PROGRESS_BAR_HOOK", None)
    node_id = getattr(node, "_id", getattr(node, "id", ""))

    def hook(value, total, preview, node_id=None):
        try:
            progress_value = int(value)
            total_value = int(total) if total is not None else 0
            context.post_message(
                NodeProgress(
                    node_id=node_id,
                    progress=progress_value,
                    total=total_value,
                )
            )

            if preview is not None:
                fmt, image, max_resolution = preview
                from io import BytesIO

                buffer = BytesIO()
                image.save(buffer, format=fmt)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                data_uri = f"data:image/{fmt.lower()};base64,{encoded}"
                context.post_message(
                    PreviewUpdate(
                        node_id=node_id,
                        value={
                            "image": data_uri,
                            "max_resolution": max_resolution,
                        },
                    )
                )
        except Exception:
            # Never let progress failures break sampling loops
            pass

    comfy.utils.set_progress_bar_enabled(True)
    comfy.utils.set_progress_bar_global_hook(hook)
    try:
        yield
    finally:
        comfy.utils.set_progress_bar_global_hook(previous_hook)


