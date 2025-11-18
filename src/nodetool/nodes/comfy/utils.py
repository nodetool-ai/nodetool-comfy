from __future__ import annotations

import base64
from contextlib import contextmanager
from typing import Generator, Optional

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import ImageRef

import comfy
import comfy.model_management
import comfy.utils

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress, OutputUpdate

# Import latent preview utilities
try:
    from latent_preview import get_previewer

    LATENT_PREVIEW_AVAILABLE = True
except ImportError:
    LATENT_PREVIEW_AVAILABLE = False

logger = get_logger(__name__)


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
    model: Optional[object] = None,
) -> Generator[None, None, None]:
    """
    Temporary ComfyUI progress hook that maps ProgressBar updates
    into Nodetool NodeProgress and PreviewUpdate messages.

    Supports latent preview decoding when model is provided. The preview
    is typically already decoded by prepare_callback, but this allows
    for direct latent decoding if needed.

    Args:
        context: Processing context for posting messages
        node: The node executing the workflow
        model: Optional ComfyUI model object for latent preview decoding
    """
    previous_hook = getattr(comfy.utils, "PROGRESS_BAR_HOOK", None)

    # Set up previewer if model is provided and latent preview is available
    previewer = None
    preview_format = "JPEG"
    if model is not None and LATENT_PREVIEW_AVAILABLE:
        try:
            # Get previewer from model's latent format
            has_load_device = hasattr(model, "load_device")
            has_model = hasattr(model, "model")
            has_latent_format = has_model and hasattr(model.model, "latent_format")
            logger.debug(
                f"Setting up previewer: model_type={type(model)}, "
                f"has_load_device={has_load_device}, "
                f"has_model={has_model}, "
                f"has_latent_format={has_latent_format}"
            )
            if has_load_device and has_model and has_latent_format:
                previewer = get_previewer(model.load_device, model.model.latent_format)
                logger.debug(
                    f"Previewer created: {previewer is not None}, "
                    f"type={type(previewer) if previewer else None}"
                )
            else:
                logger.debug(
                    "Model does not have required attributes for previewer setup"
                )
        except Exception as e:
            logger.warning(f"Could not set up previewer: {e}", exc_info=True)
    elif model is None:
        logger.debug("No model provided for previewer setup")
    elif not LATENT_PREVIEW_AVAILABLE:
        logger.debug("Latent preview module not available")

    def hook(value, total, preview, node_id=None):
        try:
            progress_value = int(value)
            total_value = int(total) if total is not None else 0
            context.post_message(
                NodeProgress(
                    node_id=node.id,
                    progress=progress_value,
                    total=total_value,
                )
            )

            if preview is not None:
                logger.debug(
                    f"Preview received: type={type(preview)}, "
                    f"is_tuple={isinstance(preview, tuple)}, "
                    f"previewer_available={previewer is not None}"
                )
                # Preview can be:
                # 1. A tuple (format, PIL.Image, max_resolution) - already decoded
                # 2. A torch.Tensor (latent) - needs decoding if previewer is available
                image = None
                fmt = preview_format

                if isinstance(preview, tuple) and len(preview) >= 2:
                    # Already decoded preview tuple
                    fmt, image, max_resolution = (
                        preview[0],
                        preview[1],
                        preview[2] if len(preview) > 2 else None,
                    )
                    logger.debug(
                        f"Decoded preview tuple: format={fmt}, "
                        f"image_type={type(image)}, "
                        f"max_resolution={max_resolution}"
                    )
                elif previewer is not None:
                    # Try to decode as latent tensor
                    try:
                        import torch

                        if isinstance(preview, torch.Tensor):
                            logger.debug(
                                f"Decoding latent tensor: shape={preview.shape}, "
                                f"dtype={preview.dtype}"
                            )
                            preview_tuple = previewer.decode_latent_to_preview_image(
                                preview_format, preview
                            )
                            fmt, image, max_resolution = preview_tuple
                            logger.debug(
                                f"Decoded latent: format={fmt}, "
                                f"image_type={type(image)}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Could not decode latent preview: {e}", exc_info=True
                        )
                        return

                if image is not None:
                    from io import BytesIO

                    buffer = BytesIO()
                    image.save(buffer, format=fmt)
                    image_data = buffer.getvalue()
                    logger.debug(
                        f"Posting preview update: node_id={node_id or node.id}, "
                        f"format={fmt}, size={len(image_data)} bytes"
                    )
                    context.post_message(
                        OutputUpdate(
                            node_id=node_id or node.id,
                            value=ImageRef(
                                data=image_data,
                            ),
                        )
                    )
                else:
                    logger.debug("Preview received but image is None after processing")
            else:
                logger.debug(
                    f"No preview for progress update: {progress_value}/{total_value}"
                )
        except Exception as e:
            logger.error(f"Error updating progress: {e}", exc_info=True)

    comfy.utils.set_progress_bar_enabled(True)
    comfy.utils.set_progress_bar_global_hook(hook)
    try:
        yield
    finally:
        comfy.utils.set_progress_bar_global_hook(previous_hook)
