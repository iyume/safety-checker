"""Mock the stable diffusion pipeline to implement safety checker.

Reference:
* https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py
* https://huggingface.co/CompVis/stable-diffusion-safety-checker
* https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/safety_checker
"""

try:
    import diffusers
    import torch
    import transformers
except ImportError as e:
    raise ImportError('please install by "pip install safety-checker[sdmock]"') from e

from contextlib import contextmanager
from typing import List, Union, cast

import diffusers
import torch
import transformers
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from PIL import Image
from transformers import CLIPImageProcessor

# Silence warning message: Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.
diffusers.utils.logging.set_verbosity_error()

# Both runwayml/stable-diffusion-v1-5 and CompVis/stable-diffusion-safety-checker
# are able to use. Their config.json are slightly different.
# Or we can provide variant loader here?
# We choose runwayml/stable-diffusion-v1-5 because it provides fp16 weight which is
# half smaller than the full-precision weight.


@contextmanager
def _suppress_transformer_warning():
    """Silence warning message on loading safety checker config: `text_config_dict` is provided which will be used to initialize `CLIPTextConfig`"""
    _origin_level = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    yield
    transformers.logging.set_verbosity(_origin_level)


# Downloads config.json and model.fp16.safetensors
with _suppress_transformer_warning():
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="safety_checker",
        use_safetensors=True,
        variant="fp16",
    )
safety_checker = cast(StableDiffusionSafetyChecker, safety_checker)


# Loads https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/feature_extractor/preprocessor_config.json
# Feature extractor is just an image preprocessor that doesn't require weight
feature_extractor = CLIPImageProcessor.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="feature_extractor",
)
feature_extractor = cast(CLIPImageProcessor, feature_extractor)


def run_safety_checker(
    image: Union[Image.Image, List[Image.Image]],
    device: Union[torch.device, str, int] = torch.device("cpu"),
) -> bool:
    """Run safety checker."""
    if not isinstance(device, torch.device):
        device = torch.device(device)
    safety_checker.to(device)  # type: ignore  # diffusers shit types
    # This pipeline is designed for SD batch generation
    if not isinstance(image, list):
        image = [image]
    # No need to do input image normalization
    safety_checker_input = feature_extractor(image, return_tensors="pt")
    # Safety checker doesn't process image in its source code, so just put in
    has_nsfw: list[bool]
    _, has_nsfw = safety_checker(
        images=safety_checker_input.pixel_values,  # dummy input
        clip_input=safety_checker_input.pixel_values.to(device),
    )
    return any(has_nsfw)
