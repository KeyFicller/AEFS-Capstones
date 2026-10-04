"""Multi-vector encoder over a ColBERT-style vision-language checkpoint."""

import os

import torch
from PIL import Image

from multimodal_doc_qa.config import Settings

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    """Guard the free-form config value: a typo must name the accepted values, not ``KeyError``."""
    try:
        return _DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(_DTYPES)}") from exc


def _load(model_name: str, settings: Settings) -> tuple[object, object]:
    """Load ``(model, processor)``. On MPS, force synchronous weight loads before ``from_pretrained``."""
    from colpali_engine import ColIdefics3, ColIdefics3Processor, ColQwen2_5, ColQwen2_5_Processor

    if settings.device == "mps":
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

    if "colqwen" in model_name.lower():
        model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=_resolve_dtype(settings.dtype),
            device_map=settings.device,
        ).eval()
        return model, ColQwen2_5_Processor.from_pretrained(model_name)

    model = ColIdefics3.from_pretrained(
        model_name,
        torch_dtype=_resolve_dtype(settings.dtype),
        device_map=settings.device,
    ).eval()
    return model, ColIdefics3Processor.from_pretrained(model_name)


class MultiVectorEncoder:
    """Page images and queries as multi-vectors. Both methods return CPU ``float32``.

    MPS tensors must not outlive the forward pass, and the index runs on CPU.
    """

    def __init__(self, model_name: str, settings: Settings) -> None:
        """Bind ``model_name``. A load failure propagates; there is no second checkpoint."""
        self.settings = settings
        self.model, self.processor = _load(model_name, settings)

    def encode_images(self, images: list[Image.Image]) -> list[torch.Tensor]:
        """Encode page images to a list of ``[n_patches, dim]`` matrices."""
        out: list[torch.Tensor] = []
        for img in images:
            batch = self.processor.process_images([img]).to(self.settings.device)
            with torch.no_grad():
                emb = self.model(**batch)
            out.append(emb[0].to(torch.float32).cpu())
            if self.settings.device == "mps":
                torch.mps.empty_cache()
        return out

    def encode_texts(self, texts: list[str]) -> list[torch.Tensor]:
        """Encode document text to a list of ``[n_tokens, dim]`` matrices.

        ``process_texts`` keeps the string on the document side of MaxSim.
        ``process_queries`` appends query-augmentation tokens and is only for questions.
        """
        out: list[torch.Tensor] = []
        for text in texts:
            batch = self.processor.process_texts([text]).to(self.settings.device)
            with torch.no_grad():
                emb = self.model(**batch)
            out.append(emb[0].to(torch.float32).cpu())
            if self.settings.device == "mps":
                torch.mps.empty_cache()
        return out

    def encode_query(self, text: str) -> torch.Tensor:
        """Encode a query string to a ``[n_tokens, dim]`` matrix."""
        batch = self.processor.process_queries([text]).to(self.settings.device)
        with torch.no_grad():
            emb = self.model(**batch)
        out = emb[0].to(torch.float32).cpu()
        if self.settings.device == "mps":
            torch.mps.empty_cache()
        return out
