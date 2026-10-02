import pytest
import torch


def test_torch_mps_available() -> None:
    assert torch.backends.mps.is_available(), "MPS backend must be available"


@pytest.mark.slow
def test_colsmol_forward_on_mps() -> None:
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor

    model = ColIdefics3.from_pretrained(
        "vidore/colSmol-500M", torch_dtype=torch.float16, device_map="mps"
    ).eval()
    processor = ColIdefics3Processor.from_pretrained("vidore/colSmol-500M")
    from PIL import Image

    batch = processor.process_images([Image.new("RGB", (512, 384), "white")]).to("mps")
    with torch.no_grad():
        out = model(**batch)
    assert out.shape[0] == 1 and out.ndim == 3
