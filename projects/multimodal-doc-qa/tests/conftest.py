import os

import pytest

# transformers 5.x materializes weights on a ThreadPoolExecutor by default; copying tensors to MPS
# from worker threads segfaults on Apple Silicon (Metal allocation is not thread-safe). Any test that
# loads a model must therefore use synchronous loading. See PROJECT.md "Known pitfalls".
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")


@pytest.fixture(autouse=True)
def _no_ambient_mdq_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test as if no ``MDQ_*`` variable were exported.

    ``MDQ_*`` is how the pipeline is configured, so exporting one is normal and supported --
    ``MDQ_EMBEDDER_MODEL`` is how you skip a slow checkpoint load, and ``MDQ_ARTIFACTS_DIR``
    is how you point a run somewhere else. But `Settings` reads them, so an exported override
    silently becomes part of every test that touches a default: the suite then asserts the
    developer's shell rather than the code, and a CLI test can even write into the real
    artifacts directory. Clearing here makes the suite behave the same in any shell.

    A test that wants an override sets it itself with ``monkeypatch.setenv``, which lands
    after this fixture because autouse fixtures are set up first.
    """
    for name in list(os.environ):
        if name.startswith("MDQ_"):
            monkeypatch.delenv(name)
