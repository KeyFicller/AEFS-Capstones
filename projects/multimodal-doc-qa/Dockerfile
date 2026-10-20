# The API for multimodal-doc-qa only. Build from the repository root, because
# dependencies are declared only in the root requirements.txt:
#   docker build -f projects/multimodal-doc-qa/Dockerfile -t mdq-api .
#
# Run with the indices mounted and the keys passed in, never baked in:
#   docker run --rm -p 8000:8000 \
#     -v "$PWD/projects/multimodal-doc-qa/artifacts:/artifacts" \
#     -v "$HOME/.cache/huggingface:/cache/hf" --env-file local.env mdq-api
#
# The model cache is the host's own, so the container reuses checkpoints that
# are already downloaded instead of fetching them again into a named volume.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    MDQ_DEVICE=cpu \
    MDQ_DTYPE=float32 \
    MDQ_ARTIFACTS_DIR=/artifacts

# tesseract: page text for the OCR arms. libgl1/libglib: image codecs under torch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first, so this layer survives every source edit. Two edits to the shared
# file: the CUDA index line is Windows-only, and the editable lines for projects that are
# not in this image would fail to resolve. The CPU index carries a +cpu torch build, so
# the image does not ship the ~4GB of CUDA libraries a CPU container never loads.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v 'download.pytorch.org/whl/cu126' requirements.txt \
    | grep -vE '^-e \./(examples|projects/terminal-coding-agent|projects/tiny-llm-pipeline|components/telemetry)' \
    | grep -vE '^-e \./(projects|components)/' > /tmp/deps.txt \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/deps.txt

# Sources only. The editable installs carry no dependencies of their own, so --no-deps
# keeps this step to a metadata write instead of a second resolve.
COPY projects/multimodal-doc-qa/src ./projects/multimodal-doc-qa/src
COPY projects/multimodal-doc-qa/pyproject.toml ./projects/multimodal-doc-qa/pyproject.toml
COPY components/repl-console/src ./components/repl-console/src
COPY components/repl-console/pyproject.toml ./components/repl-console/pyproject.toml
COPY components/budget/src ./components/budget/src
COPY components/budget/pyproject.toml ./components/budget/pyproject.toml
COPY components/augment/src ./components/augment/src
COPY components/augment/pyproject.toml ./components/augment/pyproject.toml
COPY components/intent/src ./components/intent/src
COPY components/intent/pyproject.toml ./components/intent/pyproject.toml
RUN pip install --no-deps -e ./projects/multimodal-doc-qa \
    -e ./components/repl-console -e ./components/budget \
    -e ./components/augment -e ./components/intent

EXPOSE 8000
# One worker: sessions live in this process and the encoder is a process-local checkpoint,
# so a second worker would hold a second, disjoint set of both.
CMD ["uvicorn", "multimodal_doc_qa.api:app", "--host", "0.0.0.0", "--port", "8000"]
