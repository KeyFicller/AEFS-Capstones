"""Byte-level BPE trained on the pretrain sample, plus the chat template.

Special token ids are fixed: pad, bos, eos, user, assistant, end are 0..5.
Loss is on assistant text and the following end token, not on the assistant
token itself.
"""

import hashlib
import tempfile
from pathlib import Path

from tokenizers import AddedToken, Tokenizer
from tokenizers import decoders, pre_tokenizers, trainers
from tokenizers.models import BPE

SPECIAL_TOKENS: list[str] = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]

_BOS, _USER, _ASSISTANT, _END = (
    "<|bos|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
)


class Tok:
    """Loaded tokenizer. `hash` is the first 16 hex chars of tokenizer.json."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._tok = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        """Encode text. A lone special token is a single id, not byte pieces."""
        if text in SPECIAL_TOKENS:
            token_id = self._tok.token_to_id(text)
            if token_id is None:
                raise KeyError(f"special token {text} is missing from the vocabulary")
            return [token_id]
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decode ids back to text."""
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def hash(self) -> str:
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return digest[:16]


def train_tokenizer(
    text_path: Path | str,
    out_dir: Path | str,
    vocab_size: int = 6400,
    sample_mb: int = 200,
) -> Path:
    """Train a byte-level BPE on at most `sample_mb` megabytes and save it."""
    source = Path(text_path)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    train_path, cleanup = _sample_path(source, sample_mb)
    try:
        tok = _fit(train_path, vocab_size)
    finally:
        if cleanup is not None:
            cleanup.unlink(missing_ok=True)
    out = dest / "tokenizer.json"
    tok.save(str(out))
    ids = [tok.token_to_id(token) for token in SPECIAL_TOKENS]
    if ids != list(range(len(SPECIAL_TOKENS))):
        raise RuntimeError(f"special token ids {ids}, expected 0..{len(SPECIAL_TOKENS) - 1}")
    return out


def load_tokenizer(directory: Path | str) -> Tok:
    """Load the tokenizer saved in `directory`."""
    return Tok(Path(directory) / "tokenizer.json")


def render_chat(messages: list[dict[str, str]], add_assistant: bool) -> str:
    """Render the chat template with no extra spaces or newlines."""
    parts = [_BOS]
    for message in messages:
        role = _USER if message["role"] == "user" else _ASSISTANT
        parts.append(f"{role}{message['content']}{_END}")
    if add_assistant and (not messages or messages[-1]["role"] != "assistant"):
        parts.append(_ASSISTANT)
    return "".join(parts)


def encode_chat(
    messages: list[dict[str, str]], tok: Tok, add_assistant: bool = False
) -> tuple[list[int], list[bool]]:
    """Encode a chat by pieces. The bools are the loss mask."""
    bos = tok.encode(_BOS)[0]
    user = tok.encode(_USER)[0]
    assistant = tok.encode(_ASSISTANT)[0]
    end = tok.encode(_END)[0]
    ids = [bos]
    mask = [False]
    for message in messages:
        role_id = user if message["role"] == "user" else assistant
        piece = tok.encode(message["content"])
        on_assistant = message["role"] == "assistant"
        ids.extend([role_id, *piece, end])
        mask.extend([False, *([on_assistant] * (len(piece) + 1))])
    if add_assistant and (not messages or messages[-1]["role"] != "assistant"):
        ids.append(assistant)
        mask.append(False)
    return ids, mask


def _sample_path(source: Path, sample_mb: int) -> tuple[Path, Path | None]:
    """Return a training file and a temp path to delete, if one was created."""
    raw = source.read_bytes()
    limit = sample_mb * 1024 * 1024
    if len(raw) <= limit:
        return source, None
    text = raw[:limit].decode("utf-8", errors="ignore")
    handle = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    path = Path(handle.name)
    handle.close()
    path.write_text(text, encoding="utf-8")
    return path, path


def _fit(train_path: Path, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken(token, special=True) for token in SPECIAL_TOKENS])
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train([str(train_path)], trainer)
    return tok
