"""Sampling from the three stage checkpoints, shared by the eval scripts.

`pretrain` is a base model, so its prompt is a raw continuation. `sft` and `dpo`
are chat models, so theirs go through the chat template with a trailing
`<|assistant|>`.

Under sampling every generation gets a seed derived from
(seed, stage, rep_pen, prompt, index), so a reply is reproducible regardless of
the order prompts are run in, and repeated samples differ only by `index`.

Readings per reply:

  chars     length of the reply, capped by `max_new`
  dist4     distinct 4-grams / total 4-grams. 1.0 is rich, low means looping.
  tempo     occurrences of the training template opener ("我用最直白").
            DPO over-fits this prefix; a good repetition penalty keeps it low.

`dist4` saturates around 1.2 while the template tail keeps collapsing to
meaningless `最X` stacks up to about 1.4, so the metric alone underpicks. Check
the text too. `rep_pen=1.5` is the delivered value; see PROJECT.md.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tiny_llm_pipeline.model import load_ckpt
from tiny_llm_pipeline.tokenizer import Tok, encode_chat, load_tokenizer
from tiny_llm_pipeline.train.pretrain import _device

TEMPLATE = "我用最直白"
STAGES: tuple[str, ...] = ("pretrain", "sft", "dpo")


def seed_for(seed: int, stage: str, rep_pen: float, prompt: str, index: int) -> int:
    """A stable 32-bit seed for this generation. Independent of run order."""
    parts = (seed, stage, rep_pen, prompt, index)
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def stats(text: str) -> dict[str, object]:
    """The three readings reported for one reply."""
    grams = [text[i : i + 4] for i in range(max(0, len(text) - 3))]
    return {
        "chars": len(text),
        "dist4": len(set(grams)) / max(1, len(grams)),
        "tempo": text.count(TEMPLATE),
    }


@dataclass(frozen=True)
class GenSettings:
    """Sampling knobs. Defaults match `eval/sample.py`'s argparse defaults."""

    max_new: int = 64
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 1.0
    rep_pen: float = 1.5
    seed: int = 0


@dataclass(frozen=True)
class StageReply:
    """One stage's answer to one question."""

    stage: str
    step: int
    reply: str
    stats: dict[str, object]


class Generator:
    """One stage's weights, already loaded and on `device`."""

    def __init__(
        self,
        stage: str,
        model,
        tok: Tok,
        end_id: int,
        device: torch.device,
        step: int = 0,
    ) -> None:
        self.stage = stage
        self.model = model
        self.tok = tok
        self.end_id = end_id
        self.device = device
        self.step = step

    def reply(self, prompt: str, settings: GenSettings, index: int = 0) -> str:
        """One reply to `prompt`, truncated at `<|end|>` and stripped."""
        if self.stage == "pretrain":
            ids = self.tok.encode(prompt)
        else:
            ids, _ = encode_chat(
                [{"role": "user", "content": prompt}], self.tok, add_assistant=True
            )
        torch.manual_seed(
            seed_for(settings.seed, self.stage, settings.rep_pen, prompt, index)
        )
        with torch.no_grad():
            out = self.model.generate(
                torch.tensor([ids], dtype=torch.long, device=self.device),
                settings.max_new,
                temperature=settings.temperature,
                top_k=settings.top_k,
                top_p=settings.top_p,
                repetition_penalty=settings.rep_pen,
            )
        new = out[0, len(ids) :].tolist()
        if self.end_id in new:
            new = new[: new.index(self.end_id) + 1]
        return self.tok.decode(new).strip()


@dataclass
class Bundle:
    """The three stages' weights plus the tokenizer they share."""

    tok: Tok
    device: torch.device
    generators: dict[str, Generator] = field(default_factory=dict)

    @property
    def stages(self) -> tuple[str, ...]:
        """The loaded stages, in the order they were asked for."""
        return tuple(self.generators)

    @property
    def steps(self) -> dict[str, int]:
        """Checkpoint step per stage."""
        return {stage: gen.step for stage, gen in self.generators.items()}

    @classmethod
    def load(
        cls,
        root: Path | str,
        *,
        stages: tuple[str, ...] = STAGES,
        device: torch.device | None = None,
    ) -> "Bundle":
        """Load `root/{tokenizer,<stage>/ckpt.pt}` once and keep them resident."""
        root = Path(root)
        tok = load_tokenizer(root / "tokenizer")
        dev = _device() if device is None else device
        end_id = tok.encode("<|end|>")[0]
        generators = {}
        for stage in stages:
            loaded = load_ckpt(root / stage / "ckpt.pt", expect_tokenizer_hash=tok.hash)
            model = loaded["model"].to(dev).eval()
            generators[stage] = Generator(
                stage, model, tok, end_id, dev, step=int(loaded["step"])
            )
        return cls(tok=tok, device=dev, generators=generators)

    def run(self, question: str, settings: GenSettings) -> list[StageReply]:
        """Ask every loaded stage the same question, in stage order."""
        replies = []
        for stage, generator in self.generators.items():
            text = generator.reply(question, settings)
            replies.append(
                StageReply(stage=stage, step=generator.step, reply=text, stats=stats(text))
            )
        return replies
