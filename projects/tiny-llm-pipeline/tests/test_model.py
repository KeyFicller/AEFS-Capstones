import torch

from tiny_llm_pipeline.config import CHINCHILLA_TOKENS_PER_PARAM, DEFAULT_MODEL, PRETRAIN_TOKENS
from tiny_llm_pipeline.model import TinyLM


def test_param_count_matches_chinchilla_budget() -> None:
    m = TinyLM(DEFAULT_MODEL)
    assert m.num_params() == 28_975_616
    assert PRETRAIN_TOKENS == CHINCHILLA_TOKENS_PER_PARAM * m.num_params()


def test_causal_mask() -> None:
    torch.manual_seed(0)
    m = TinyLM(DEFAULT_MODEL).eval()
    idx = torch.randint(0, DEFAULT_MODEL.vocab_size, (1, 6))
    a = m(idx)[0]
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % DEFAULT_MODEL.vocab_size
    b = m(idx2)[0]
    # Changing the last token must not change earlier positions.
    assert torch.allclose(a[0, :5], b[0, :5], atol=1e-5)
    # It must change the last position itself.
    assert not torch.allclose(a[0, 5], b[0, 5])


def test_repetition_penalty_lowers_seen_tokens_both_signs() -> None:
    """Dividing alone would raise a negative score, which is the opposite."""
    from tiny_llm_pipeline.model import _filter_logits

    logits = torch.tensor([[2.0, -2.0, 0.5]])
    _filter_logits(logits, torch.tensor([[0, 1]]), top_k=0, top_p=1.0, repetition_penalty=2.0)
    assert torch.allclose(logits, torch.tensor([[1.0, -4.0, 0.5]]))


def test_mps_smoke_one_step_loss_decreases() -> None:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = TinyLM(DEFAULT_MODEL).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, DEFAULT_MODEL.vocab_size, (4, 64), device=dev)
    losses = []
    for _ in range(20):
        _, loss = m(x, targets=x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
