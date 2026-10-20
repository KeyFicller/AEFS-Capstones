`python3 -m pytest tests -q` 全绿，但按 `README.md` 的线上规模（约 12 万条事件）跑会超时。

在保持输出语义不变的前提下，让 `rollup.build_rollup` 在 8 秒内处理完 12 万条事件。不要改 `tests/`，也不要降低 `README.md` 的要求。
