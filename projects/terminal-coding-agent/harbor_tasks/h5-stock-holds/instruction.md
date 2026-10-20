`python3 -m pytest tests -q` 全绿，但仓库不满足 `README.md` 的「现行口径」：库存预留存在若干行为错误（过期回收、幂等键、确认/释放记账）。

按 `README.md` 的「现行口径」修好 `engine.py`。不要放宽或删除已有测试，也不要改 `tests/`。
