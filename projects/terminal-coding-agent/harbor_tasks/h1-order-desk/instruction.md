`python3 -m pytest tests -q` 有一个测试是红的。收银台 `quote` 算出来的应付和仓库 `README.md` 里的「现行口径」不一致。

按现行口径修好计价和退款。不要放宽或删除已有测试，也不要改 `tests/`。
