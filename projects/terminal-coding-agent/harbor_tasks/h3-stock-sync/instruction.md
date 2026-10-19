`python3 -m pytest tests -q` 有一个测试是红的。同一件商品写成 `ab-1` 和 `AB1` 时，库存被记成了两份。

进销存规则以 `README.md` 的「现行口径」为准。不要放宽或删除已有测试，也不要改 `tests/`。
