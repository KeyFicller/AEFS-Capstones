`python3 -m metrics sample.txt` 应该输出：

```
lines=2
words=5
chars=31
```

并以退出码 0 结束；`python3 -m metrics --labelled sample.txt` 应该输出 `Lines: 2` /
`Words: 5` / `Chars: 31` 三行。

现在两条命令都以退出码 1 结束，stderr 末尾是 `KeyError: 'lines'`。修好它。
