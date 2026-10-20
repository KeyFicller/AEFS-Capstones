# 事件汇总（event rollup）

按租户 / 分钟汇总事件，同时给出会话数与最大间隔。入口是 `rollup.build_rollup(events)`。

## 输入

`events` 是 dict 列表，字段：

- `tenant`：str
- `user`：str
- `ts_ms`：int，毫秒时间戳，`>= 0`
- 其他字段忽略

顺序任意，允许重复时间戳。

- 缺字段 → `KeyError`
- `ts_ms` 为负 → `ValueError`
- `ts_ms` 不是 int → `TypeError`

## 会话（session）

同一 `(tenant, user)` 的事件按 `ts_ms` 升序排列。相邻两条事件间隔 `<= 30 × 60 × 1000` 毫秒即属于同一会话。

链式传递：0 / 25 / 50 分钟三条事件是一个会话（首尾超过 30 分钟也算）。

会话 id = `"{tenant}:{user}:{会话首条事件的 ts_ms}"`。

## 输出

```
{tenant: {minute: {"count": int, "unique_sessions": int, "max_gap_ms": int}}}
```

- `minute = ts_ms // 60000`
- 只包含出现过事件的租户 / 分钟，空输入返回 `{}`。
- `count`：该租户这一分钟内的事件数。
- `unique_sessions`：这一分钟内出现过至少一条事件的会话数。
- `max_gap_ms`：同一会话的**相邻**两条事件若都落在这一分钟内，取二者间隔；取所有这类相邻对的最大值。没有这种相邻对时为 `0`（不同分钟之间的间隔不计入任何一桶）。

## 性能

线上一次回放约 12 万条事件，要求单次 `build_rollup` 在普通容器中 8 秒内完成。

当前实现在小样本上结果正确，但按线上规模回放会超时。
