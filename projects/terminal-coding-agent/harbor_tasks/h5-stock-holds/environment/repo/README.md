# 库存预留（stock holds）

仓库存放短期预留：下单先占库存，付款后确认，超时或撤销后释放。入口是 `engine.Engine`。

时间由调用方传入（整数 tick，单调递增），引擎不读墙钟。所有数量是整数。

## 现行口径

- `Engine(stock)`：`stock` 是 `{sku: 初始库存}`。
- `available(sku, now)` = `stock[sku] − 已确认数量 − 活跃预留数量`。活跃 = 未过期、未释放、未确认。
- `snapshot(now)`：按**创建顺序**返回 dict 列表，只含 `active` 与 `confirmed` 的 hold；字段 `id` / `key` / `sku` / `qty` / `expires_at` / `state`。已过期、已释放的不出现。
- **过期**：`now >= expires_at` 即过期。每个公开操作开始前，必须先回收所有「已过期但仍活跃」的预留，把数量还回可用。
  - 边界：`now == expires_at` 时已经过期，不再占用库存。
- `reserve(key, sku, qty, ttl, now)`：
  - `qty <= 0` 或 `ttl <= 0` → `ValueError`；未知 `sku` → `KeyError`。
  - 幂等键是 `key`。同 key、同 `(sku, qty)` → 返回首次创建的那个 hold，**不重复占用库存**，也不因为该 hold 当前是 active / confirmed / released / expired 而改变返回值。
  - 同 key、不同 `(sku, qty)` → `ConflictError`。
  - 可用量不足 → 返回 `None`，不改变任何状态。
  - 成功 → 新 hold：`id` 依次为 `h1`、`h2`…（按创建顺序），`expires_at = now + ttl`，`state = "active"`。
- `confirm(hold_id, now)`：
  - 未知 id → `KeyError`。
  - `active` → 数量从「活跃」转入「已确认」；**`available` 不变**（同一份库存只是换了归属）。返回该 hold。
  - `confirmed` → 幂等，返回同一 hold，不重复计入已确认。
  - `released` / `expired` → `StateError`。
- `release(hold_id, now)`：
  - 未知 id → `KeyError`。
  - `active` → 数量回到可用，状态 `released`。
  - `released` / `expired` → 幂等 no-op。
  - `confirmed` → `StateError`（已消费的库存不能退回）。

## 不变量（对任意操作序列都成立）

1. 所有 sku 的 `available(sku, now) >= 0`。
2. 每个 sku：`已确认数量 + 活跃预留数量 <= stock[sku]`。
3. 同一 key 重复 `reserve` 不改变 `snapshot(now)`。
4. `snapshot` 不包含过期 hold。

## 历史说明

`engine.py` 里个别注释描述的是 2024 年的旧行为（那时过期只在读取库存时回收、幂等表按 key 覆盖最后一次请求）。现行口径以本文为准。
