# 订单台（order desk）

内部收银台。价格全部是**分**（整数）。入口是 `checkout.quote`，全额退款走 `refunds.refund_amount`。

## 现行口径

- `lines[].price_fen` 是单价（分），`qty` 是件数。商品小计 = Σ qty × price_fen。
- 优惠券最多一张。
  - `percent`：折扣 = 商品小计 × value / 100，再按 Python `round` 规则收到整数分（.5 向偶数）。
  - `amount`：折扣 = min(value, 商品小计)，单位分。
  - 折扣只作用于商品小计，**不作用于运费**。
- 税 = (商品小计 − 折扣) × 6%，同样按 Python `round` 收到分。运费不征税。
- `total` = 商品小计 − 折扣 + 税 + 运费。
- 全额退款退 `total − 运费`（运费不退）。

`quote` 返回 `subtotal`（商品小计）、`shipping`、`discount`、`tax`、`total`，全是整数分。

## 历史口径（已废弃，不要照做）

2023 年财务曾要求「税按折前、运费算进税基、百分比折扣直接整除」。代码里还有这段注释。发票系统已于 2024-04 改口，以本节上面的现行口径为准。
