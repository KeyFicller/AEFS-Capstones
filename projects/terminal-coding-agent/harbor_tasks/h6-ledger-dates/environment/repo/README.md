# 订阅账单（ledger）

按自然月给订阅计费：套餐月费 + 附加项，按实际在网天数比例摊分，最后应用信用额度。

入口是 `api.handlers.get_invoice(repo, tenant, period)`。金额单位是**分**，全部为整数。

## 现行口径

### 计费区间

- `period` 是 `"YYYY-MM"`，代表该自然月的**全部日期**。
- 订阅 `start_date` / `end_date` 都是**闭区间**（`end_date` 为 `None` 表示长期有效）。
- 某天的「在网」定义：`start_date <= day <= end_date`。
- 某条收费项在当月应收 = `月费 × 当月在网天数 / 当月天数`，**先保留精确有理数，不提前取整**。
- 附加项只在**订阅在网期间**计费；订阅与附加项的在网区间取交集。

### 取整

- 只有最终金额才取整，规则是 **round half to even**（.5 向偶数分取整）。
- `rounding_mode` 两种：
  - `per_invoice`（**现行默认**）：把当月所有收费项的精确金额相加，**只对合计取整一次**。
  - `per_line`：每条收费项各自取整，再求和。
- `lines[].amount_fen` 是**展示值**：每条按 half-to-even 单独取整。`per_invoice` 下 `sum(lines)` 可能比 `total_fen` 少/多 1 分，这是预期行为。

### 信用额度

- 每个租户有一笔余额（分），只能为正。
- 账单先算出 `subtotal_fen`（取整后），再抵扣：`applied = min(balance, subtotal)`，`total_fen = subtotal − applied`。
- 未用完的余额**结转**到下月：`credit_carried_fen = balance − applied`。
- 抵扣发生在取整之后，不参与取整。

### 输出

`get_invoice` 返回：

```
{
  "tenant": str,
  "period": "YYYY-MM",
  "rounding": "per_invoice" | "per_line",
  "lines": [{"label": str, "amount_fen": int}, ...],   # 展示值
  "subtotal_fen": int,
  "credit_applied_fen": int,
  "credit_carried_fen": int,
  "total_fen": int,
}
```

- `lines` 顺序：套餐项在前，附加项按 `code` 升序。
- 当月完全不在网 → `lines` 为空、各金额为 0，信用余额不变。
- 未知 `tenant` → `KeyError`；`period` 格式错误 → `ValueError`。

### 配置

`config.settings.from_env()` 读取环境变量 `LEDGER_ROUNDING_MODE`；**未设置时必须是 `per_invoice`**。调用方也可以显式传入 `Settings(rounding_mode=...)`。

## 历史说明

`config/settings.py` 里的默认值是 2023 年结算导出任务用的旧口径，导出任务至今仍显式设置该环境变量。现行默认以本文为准。
