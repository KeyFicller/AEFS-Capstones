# 订阅账单（plan billing）

入口是 `billing.bill(subscription, today)`，返回今天要扣的金额（分），不该扣就是 0。不做按天折算。

`subscription` 字段：`plan`（`basic` / `pro`）、`status`、`started`（`YYYY-MM-DD`）、`trial_days`（整数）。

## 现行口径

- 价格：`basic` 2900，`pro` 9900。
- `status` 忽略大小写。`cancelled` 永不扣款。`trialing` 和 `active` 同一套日期规则。其它状态不扣。
- `today` 早于 `started` 不扣。
- 试用从 `started` 当天起算，含头含尾，共 `trial_days` 天。`trial_days=7` 且 `started=2024-01-01` 时，1 月 1 日到 1 月 7 日免费，**1 月 8 日首扣**。`trial_days=0` 则 `started` 当天首扣。
- 首扣日之后，每个月的同一日再扣一次。如果那个月没有这个日（例如 31 日遇到 2 月），改在该月最后一天扣，且只扣这一天。

## 历史口径（已废弃）

首发时试用「多送一天」补偿故障，2 月固定在 28 日扣 31 日锚点的用户，`cancelled` 只认全小写。这三条都已收回。
