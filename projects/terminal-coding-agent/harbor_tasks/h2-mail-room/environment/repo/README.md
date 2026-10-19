# 收发室（mail room）

进程内通知。没有队列，调用方下次再调 `dispatcher.dispatch`。

## 现行口径

`dispatch(job, now_hm, sender)`：

- `job` 含 `id`、`user_id`、`subject`、`channels`（频道名列表）。
- `now_hm` 是零填充的 `HH:MM`。
- `sender(channel, job)` 成功则该频道记入 `sent`；抛异常记入 `failed`，其它频道继续。
- 去重粒度是 `(user_id, id, channel)`，只记住**已经成功发送**的频道。主题不参与去重。
- 静默时段是 `[22:00, 07:00)`：22:00 静默，07:00 不静默，跨午夜。静默期内除 `sms` 外的频道记入 `skipped`，不发送、也不算已发送。
- 再次调用同一 job：已成功的频道不再发；上次失败的要重试；上次因静默跳过的，若现在不在静默期，要补发。
- 返回 `{"sent", "skipped", "failed", "suppressed"}`。`suppressed` 恒为 `False`（不再有「整单吞掉」）。

## 历史口径（已废弃）

上线第一周用邮件主题做全局去重，静默时段写成了闭区间，失败一次就整单标记完成。现行口径取代这三条。
