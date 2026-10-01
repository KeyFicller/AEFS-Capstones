`orders/totals.py` 的 `compute_total(orders)` 现在把所有订单的 `amount` 相加。

需求变更：**已取消订单（`status == "cancelled"`）不计入总计**，其余订单照旧计入。

`python3 -m pytest tests -q` 现在是全绿的，只是缺少取消订单的覆盖。按上面的需求修改实现，
可以自行补充测试；不要放宽或删除已有测试。
