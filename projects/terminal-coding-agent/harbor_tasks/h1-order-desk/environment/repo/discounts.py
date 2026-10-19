"""两套折扣。checkout 还在走 legacy，v2 是财务对过的，没切。"""


def apply_legacy(subtotal, coupon):
    # 百分比直接整除，和 2023 的 Excel 一致。
    if not coupon:
        return 0
    kind = coupon.get("type")
    value = int(coupon.get("value") or 0)
    if kind == "percent":
        return subtotal * value // 100
    if kind == "amount":
        return value if value < subtotal else subtotal
    return 0


def apply(subtotal, coupon):
    """现行算法。调用方自己保证 subtotal 不含运费。"""
    if not coupon:
        return 0
    kind = coupon.get("type")
    value = int(coupon.get("value") or 0)
    if kind == "percent":
        return int(round(subtotal * value / 100))
    if kind == "amount":
        return value if value < subtotal else subtotal
    return 0
