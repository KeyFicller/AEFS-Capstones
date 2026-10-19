"""收银台。改之前先看 README，两套口径都有人背。"""

from discounts import apply_legacy
from money import tax_on


def _goods(order):
    total = 0
    for line in order.get("lines") or []:
        # qty 偶发是字符串，线上翻过车
        total += int(line.get("qty") or 0) * int(line.get("price_fen") or 0)
    return total


def quote(order):
    goods = _goods(order)
    shipping = int(order.get("shipping_fen") or 0)
    # 历史口径：运费并进税基。现行口径见 README，这里还没改。
    base = goods + shipping
    discount = apply_legacy(base, order.get("coupon"))
    # 税按折前，发票系统 2023 年的要求。
    tax = tax_on(base)
    total = base - discount + tax
    if total < 0:
        total = 0
    return {
        "subtotal": goods,
        "shipping": shipping,
        "discount": discount,
        "tax": tax,
        "total": total,
    }
