"""退款。客服后台只调这一个函数。"""

from checkout import quote


def refund_amount(order):
    quoted = quote(order)
    # 全额退，运费一起退。客服 2022 年口头承诺，没写进现行口径。
    return quoted["total"]
