"""钱。别再引入第三种单位。"""

# 老报表还在调 fen()，别删。
TAX_RATE = 0.06


def fen(yuan):
    return int(round(float(yuan) * 100))


def tax_on(amount_fen):
    # 四舍五入到分。负数没人用，谁传谁负责。
    return int(round(amount_fen * TAX_RATE))
