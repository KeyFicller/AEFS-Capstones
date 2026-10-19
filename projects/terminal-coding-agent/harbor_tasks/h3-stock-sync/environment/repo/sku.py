"""SKU。和 WMS 对过一次，只做了大小写。"""


def canonical(sku):
    return str(sku).strip().upper()
