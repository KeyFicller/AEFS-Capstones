"""台账。单条调整会夹紧，导入批量不走这里。"""


class Ledger:
    def __init__(self):
        self.stock = {}

    def add(self, sku, delta):
        current = self.stock.get(sku, 0) + int(delta)
        if current < 0:
            current = 0
        self.stock[sku] = current
        return current
