#!/usr/bin/python3

from typing import Optional, List, Sequence, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal
import logging, subprocess, re
import asyncio, argparse
import jsonpickle


@dataclass
class Lot:
    date: datetime
    symbol: str
    quantity: Decimal
    price: Decimal


@dataclass
class Lots:
    lots: List[Lot] = field(default_factory=list)

    def for_symbol(self, symbol: str) -> "Lots":
        return Lots([l for l in self.lots if l.symbol == symbol])

    def get_last_buy_price(self, symbol: str) -> Optional[Decimal]:
        symbol_lots = [l for l in self.lots if l.symbol == symbol]
        if len(symbol_lots) == 0:
            return None
        held = sum([l.quantity for l in symbol_lots])
        if held == 0:
            return None
        sorted_lots = sorted(symbol_lots, key=lambda v: v.date)
        return sorted_lots[-1].price

    def get_basis(self, symbol: str) -> Optional[Decimal]:
        symbol_lots = [l for l in self.lots if l.symbol == symbol]
        total_quantity = sum([l.quantity for l in symbol_lots])
        if total_quantity == 0:
            return None
        return Decimal(
            sum([l.price * l.quantity for l in symbol_lots]) / total_quantity
        )


def parse(raw: str) -> Lots:
    lots: List[Lot] = []

    p = re.compile("(\S+)\s+(\S+)\s+{\$(\S+)} \[(\S+)\]")
    for line in raw.split("\n"):
        m = p.search(line)
        if m:
            quantity = Decimal(m.group(1))
            symbol = m.group(2)
            price = Decimal(m.group(3))
            date = datetime.strptime(m.group(4), "%y-%b-%d")
            lots.append(Lot(date, symbol, quantity, price))

    return Lots(lots)


@dataclass
class Ledger:
    path: str

    def lots(self, expression: List[str]) -> Lots:
        command = [
            "ledger",
            "-f",
            self.path,
            "balance",
            "--no-total",
            "--flat",
            "--lots",
        ] + expression
        sp = subprocess.run(command, stdout=subprocess.PIPE)

        log = logging.getLogger("ledger")
        log.info(" ".join(command).replace("\n", "NL"))

        return parse(sp.stdout.strip().decode("utf-8"))


async def main():
    parser = argparse.ArgumentParser(description="lots tool")
    parser.add_argument(
        "-f", "--ledger-file", action="store", default=None, required=True
    )
    parser.add_argument("-e", "--expression", action="store", default="stocks:")
    parser.add_argument("-t", "--today", action="store", default=None, required=False)
    parser.add_argument("-j", "--json", action="store", default="lots.json")
    args = parser.parse_args()

    ledger = Ledger(args.ledger_file)
    lots = ledger.lots([args.expression])

    with open(args.json, "w") as file:
        file.write(jsonpickle.encode(lots))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)7s] %(message)s")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
