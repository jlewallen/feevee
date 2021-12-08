from typing import List, Optional, Any, Dict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import sqlite3, os, logging
import jsonpickle

log = logging.getLogger("storage")


@dataclass
class SymbolRow:
    symbol: str
    modified: datetime
    data: str


@dataclass
class TradeRow:
    symbol: str
    ts: datetime
    price: Decimal
    quantity: Decimal


@dataclass
class BarRow:
    symbol: str
    ts: datetime
    volume: Decimal
    high: Decimal
    low: Decimal
    opening: Decimal
    vwap: Decimal


@dataclass
class NoteRow:
    symbol: str
    ts: datetime
    noted_price: Decimal
    future_price: Optional[Decimal]
    body: str


class SymbolStorage:
    def open(self, path: Optional[str] = None, name: str = "feevee.db"):
        path = os.path.join(path if path else os.environ["MONEY_CACHE"], name)
        self.db = sqlite3.connect(path)
        self.dbc = self.db.cursor()
        self.dbc.execute(
            "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, created DATETIME NOT NULL, modified DATETIME NOT NULL, email TEXT NOT NULL, password TEXT NOT NULL)"
        )
        self.dbc.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS users_email_idx ON users (email)"
        )
        self.dbc.execute(
            "CREATE TABLE IF NOT EXISTS user_symbol (user_id INTEGER NOT NULL REFERENCES users(id), symbol TEXT NOT NULL)"
        )
        self.dbc.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS user_symbol_idx ON user_symbol (user_id, symbol)"
        )
        self.dbc.execute(
            "CREATE TABLE IF NOT EXISTS user_lot (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id), symbol TEXT NOT NULL, purchased DATETIME NOT NULL, quantity TEXT NOT NULL, price TEXT NOT NULL)"
        )
        self.dbc.execute(
            "CREATE INDEX IF NOT EXISTS user_lot_idx ON user_lot (user_id, symbol)"
        )
        self.dbc.execute(
            "CREATE TABLE IF NOT EXISTS user_lot_ledger (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id), modified DATETIME NOT NULL, body TEXT NOT NULL)"
        )
        self.dbc.execute(
            "CREATE INDEX IF NOT EXISTS user_lot_ledger_idx ON user_lot_ledger (user_id)"
        )

        self.dbc.execute(
            "CREATE TABLE IF NOT EXISTS symbols (symbol TEXT NOT NULL, modified DATETIME NOT NULL, data TEXT NOT NULL)"
        )
        self.dbc.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS symbols_symbol_idx ON symbols (symbol)"
        )

        self.dbc.execute(
            "CREATE TABLE IF NOT EXISTS bars (symbol TEXT NOT NULL, ts DATETIME NOT NULL, high DECIMAL(10, 5), low DECIMAL(10, 5), volume DECIMAL(10, 5), open DECIMAL(10, 5), vwap DECIMAL(10, 5), trade_count DECIMAL(10, 5))"
        )
        self.dbc.execute(
            "CREATE INDEX IF NOT EXISTS bars_symbol_ts_idx ON bars (symbol, ts)"
        )

        self.dbc.execute(
            "CREATE TABLE IF NOT EXISTS trades (symbol TEXT NOT NULL, ts DATETIME NOT NULL, price DECIMAL(10, 5), quantity DECIMAL(10, 5))"
        )
        self.dbc.execute(
            "CREATE INDEX IF NOT EXISTS trades_symbol_ts_idx ON trades (symbol, ts)"
        )

        self.dbc.execute(
            "CREATE TABLE IF NOT EXISTS notes (symbol TEXT NOT NULL, ts DATETIME NOT NULL, noted_price DECIMAL(10, 5) NOT NULL, future_price DECIMAL(10, 5), body TEXT NOT NULL)"
        )
        self.dbc.execute(
            "CREATE INDEX IF NOT EXISTS notes_symbol_ts_idx ON notes (symbol, ts)"
        )
        self.db.commit()

    def close(self):
        self.db.close()

    def has_symbol(self, symbol: str):
        return self.get_symbol(symbol) is not None

    def get_symbol(self, symbol: str):
        for row in self.dbc.execute(
            "SELECT modified, data FROM symbols WHERE symbol = ?", [symbol]
        ):
            return SymbolRow(symbol, row[0], row[1])
        return None

    def get_all_symbols(self):
        rows = self.dbc.execute(
            "SELECT symbol, modified, data FROM symbols ORDER BY symbol"
        )
        return {row[0]: SymbolRow(row[0], row[1], row[2]) for row in rows}

    def set_symbol(self, symbol: str, data: Any):
        serialized = jsonpickle.encode(data)
        log.info(f"{serialized}")
        self.dbc.execute(
            """
        INSERT INTO symbols (symbol, modified, data) VALUES (?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET modified = excluded.modified, data = excluded.data
  """,
            [symbol, datetime.now(), serialized],
        )
        self.db.commit()

    def has_trade(self, symbol: str):
        return self.get_trade(symbol) is not None

    def get_trade(self, symbol: str):
        for row in self.dbc.execute(
            "SELECT ts, price, quantity FROM trades WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
            [symbol],
        ):
            ts = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
            return TradeRow(symbol, ts, Decimal(row[1]), Decimal(row[2]))
        return None

    def add_trade(self, symbol: str, ts: datetime, price: Decimal, quantity: Decimal):
        self.dbc.execute(
            "INSERT INTO trades (symbol, ts, price, quantity) VALUES (?, ?, ?, ?)",
            [symbol, ts, price, quantity],
        )
        self.db.commit()

    def has_bars(self, symbol: str):
        return len(self.get_bars(symbol)) > 0

    def get_bars(self, symbol: str):
        bars = []
        for row in self.dbc.execute(
            "SELECT ts, volume, high, low, open, vwap FROM bars WHERE symbol = ? AND ts >= date('now', '-1 days') ORDER BY ts ASC",
            [symbol],
        ):
            ts = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
            bars.append(
                BarRow(
                    symbol,
                    ts,
                    Decimal(row[1]),
                    Decimal(row[2]),
                    Decimal(row[3]),
                    Decimal(row[4]),
                    Decimal(row[5]),
                )
            )
        return bars

    def add_bars(
        self,
        symbol: str,
        ts: datetime,
        high: Decimal,
        low: Decimal,
        volume: Decimal,
        opening: Decimal,
        vwap: Decimal,
        trade_count: Decimal,
    ):
        self.dbc.execute(
            "INSERT INTO bars (symbol, ts, high, low, volume, open, vwap, trade_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                symbol,
                ts,
                str(high),
                str(low),
                str(volume),
                str(opening),
                str(vwap),
                str(trade_count),
            ],
        )
        self.db.commit()

    def get_notes(self, symbol: str):
        notes = []
        for row in self.dbc.execute(
            "SELECT ts, noted_price, future_price, body FROM notes WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
            [symbol],
        ):
            ts = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
            notes.append(
                NoteRow(
                    symbol,
                    ts,
                    Decimal(row[1]),
                    Decimal(row[2]) if row[2] else None,
                    row[3],
                )
            )
        return notes

    def get_all_notes(self) -> Dict[str, List[NoteRow]]:
        notes = {}
        for row in self.dbc.execute(
            "SELECT symbol, ts, noted_price, future_price, body FROM notes ORDER BY ts DESC",
        ):
            symbol = row[0]
            if symbol in notes:
                continue

            ts = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S.%f")
            note_row = NoteRow(
                symbol,
                ts,
                Decimal(row[2]),
                Decimal(row[3]) if row[3] else None,
                row[4],
            )
            notes[symbol] = [note_row]
        return notes

    def add_notes(
        self,
        symbol: str,
        ts: datetime,
        noted_price: Decimal,
        future_price: Optional[Decimal],
        body: str,
    ):
        self.dbc.execute(
            "INSERT INTO notes (symbol, ts, noted_price, future_price, body) VALUES (?, ?, ?, ?, ?)",
            [
                symbol,
                ts,
                str(noted_price),
                str(future_price) if future_price else None,
                body,
            ],
        )
        self.db.commit()
