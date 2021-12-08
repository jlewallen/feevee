from typing import List, Optional, Any, Dict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import sqlite3, os, logging
import jsonpickle

log = logging.getLogger("storage")


@dataclass
class UserKey:
    uid: int
    generation: str


@dataclass
class SymbolRow:
    symbol: str
    modified: datetime
    data: str


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
            "CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id), symbol TEXT NOT NULL, created DATETIME NOT NULL, noted_price DECIMAL(10, 5) NOT NULL, future_price DECIMAL(10, 5), body TEXT NOT NULL)"
        )
        self.dbc.execute(
            "CREATE INDEX IF NOT EXISTS notes_symbol_ts_idx ON notes (user_id, symbol, created)"
        )
        self.db.commit()

    def close(self):
        self.db.close()

    def get_user_key_by_user_id(self, user_id: int) -> List[UserKey]:
        return [
            UserKey(row[0], str(row[1]))
            for row in self.dbc.execute(
                "SELECT id, modified, MAX(notes.created) AS notes_modified FROM users LEFT JOIN notes ON (users.id = notes.user_id) WHERE users.id = ? GROUP BY users.id, users.modified",
                [user_id],
            )
        ]

    def get_all_user_keys(self) -> List[UserKey]:
        return [
            UserKey(row[0], str(row[1]))
            for row in self.dbc.execute(
                "SELECT id, modified, MAX(notes.created) AS notes_modified FROM users LEFT JOIN notes ON (users.id = notes.user_id) GROUP BY users.id, users.modified"
            )
        ]

    def has_symbol(self, symbol: str):
        return self.get_symbol(symbol) is not None

    def get_symbol(self, symbol: str):
        for row in self.dbc.execute(
            "SELECT modified, data FROM symbols WHERE symbol = ?", [symbol]
        ):
            return SymbolRow(symbol, row[0], row[1])
        return None

    def get_all_symbols(self, user_key: UserKey):
        rows = self.dbc.execute(
            "SELECT symbol, modified, data FROM symbols WHERE symbol IN (SELECT symbol FROM user_symbol WHERE user_id = ?) ORDER BY symbol",
            [user_key.uid],
        )
        return [SymbolRow(row[0], row[1], row[2]) for row in rows]

    def add_symbols(self, user_key: UserKey, symbols: List[str]):
        for symbol in symbols:
            self.dbc.execute(
                "INSERT INTO user_symbol (user_id, symbol) VALUES (?, ?) ON CONFLICT DO NOTHING",
                [user_key.uid, symbol],
            )

        self._user_modified(user_key)

        self.db.commit()

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

    def get_notes(self, user_key: UserKey, symbol: str):
        notes = []
        for row in self.dbc.execute(
            "SELECT created, noted_price, future_price, body FROM notes WHERE user_id = ? AND symbol = ? ORDER BY created DESC LIMIT 1",
            [user_key.uid, symbol],
        ):
            created = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
            notes.append(
                NoteRow(
                    symbol,
                    created,
                    Decimal(row[1]),
                    Decimal(row[2]) if row[2] else None,
                    row[3],
                )
            )
        return notes

    def get_all_notes(self, user_key: UserKey) -> Dict[str, List[NoteRow]]:
        notes = {}
        for row in self.dbc.execute(
            "SELECT symbol, created, noted_price, future_price, body FROM notes WHERE user_id = ? ORDER BY created DESC",
            [user_key.uid],
        ):
            symbol = row[0]
            if symbol in notes:
                continue

            created = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S.%f")
            note_row = NoteRow(
                symbol,
                created,
                Decimal(row[2]),
                Decimal(row[3]) if row[3] else None,
                row[4],
            )
            notes[symbol] = [note_row]
        return notes

    def add_notes(
        self,
        user_key: UserKey,
        symbol: str,
        created: datetime,
        noted_price: Decimal,
        future_price: Optional[Decimal],
        body: str,
    ):
        self.dbc.execute(
            "INSERT INTO notes (user_id, symbol, created, noted_price, future_price, body) VALUES (?, ?, ?, ?, ?, ?)",
            [
                user_key.uid,
                symbol,
                created,
                str(noted_price),
                str(future_price) if future_price else None,
                body,
            ],
        )

        # self._user_modified(user_key)

        self.db.commit()

    def _user_modified(self, user_key: UserKey):
        self.dbc.execute(
            "UPDATE users SET modified = ? WHERE id = ?",
            [
                datetime.utcnow(),
                user_key.uid,
            ],
        )
