from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from aiocache import cached, Cache
import aiosqlite, os, logging, jsonpickle

log = logging.getLogger("storage")

UserId = int


@dataclass
class Criteria:
    symbol: Optional[str] = None
    page: Optional[int] = None


@dataclass
class UserKey:
    uid: UserId
    modified: datetime
    symbols: Dict[str, datetime] = field(default_factory=dict)

    @property
    def symbol_key(self) -> str:
        values = list(self.symbols.values())
        if values:
            return str(max(values))
        return ""

    def __str__(self):
        return f"UserKey<{self.uid}, {self.modified}, {self.symbol_key}>"

    def __repr__(self):
        return str(self)


@dataclass
class SymbolRow:
    symbol: str
    modified: datetime
    earnings: bool
    candles: bool
    options: bool
    data: str


@dataclass
class NoteRow:
    symbol: str
    ts: datetime
    body: str


@dataclass
class UserSymbol:
    symbol: SymbolRow
    notes: Optional[NoteRow]


@dataclass
class SymbolStorage:
    db: Optional[aiosqlite.Connection] = None

    async def open(self, name: str = "feevee.db", path: Optional[str] = None):
        log.info(f"db:opening")
        path = os.path.join(path if path else os.environ["MONEY_CACHE"], name)
        self.db = await aiosqlite.connect(path)
        await self.db.execute(
            "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, created DATETIME NOT NULL, modified DATETIME NOT NULL, email TEXT NOT NULL, password TEXT NOT NULL)"
        )
        await self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS users_email_idx ON users (email)"
        )
        await self.db.execute(
            "CREATE TABLE IF NOT EXISTS user_symbol (user_id INTEGER NOT NULL REFERENCES users(id), symbol TEXT NOT NULL)"
        )
        await self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS user_symbol_idx ON user_symbol (user_id, symbol)"
        )
        await self.db.execute(
            "CREATE TABLE IF NOT EXISTS user_lot (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id), symbol TEXT NOT NULL, purchased DATETIME NOT NULL, quantity TEXT NOT NULL, price TEXT NOT NULL)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS user_lot_idx ON user_lot (user_id, symbol)"
        )
        await self.db.execute(
            "CREATE TABLE IF NOT EXISTS user_lot_ledger (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id), modified DATETIME NOT NULL, body TEXT NOT NULL)"
        )
        await self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS user_lot_ledger_idx ON user_lot_ledger (user_id)"
        )

        await self.db.execute(
            "CREATE TABLE IF NOT EXISTS symbols (symbol TEXT NOT NULL, modified DATETIME NOT NULL, safe BOOL NOT NULL, earnings BOOL NOT NULL, candles BOOL NOT NULL, options BOOL NOT NULL, data TEXT NOT NULL)"
        )
        await self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS symbols_symbol_idx ON symbols (symbol)"
        )

        await self.db.execute(
            "CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id), symbol TEXT NOT NULL, modified DATETIME NOT NULL, noted_price DECIMAL(10, 5) NOT NULL, future_price DECIMAL(10, 5), body TEXT NOT NULL)"
        )
        await self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS notes_idx ON notes (user_id, symbol)"
        )
        await self.db.commit()

    async def close(self):
        assert self.db
        await self.db.close()

    async def get_user_key_by_user_id(self, user_id: int) -> UserKey:
        assert self.db
        dbc = await self.db.execute(
            """
            SELECT u.id, u.modified, MAX(n.modified) AS notes_modified
            FROM users AS u LEFT JOIN notes AS n ON (u.id = n.user_id)
            WHERE u.id = ?
            GROUP BY u.id, u.modified
            """,
            [user_id],
        )
        symbol_keys = await self._get_user_symbol_keys(user_id)
        found = [
            UserKey(row[0], self._parse_datetime(row[1]), symbol_keys)
            for row in await dbc.fetchall()
        ]
        assert found
        return found[0]

    async def get_all_user_ids(self) -> List[UserId]:
        assert self.db
        dbc = await self.db.execute("SELECT id FROM users")
        return [row[0] for row in await dbc.fetchall()]

    async def get_all_notes(self, user_key: UserKey) -> Dict[str, NoteRow]:
        assert self.db
        notes = {}
        dbc = await self.db.execute(
            "SELECT symbol, modified, noted_price, future_price, body FROM notes WHERE user_id = ? ORDER BY modified DESC",
            [user_key.uid],
        )
        for row in await dbc.fetchall():
            symbol = row[0]
            if symbol in notes:
                continue

            note_row = NoteRow(
                symbol,
                self._parse_datetime(row[1]),
                row[4],
            )
            notes[symbol] = note_row
        return notes

    async def get_all_symbols(
        self, user_key: UserKey, criteria: Criteria
    ) -> Dict[str, UserSymbol]:
        assert self.db

        dbc = await self.db.execute(
            """
            SELECT s.symbol, s.modified, s.earnings, s.candles, s.options, s.data, notes.modified, notes.body AS notes
            FROM symbols AS s
            LEFT JOIN notes ON (s.symbol = notes.symbol)
            WHERE s.safe AND s.symbol IN (SELECT symbol FROM user_symbol WHERE user_id = ?)
            AND (? is NULL OR s.symbol = ?)
            ORDER BY s.symbol
            """,
            [user_key.uid, criteria.symbol, criteria.symbol],
        )

        rows = await dbc.fetchall()

        if criteria.page:
            rows = list(rows)
            rows = rows[criteria.page * 10 : criteria.page * 10 + 10]

        symbols = {
            row[0]: UserSymbol(
                SymbolRow(
                    row[0],
                    self._parse_datetime(row[1]),
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                ),
                NoteRow(
                    row[0],
                    self._parse_datetime(row[6]),
                    row[7],
                )
                if row[6]
                else None,
            )
            for row in rows
        }

        filtering: Optional[List[str]] = None
        if "FEEVEE_SYMBOLS" in os.environ:
            filtering = os.environ["FEEVEE_SYMBOLS"].split(" ")

        return {
            key: value
            for key, value in symbols.items()
            if filtering is None or key in filtering
        }

    async def add_symbols(self, user_key: UserKey, symbols: List[str]) -> List[str]:
        assert self.db

        changed: List[str] = []
        for symbol in symbols:
            # TODO Make data nullable in the future.
            await self.db.execute(
                """
                INSERT INTO symbols (symbol, modified, candles, earnings, candles, safe, data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                [symbol, datetime.utcnow(), True, False, False, False, "{}"],
            )

            changes_before = self.db.total_changes
            await self.db.execute(
                "INSERT INTO user_symbol (user_id, symbol) VALUES (?, ?) ON CONFLICT DO NOTHING",
                [user_key.uid, symbol],
            )
            if changes_before != self.db.total_changes:
                log.info(f"{symbol:6} added")
                changed.append(symbol)

        if len(changed) > 0:
            await self._user_modified(user_key)

        await self.db.commit()

        return changed

    async def remove_symbols(self, user_key: UserKey, symbols: List[str]) -> List[str]:
        assert self.db

        for symbol in symbols:
            await self.db.execute(
                "DELETE FROM user_symbol WHERE user_id = ? AND symbol = ?",
                [user_key.uid, symbol],
            )
            log.info(f"{symbol:6} removed")

        await self._user_modified(user_key)

        await self.db.commit()

        return []

    async def set_symbol(self, symbol: str, safe: bool, data: Dict[str, Any]):
        assert self.db
        serialized = jsonpickle.encode(data)
        log.info(f"{serialized}")
        await self.db.execute(
            """
            INSERT INTO symbols (symbol, modified, candles, options, earnings, safe, data) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE
            SET modified = excluded.modified, safe = excluded.safe, data = excluded.data
            """,
            [symbol, datetime.now(), True, False, False, safe, serialized],
        )

        await self.db.commit()

    async def update_lots(self, user_key: UserKey, lots: str):
        assert self.db
        await self.db.execute(
            """
            INSERT INTO user_lot_ledger (user_id, modified, body) VALUES (?, ?, ?) 
            ON CONFLICT(user_id) DO UPDATE
            SET modified = excluded.modified, body = excluded.body
            """,
            [
                user_key.uid,
                datetime.utcnow(),
                lots,
            ],
        )

        await self._user_modified(user_key)

        updated_user_key = await self.get_user_key_by_user_id(user_key.uid)

        await self.db.commit()

        return updated_user_key

    async def get_lots(self, user_key: UserKey) -> str:
        assert self.db
        dbc = await self.db.execute(
            "SELECT user_id, modified, body FROM user_lot_ledger WHERE user_id = ?",
            [user_key.uid],
        )
        for row in await dbc.fetchall():
            return row[2]
        return ""

    async def add_notes(
        self,
        user_key: UserKey,
        symbol: str,
        modified: datetime,
        noted_price: Decimal,
        future_price: Optional[Decimal],
        body: str,
    ):
        assert self.db
        await self.db.execute(
            """
            INSERT INTO notes (user_id, symbol, modified, noted_price, future_price, body)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, symbol) DO UPDATE
            SET modified = excluded.modified, noted_price = excluded.noted_price, future_price = excluded.future_price, body = excluded.body
            """,
            [
                user_key.uid,
                symbol,
                modified,
                str(noted_price),
                str(future_price) if future_price else None,
                body,
            ],
        )

        updated_user_key = await self.get_user_key_by_user_id(user_key.uid)

        await self.db.commit()

        return updated_user_key

    async def _get_user_symbol_keys(self, user_id: int) -> Dict[str, datetime]:
        assert self.db
        keys = {}
        dbc = await self.db.execute(
            """
            SELECT symbol, MAX(notes.modified) AS key
            FROM notes
            WHERE user_id = ?
            GROUP BY symbol
            """,
            [user_id],
        )
        for row in await dbc.fetchall():
            keys[row[0]] = self._parse_datetime(row[1])
        return keys

    async def _user_modified(self, user_key: UserKey):
        assert self.db
        await self.db.execute(
            "UPDATE users SET modified = ? WHERE id = ?",
            [
                datetime.utcnow(),
                user_key.uid,
            ],
        )

    def _parse_datetime(self, value: str) -> datetime:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")


db: Optional[SymbolStorage] = None


async def get_db():
    global db
    if db:
        return db
    db = SymbolStorage()
    log.info(f"db:created")
    return db
