from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import aiosqlite, os, logging, jsonpickle

log = logging.getLogger("storage")

UserId = int


@dataclass
class UserKey:
    uid: UserId
    generation: str
    symbols: Dict[str, str] = field(default_factory=dict)

    def __str__(self):
        return f"UserKey<{self.uid}, {self.generation}, {len(self.symbols.keys())}>"

    def __repr__(self):
        return str(self)


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


@dataclass
class SymbolStorage:
    db: Optional[aiosqlite.Connection] = None

    async def open(self, path: Optional[str] = None, name: str = "feevee.db"):
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
            "CREATE INDEX IF NOT EXISTS user_lot_ledger_idx ON user_lot_ledger (user_id)"
        )

        await self.db.execute(
            "CREATE TABLE IF NOT EXISTS symbols (symbol TEXT NOT NULL, modified DATETIME NOT NULL, data TEXT NOT NULL)"
        )
        await self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS symbols_symbol_idx ON symbols (symbol)"
        )

        await self.db.execute(
            "CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id), symbol TEXT NOT NULL, created DATETIME NOT NULL, noted_price DECIMAL(10, 5) NOT NULL, future_price DECIMAL(10, 5), body TEXT NOT NULL)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS notes_symbol_ts_idx ON notes (user_id, symbol, created)"
        )
        await self.db.commit()

    async def close(self):
        assert self.db
        await self.db.close()

    async def get_user_key_by_user_id(self, user_id: int) -> UserKey:
        assert self.db
        symbol_keys = await self._get_user_symbol_keys(user_id)
        dbc = await self.db.execute(
            "SELECT id, modified, MAX(notes.created) AS notes_modified FROM users LEFT JOIN notes ON (users.id = notes.user_id) WHERE users.id = ? GROUP BY users.id, users.modified",
            [user_id],
        )
        found = [
            UserKey(row[0], str(row[1]), symbol_keys) for row in await dbc.fetchall()
        ]
        assert found
        return found[0]

    async def get_all_user_ids(self) -> List[UserId]:
        assert self.db
        dbc = await self.db.execute("SELECT id FROM users")
        return [row[0] for row in await dbc.fetchall()]

    async def has_symbol(self, symbol: str):
        assert self.db
        return self.get_symbol(symbol) is not None

    async def get_symbol(self, symbol: str):
        assert self.db
        dbc = await self.db.execute(
            "SELECT modified, data FROM symbols WHERE symbol = ?", [symbol]
        )
        for row in await dbc.fetchall():
            return SymbolRow(symbol, row[0], row[1])
        return None

    async def get_all_symbols(self, user_key: UserKey) -> Dict[str, SymbolRow]:
        assert self.db
        dbc = await self.db.execute(
            "SELECT symbol, modified, data FROM symbols WHERE symbol IN (SELECT symbol FROM user_symbol WHERE user_id = ?) ORDER BY symbol",
            [user_key.uid],
        )
        rows = await dbc.fetchall()
        return {row[0]: SymbolRow(row[0], row[1], row[2]) for row in rows}

    async def add_symbols(self, user_key: UserKey, symbols: List[str]):
        assert self.db
        for symbol in symbols:
            await self.db.execute(
                "INSERT INTO user_symbol (user_id, symbol) VALUES (?, ?) ON CONFLICT DO NOTHING",
                [user_key.uid, symbol],
            )

        await self._user_modified(user_key)

        await self.db.commit()

    async def set_symbol(self, symbol: str, data: Any):
        assert self.db
        serialized = jsonpickle.encode(data)
        log.info(f"{serialized}")
        await self.db.execute(
            """
        INSERT INTO symbols (symbol, modified, data) VALUES (?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET modified = excluded.modified, data = excluded.data
  """,
            [symbol, datetime.now(), serialized],
        )
        await self.db.commit()

    async def get_notes(self, user_key: UserKey, symbol: str):
        assert self.db
        notes = []
        dbc = await self.db.execute(
            "SELECT created, noted_price, future_price, body FROM notes WHERE user_id = ? AND symbol = ? ORDER BY created DESC LIMIT 1",
            [user_key.uid, symbol],
        )
        for row in await dbc.fetchall():
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

    async def _get_user_symbol_keys(self, user_id: int) -> Dict[str, str]:
        assert self.db
        keys = {}
        dbc = await self.db.execute(
            "SELECT symbol, MAX(notes.created) AS key FROM notes WHERE user_id = ? GROUP BY symbol",
            [user_id],
        )
        for row in await dbc.fetchall():
            keys[row[0]] = row[1]
        return keys

    async def get_all_notes(self, user_key: UserKey) -> Dict[str, List[NoteRow]]:
        assert self.db
        notes = {}
        dbc = await self.db.execute(
            "SELECT symbol, created, noted_price, future_price, body FROM notes WHERE user_id = ? ORDER BY created DESC",
            [user_key.uid],
        )
        for row in await dbc.fetchall():
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

    async def add_notes(
        self,
        user_key: UserKey,
        symbol: str,
        created: datetime,
        noted_price: Decimal,
        future_price: Optional[Decimal],
        body: str,
    ):
        assert self.db
        await self.db.execute(
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

        updated_user_key = await self.get_user_key_by_user_id(user_key.uid)

        await self.db.commit()

        return updated_user_key

    async def _user_modified(self, user_key: UserKey):
        assert self.db
        await self.db.execute(
            "UPDATE users SET modified = ? WHERE id = ?",
            [
                datetime.utcnow(),
                user_key.uid,
            ],
        )


db: Optional[SymbolStorage] = None


async def get_db():
    global db
    if db:
        return db
    db = SymbolStorage()
    log.info(f"db:created")
    return db
