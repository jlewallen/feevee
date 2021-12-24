from decimal import Decimal, ROUND_HALF_UP
from typing import (
    List,
    Dict,
    Optional,
    Sequence,
    Any,
    Callable,
    Any,
    Iterable,
)
from dataclasses import dataclass, field
from aiocache import cached, Cache
from aiocache.serializers import PickleSerializer
import pandas
from storage import Criteria, UserSymbol, get_db, UserKey, SymbolRow, NoteRow, UserId
from asyncio_throttle import Throttler  # type: ignore
from datetime import datetime, timedelta
from pytz import timezone
from time import mktime
from aiocron import crontab
import logging, os, json, re, threading, itertools
import hashlib, asyncio, aiofiles
import finnhub
import charts, archive, prices as pricing, lots as stocklots


log = logging.getLogger("feevee")
MoneyCache = os.environ["MONEY_CACHE"]
RedisAddress = (
    os.environ["FEEVEE_REDIS_HOST"] if "FEEVEE_REDIS_HOST" in os.environ else None
)
Caching = (
    dict(cache=Cache.REDIS, serializer=PickleSerializer(), endpoint=RedisAddress)
    if RedisAddress
    else {}
)
FinnHubKey = os.environ["FINN_HUB_KEY"] if "FINN_HUB_KEY" in os.environ else None
PriorityMiddle = 0.5
MetaPaths: Dict[str, str] = {
    "sa": os.path.join(MoneyCache, "seeking-alpha/seeking-alpha.json"),
}
CandlesPerMinute = 20


def get_cache():
    if RedisAddress:
        return Cache(Cache.REDIS, serializer=PickleSerializer(), endpoint=RedisAddress)
    return Cache(Cache.MEMORY)


def is_market_open(t: Optional[datetime] = None, blur: int = 0) -> bool:
    tz = timezone("EST")
    now = datetime.now(tz) if t is None else tz.normalize(t.astimezone(tz))

    if now.weekday() == 5 or now.weekday() == 6:
        return False

    opening_bell = now.replace(hour=9, minute=30, second=0, microsecond=0)
    closing_bell = now.replace(hour=16, minute=0, second=0, microsecond=0)

    if blur:
        opening_bell -= timedelta(minutes=blur)
        closing_bell += timedelta(minutes=blur)

    log.debug(f"market: now={now} opening={opening_bell} closing={closing_bell}")

    return now >= opening_bell and now < closing_bell


def is_after_todays_market_bell(t: Optional[datetime] = None):
    tz = timezone("EST")
    now = datetime.now(tz) if t is None else tz.normalize(t.astimezone(tz))
    if now.weekday() == 5 or now.weekday() == 6:
        return False
    closing_bell = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return now > closing_bell


@dataclass
class TagPriority:
    tag: set
    priority: float
    interval: float


@dataclass
class Portfolio:
    user: UserKey
    lots: stocklots.Lots
    meta: Dict[str, Dict[str, Any]]


@dataclass
class Notes:
    rows: List[NoteRow] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    prices: List[Decimal] = field(default_factory=list)

    def time(self) -> Optional[float]:
        if self.rows:
            return self.rows[0].ts.timestamp()
        return None


@dataclass
class Stock:
    symbol: str
    info: Optional[SymbolRow]
    version: str = ""
    prices: Optional[pricing.SymbolPrices] = None
    lots: stocklots.Lots = field(default_factory=stocklots.Lots)
    notes: Notes = field(default_factory=Notes)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_slow(self) -> bool:
        return self.tagged("slow")

    @property
    def has_tags(self) -> bool:
        return len(self.notes.tags) > 0

    def key(self) -> List[str]:
        return [self.symbol, self.version]

    def tagged(self, tag: str) -> bool:
        return tag in self.notes.tags


@cached(ttl=60)
async def load_meta() -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = dict()
    for key, path in MetaPaths.items():
        async with aiofiles.open(path, mode="r") as file:
            log.info(f"profile:meta {key} = {path}")
            json_data = await file.read()
            meta[key] = json.loads(json_data)
    return meta


@dataclass
class StockInfo:
    user: UserKey
    criteria: Criteria = field(default_factory=Criteria)
    all_symbols: Optional[Dict[str, UserSymbol]] = None

    async def load(self):
        await self.load_all_symbols()

    async def load_all_symbols(self) -> Dict[str, UserSymbol]:
        if self.all_symbols is None:
            db = await get_db()
            self.all_symbols = await db.get_all_symbols(self.user, self.criteria)
        return self.all_symbols

    async def load_symbol_info(self, symbol: str) -> UserSymbol:
        all_symbols = await self.load_all_symbols()
        assert symbol in all_symbols
        return all_symbols[symbol]

    async def load_symbol_notes(self, symbol: str) -> Notes:
        all_symbols = await self.load_all_symbols()
        if symbol in all_symbols:
            user_symbol = all_symbols[symbol]
            if user_symbol.note:
                return await self._parse_symbol_notes(user_symbol.note)
        return Notes()

    async def _parse_symbol_notes(self, row: NoteRow) -> Notes:
        tags_pattern = re.compile("#\s*(\S+)")
        tags_match = tags_pattern.findall(row.body)
        prices_pattern = re.compile("@\s*(\S+)")
        prices_match = prices_pattern.findall(row.body)
        prices = [Decimal(m) for m in prices_match]
        return Notes([row], [row.body], tags_match, prices)


@dataclass
class SymbolMessage:
    user: UserKey
    symbol: str
    render: bool = False


@dataclass
class CheckSymbolMessage(SymbolMessage):
    pass


@dataclass
class RefreshDailyMessage(SymbolMessage):
    maximum_age: float = 86400


@dataclass
class RefreshCandlesMessage(SymbolMessage):
    maximum_age: float = 30


@dataclass
class RefreshIndicatorsMessage(SymbolMessage):
    pass


@dataclass
class RefreshChartsMessage(SymbolMessage):
    pass


class MessagePublisher:
    async def push(self, m: SymbolMessage):
        raise NotImplementedError


class MessageHandler:
    async def service(
        self,
        messages: MessagePublisher,
        portfolio: Portfolio,
        stocks: List[Stock],
    ) -> None:
        raise NotImplementedError

    async def handle(
        self,
        messages: MessagePublisher,
        m: SymbolMessage,
    ) -> None:
        raise NotImplementedError


@dataclass
class MessageWorker:
    handler: MessageHandler
    name: str = "message-worker"
    queue: Optional[asyncio.Queue] = None
    stop_event: Optional[threading.Event] = None
    messages: Optional[MessagePublisher] = None
    task: Optional[asyncio.Task] = None
    concurrency: int = 1

    async def _push(self, m: SymbolMessage):
        assert self.queue
        await self.queue.put(m)

    def start(self, messages: MessagePublisher):
        if self.stop_event:
            return

        assert messages
        self.messages = messages
        self.stop_event = threading.Event()
        self.queue = asyncio.Queue()
        for i in range(self.concurrency):
            self.task = asyncio.create_task(self._worker())

    def stop(self):
        if self.stop_event:
            self.stop_event.set()

    def _run(self):
        log.info(f"{self.name} started")
        loop = asyncio.new_event_loop()
        try:
            self.queue = asyncio.Queue(loop=loop)
            loop.run_until_complete(self._worker())
        finally:
            loop.stop()
        log.info(f"{self.name} exiting")

    async def _worker(self):
        assert self.messages
        assert self.stop_event
        assert self.queue

        while not self.stop_event.is_set():
            try:
                log.info(f"{self.name}:waiting")
                while not self.stop_event.is_set():
                    try:
                        message: SymbolMessage = self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(1)
                        if self.stop_event.is_set():
                            break
                        else:
                            continue

                    log.info(f"{message.symbol:6} {self.name}:have {message}")
                    started = datetime.utcnow()
                    await self.handler.handle(self.messages, message)
                    self.queue.task_done()
                    finished = datetime.utcnow()
                    elapsed = finished - started
                    log.info(
                        f"{message.symbol:6} {self.name}:done {message} elapsed={elapsed}"
                    )
            except asyncio.exceptions.CancelledError:
                log.info(f"{self.name}:cancel")
                return
            except:
                log.exception(f"{self.name}:error", exc_info=True)
                await asyncio.sleep(10)


@dataclass
class Touched:
    touched: Dict[str, datetime] = field(default_factory=dict)

    def can_touch(self, key: str, threshold: timedelta) -> bool:
        if key in self.touched:
            elapsed = datetime.utcnow() - self.touched[key]
            if elapsed < threshold:
                return False
        return True

    def touch(self, key: str):
        self.touched[key] = datetime.utcnow()


@dataclass
class StockSorter:
    tag_priorities: Dict[str, float] = field(default_factory=dict)

    def sort(self, stocks: List[Stock]) -> List[Stock]:
        def get_sort_key(stock: Stock):
            p = PriorityMiddle
            for tag, priority in self.tag_priorities.items():
                if stock.tagged(tag):
                    p = priority
                    break

            time = (
                stock.prices.candle.time.timestamp()
                if stock.prices and stock.prices.candle
                else 0
            )
            return (time, p)

        return list(sorted(stocks, key=get_sort_key))


@dataclass
class Financials:
    throttler: Throttler = Throttler(rate_limit=30, period=60)

    async def query(self, symbol: str, csv_path: str, intraday: bool):
        if FinnHubKey is None:
            log.info(f"{symbol:6} financials:no-key")
            return

        async with self.throttler:
            fc = finnhub.Client(api_key=FinnHubKey)
            window = timedelta(days=3) if intraday else timedelta(days=363)
            now = datetime.today().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            start = now - window
            log.info(f"{symbol:6} financials:querying {start} -> {now}")
            res = fc.stock_candles(
                symbol,
                "5" if intraday else "D",
                int(mktime(start.timetuple())),
                int(mktime(now.timetuple())),
            )
            if res["s"] != "ok":
                log.info(f"{symbol:6} financials:no-data")
                return

        try:
            await self._merge_ohlc(symbol, csv_path, res)
        except:
            log.exception(f"{symbol:6} financials:error")

    async def _merge_ohlc(self, symbol: str, csv_path: str, res: Dict[str, Any]):
        df = Candles(symbol, []).to_df()
        if os.path.isfile(csv_path):
            df = await charts.read_prices_csv(csv_path)
            log.info(f"{symbol:6} financials:read {csv_path} samples={len(df.index)}")

        size_before = len(df.index)
        candles = parse_candles(symbol, **res)
        incoming = candles.to_df()
        for index, row in incoming.iterrows():
            if index in df.index:
                continue
            log.debug(f"{symbol:6} financials:new")
            df = df.append(row)

        size_after = len(df.index)
        if size_after != size_before:
            await charts.write_prices_csv(csv_path, df)
            log.info(
                f"{symbol:6} financials:wrote {csv_path} samples={size_after - size_before}"
            )


@dataclass
class ManageDailies(MessageHandler):
    financials: Financials
    tag_priorities: Dict[str, float] = field(default_factory=dict)
    touched: Touched = field(default_factory=Touched)

    async def service(
        self,
        messages: MessagePublisher,
        portfolio: Portfolio,
        stocks: List[Stock],
    ) -> None:
        if is_market_open():
            return

        missing = [s for s in stocks if s.prices and s.prices.daily_end is None]
        if missing:
            log.info(f"queue:dailies {[s.symbol for s in missing]}")
        for stock in missing:
            await messages.push(
                RefreshDailyMessage(portfolio.user, stock.symbol, maximum_age=0)
            )

        # This is an easy way of seeing if it's one hour after today's bell,
        # which is when we start to care about this.
        if is_after_todays_market_bell(datetime.now() - timedelta(hours=1)):
            sorter = StockSorter(tag_priorities=self.tag_priorities)
            for s in sorter.sort(stocks):
                log.debug(f"{s.symbol:6} daily:test {s.prices}")
                if s.prices is None:
                    continue
                if s.prices.candle and (
                    s.prices.daily_end is None
                    or s.prices.daily_end.time.date() < s.prices.candle.time.date()
                ):
                    if self.touched.can_touch(s.symbol, timedelta(minutes=60)):
                        log.info(f"{s.symbol:6} daily:queue {s.prices}")
                        await messages.push(
                            RefreshDailyMessage(portfolio.user, s.symbol, maximum_age=0)
                        )
                        self.touched.touch(s.symbol)

    async def handle(self, messages: MessagePublisher, m: SymbolMessage):
        assert isinstance(m, RefreshDailyMessage)

        daily_path = os.path.join(
            MoneyCache, charts.get_relative_daily_prices_path(m.symbol)
        )

        daily_prices_time = archive.get_time(daily_path)
        if m.maximum_age > 0 and daily_prices_time:
            local_now = datetime.now().timestamp()
            if local_now - daily_prices_time.timestamp() < m.maximum_age:
                log.info(f"{m.symbol:6} daily:fresh")
                return

        await self.financials.query(m.symbol, daily_path, False)

        await messages.push(RefreshChartsMessage(m.user, m.symbol))


@dataclass
class ManageCandles(MessageHandler):
    financials: Financials
    tag_priorities: Dict[str, float] = field(default_factory=dict)
    touched: Touched = field(default_factory=Touched)

    async def service(
        self,
        messages: MessagePublisher,
        portfolio: Portfolio,
        stocks: List[Stock],
    ) -> None:
        if not is_market_open(blur=30):
            return

        def can_refresh(stock: Stock) -> bool:
            if stock.is_slow or not stock.has_tags:
                return False
            return self.touched.can_touch(stock.symbol, timedelta(minutes=10))

        sorter = StockSorter(tag_priorities=self.tag_priorities)
        available = [stock for stock in stocks if can_refresh(stock)]
        ordered = sorter.sort(available)
        refreshing = ordered[:CandlesPerMinute]
        symbols = [stock.symbol for stock in refreshing]
        if symbols:
            log.info(f"queue:candles {symbols}")
        for symbol in symbols:
            self.touched.touch(symbol)
            await messages.push(RefreshCandlesMessage(portfolio.user, symbol))

    async def handle(self, messages: MessagePublisher, m: SymbolMessage):
        intraday_path = os.path.join(
            MoneyCache, charts.get_relative_intraday_prices_path(m.symbol)
        )

        await self.financials.query(m.symbol, intraday_path, True)

        if m.render:
            await messages.push(RefreshChartsMessage(m.user, m.symbol))


@dataclass
class SymbolChecker(MessageHandler):
    async def service(
        self,
        messages: MessagePublisher,
        portfolio: Portfolio,
        stocks: List[Stock],
    ) -> None:
        pass

    async def handle(self, messages: MessagePublisher, m: SymbolMessage):
        assert isinstance(m, CheckSymbolMessage)
        if FinnHubKey is None:
            log.info(f"{m.symbol:6} check-sym:no-key")
            return

        fc = finnhub.Client(api_key=FinnHubKey)
        res = fc.symbol_lookup(m.symbol)
        matches = res.get("result")
        if matches is None or len(matches) == 0:
            log.info(f"{m.symbol:6} check-sym:no-match {res}")
            return

        exact = [match for match in matches if match["symbol"] == m.symbol]
        if len(exact) > 1:
            log.info(f"{m.symbol:6} check-sym:no-match {res}")
            return

        db = await get_db()
        for match in exact:
            log.info(f"{m.symbol:6} {exact}")
            await db.set_symbol(m.symbol, True, dict(name=match["description"]))

        await messages.push(RefreshDailyMessage(m.user, m.symbol, maximum_age=0))

    async def create_throttle(self):
        return Throttler(rate_limit=5, period=60)


@dataclass
class ManageIndicators(MessageHandler):
    async def service(
        self,
        messages: MessagePublisher,
        portfolio: Portfolio,
        stocks: List[Stock],
    ) -> None:
        pass

    async def handle(self, messages: MessagePublisher, m: SymbolMessage):
        if FinnHubKey is None:
            log.info(f"{m.symbol:6} indicators:no-key")
            return

    async def create_throttle(self):
        return Throttler(rate_limit=5, period=60)


class RefreshChartsHandler(MessageHandler):
    async def create_throttle(self):
        return Throttler(rate_limit=120, period=60)


@dataclass
class RefreshQueue(MessagePublisher):
    repository: "SymbolRepository"
    candles: MessageWorker
    dailies: MessageWorker
    charts: MessageWorker
    indicators: MessageWorker
    checker: MessageWorker
    tasks: List[asyncio.Task] = field(default_factory=list)
    workers: List[MessageWorker] = field(default_factory=list)

    async def push(self, m: SymbolMessage):
        try:
            if isinstance(m, RefreshChartsMessage):
                await self.charts._push(m)
            if isinstance(m, RefreshCandlesMessage):
                await self.candles._push(m)
            if isinstance(m, RefreshDailyMessage):
                await self.dailies._push(m)
            if isinstance(m, RefreshIndicatorsMessage):
                await self.indicators._push(m)
            if isinstance(m, CheckSymbolMessage):
                await self.checker._push(m)
        except:
            logging.exception(f"{m} error", exc_info=True)

    async def _crond(self, spec: str, handler):
        while True:
            try:
                log.info(f"crond:started")
                while True:
                    await crontab(spec).next()
                    await handler()
            except asyncio.exceptions.CancelledError:
                log.info(f"crond:stopped")
                return
            except:
                log.exception(f"crond:error", exc_info=True)
                await asyncio.sleep(10)

    async def _opening(self):
        log.info(f"bell-opening:ding")

    async def _user_minute(self, user_id: UserId):
        db = await get_db()

        user = await db.get_user_key_by_user_id(user_id)

        stock_info = StockInfo(user)
        portfolio = await self.repository.get_portfolio(user)
        stocks = await self.repository.get_all_stocks(
            user, portfolio, stock_info, Criteria()
        )

        return await asyncio.gather(
            self.candles.handler.service(self, portfolio, stocks),
            self.dailies.handler.service(self, portfolio, stocks),
            self.indicators.handler.service(self, portfolio, stocks),
            self.checker.handler.service(self, portfolio, stocks),
        )

    async def _minute(self):
        log.info(f"minute")
        user_ids = await self.repository.get_all_users()
        return asyncio.gather(*[self._user_minute(user_id) for user_id in user_ids])

    async def _user_closing(self, user_id: UserId):
        user = await self.repository.get_user(user_id)
        portfolio = await self.repository.get_portfolio(user)
        # TODO Refresh daily

    async def _closing(self):
        log.info(f"bell-closing:ding")
        user_ids = await self.repository.get_all_users()
        return asyncio.gather(*[self._user_closing(user_id) for user_id in user_ids])

    async def _watch(self):
        while True:
            try:
                await asyncio.sleep(1)
            except:
                log.info(f"stopping")
                db = await get_db()
                await db.close()
                self.charts.stop()
                self.candles.stop()
                self.dailies.stop()
                self.indicators.stop()
                self.checker.stop()
                break

    def _get_watch_dir(self) -> str:
        return os.path.join(MoneyCache, charts.PriceDirectory)

    async def _delayed_start(self):
        await asyncio.sleep(60)

        log.info(f"delayed-start")
        self.tasks.append(asyncio.create_task(self._crond("* * * * *", self._minute)))

    async def start(self):
        if self.tasks:
            return

        log.info(f"queues:starting")

        asyncio.get_event_loop().set_debug(True)

        await pricing.initialize(self._get_watch_dir())

        db = await get_db()

        file_name = (
            os.environ["FEEVEE_DB"] if "FEEVEE_DB" in os.environ else "feevee.db"
        )

        await db.open(file_name)

        self.charts.start(self)
        self.dailies.start(self)
        self.candles.start(self)
        self.indicators.start(self)
        self.checker.start(self)

        # NOTE: These are in PST!
        open_cron = "0 6 * * mon,tue,wed,thu,fri"
        close_cron = "45 13 * * mon,tue,wed,thu,fri"

        self.tasks.append(asyncio.create_task(self._watch()))
        self.tasks.append(asyncio.create_task(self._crond(open_cron, self._opening)))
        self.tasks.append(asyncio.create_task(self._crond(close_cron, self._closing)))
        self.tasks.append(asyncio.create_task(self._delayed_start()))


@dataclass
class SymbolRepository:
    async def get_all_users(self) -> List[UserId]:
        db = await get_db()
        return await db.get_all_user_ids()

    async def get_user(self, uid: UserId) -> UserKey:
        db = await get_db()
        return await db.get_user_key_by_user_id(uid)

    async def get_portfolio(self, user: UserKey) -> Portfolio:
        meta = await load_meta()
        db = await get_db()
        lots_raw = await db.get_lots(user)
        lots = stocklots.parse(lots_raw)
        return Portfolio(user, lots, meta)

    async def get_all_stocks(
        self,
        user: UserKey,
        portfolio: Portfolio,
        stock_info: StockInfo,
        criteria: Criteria,
    ) -> List[Stock]:
        db = await get_db()
        symbols = await db.get_all_symbols(user, criteria)
        return await chunked(
            "batch-db",
            symbols,
            lambda symbol: self.get_stock(user, symbol, portfolio, stock_info),
        )

    async def get_stock(
        self,
        user: UserKey,
        symbol: str,
        portfolio: Portfolio,
        stock_info: StockInfo,
    ) -> Stock:
        info = await stock_info.load_symbol_info(symbol)
        notes = await stock_info.load_symbol_notes(symbol)
        symbol_prices = await pricing.get_prices(symbol)
        symbol_lots = portfolio.lots.for_symbol(symbol)
        notes_time = notes.time()

        assert info

        hashing = finish_key(
            [
                symbol,
                str(notes_time),
                pricing.get_symbol_prices_cache_key(symbol),
                symbol_lots.lots_key,
            ]
        )

        def get_meta(sub_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if symbol in sub_meta:
                return sub_meta[symbol]
            return None

        meta = {key: get_meta(sm) for key, sm in portfolio.meta.items()}

        version = hashlib.sha1(bytes(hashing, encoding="utf8")).hexdigest()
        log.info(f"{symbol:6} loading stock {version} notes-time={notes_time}")
        return Stock(
            symbol,
            info.symbol,
            version,
            symbol_prices,
            symbol_lots,
            notes,
            meta,
        )

    async def add_symbols(self, user: UserKey, symbols: List[str]) -> List[str]:
        db = await get_db()
        return await db.add_symbols(user, symbols)

    async def remove_symbols(self, user: UserKey, symbols: List[str]) -> List[str]:
        db = await get_db()
        return await db.remove_symbols(user, symbols)

    async def update_lots(self, user: UserKey, lots: str) -> UserKey:
        db = await get_db()
        return await db.update_lots(user, lots)

    async def save_notes(self, user: UserKey, symbol: str, noted_price: str, body: str):
        stock_info = StockInfo(user)
        db = await get_db()
        portfolio = await self.get_portfolio(user)
        notes = await db.get_notes(user, symbol)
        if len(notes) == 0 or notes[0].body != body:
            user = await db.add_notes(
                user, symbol, datetime.utcnow(), Decimal(noted_price), None, body
            )

            # HACK This isn't updating the user in Portfolio
            portfolio.user = user
        return await self.get_stock(user, symbol, portfolio, stock_info)


def rounding(v: Decimal) -> Decimal:
    return v.quantize(Decimal("0.01"), ROUND_HALF_UP)


def _include_candles(prices: charts.Prices, candles: charts.Prices) -> charts.Prices:
    if candles.empty:
        log.info(f"{candles.symbol:6} candle:emtpy")
        return prices

    daily_max_time = prices.daily.index.max()
    intra_max_time = candles.daily.index.max()

    candles_after_daily = candles.after(daily_max_time)
    if candles_after_daily.empty:
        log.info(
            f"{candles.symbol:6} candles:ignore daily-max={daily_max_time} intra-max={intra_max_time}"
        )
        return prices

    opening = rounding(candles_after_daily.daily[charts.DailyOpenColumn][0])
    closing = rounding(candles_after_daily.daily[charts.DailyCloseColumn][-1])
    low = rounding(candles_after_daily.daily[charts.DailyLowColumn].min())
    high = rounding(candles_after_daily.daily[charts.DailyHighColumn].max())
    volume = rounding(Decimal(candles_after_daily.daily[candles.volume_column()].sum()))
    last_key = candles_after_daily.daily.index[-1].strftime("%Y-%m-%d %H:%M:%S")

    # We clone here to avoid having this modified dataframe modified while
    # in-memory and cached as the boring old daily data frame.
    copy = prices.clone()
    copy.daily.at[last_key, charts.DailyOpenColumn] = opening
    copy.daily.at[last_key, charts.DailyLowColumn] = low
    copy.daily.at[last_key, charts.DailyHighColumn] = high
    copy.daily.at[last_key, charts.DailyCloseColumn] = closing
    copy.daily.at[last_key, charts.DailyVolumeColumn] = volume
    log.info(
        f"{candles.symbol:6} candles:include daily-max={daily_max_time} candles-max={intra_max_time} candles={len(candles_after_daily.daily.index)} open={opening} low={low} high={high} closing={closing}"
    )
    return copy


def subtract_days(f: datetime, days: int) -> datetime:
    if f.weekday() == 0:
        return f - timedelta(days=days + 2)
    return f - timedelta(days=days)


async def load_days_of_symbol_candles(symbol: str, days: int) -> charts.Prices:
    symbol_prices = await pricing.get_prices(symbol)
    prices = symbol_prices.candle_prices()
    if prices.empty:
        log.info(f"{symbol:6} candles:load-empty")
        return charts.Prices(prices.symbol, charts.make_empty_prices_df())
    date_range = prices.date_range()
    end = date_range[1]
    start = subtract_days(end, days)
    log.info(
        f"{symbol:6} candles:df days={days} range={date_range} start={start} end={end}"
    )
    return charts.Prices(prices.symbol, prices.daily[start:])  # type:ignore


async def load_daily_symbol_prices(symbol: str) -> charts.Prices:
    started = datetime.utcnow()
    symbol_prices = await pricing.get_prices(symbol)
    prices = symbol_prices.daily_prices()
    elapsed = datetime.utcnow() - started
    log.debug(f"{symbol:6} daily:df elapsed={elapsed}")
    return prices


async def load_symbol_prices(symbol: str):
    started = datetime.utcnow()
    candles = await load_days_of_symbol_candles(symbol, 1)
    daily = await load_daily_symbol_prices(symbol)
    with_candles = _include_candles(daily, candles)
    elapsed = datetime.utcnow() - started
    log.debug(f"{symbol:6} prices:df elapsed={elapsed}")
    return with_candles


@dataclass
class Candle:
    time: datetime
    opening: Decimal
    low: Decimal
    high: Decimal
    closing: Decimal
    volume: Decimal


@dataclass
class Candles:
    symbol: str
    candles: List[Candle]

    def last_time(self) -> Optional[datetime]:
        return self.candles[-1].time if self.candles else None

    def to_df(self):
        index = [c.time for c in self.candles]
        data = [
            [
                rounding(c.opening),
                rounding(c.low),
                rounding(c.high),
                rounding(c.closing),
                rounding(c.volume),
            ]
            for c in self.candles
        ]
        columns = [
            charts.DailyOpenColumn,
            charts.DailyLowColumn,
            charts.DailyHighColumn,
            charts.DailyCloseColumn,
            charts.DailyVolumeColumn,
        ]
        df = pandas.DataFrame(data, columns=columns, index=index)
        df.index.name = charts.DailyDateColumn
        return df


def parse_candles(
    symbol: str,
    t: Optional[List[int]] = None,
    o: Optional[List[float]] = None,
    c: Optional[List[float]] = None,
    h: Optional[List[float]] = None,
    l: Optional[List[float]] = None,
    v: Optional[List[float]] = None,
    **kwargs,
) -> Candles:
    assert o
    assert c
    assert h
    assert l
    assert t
    assert v

    length = len(t)
    assert len(o) == length
    assert len(c) == length
    assert len(h) == length
    assert len(l) == length
    assert len(v) == length

    datetimes = [datetime.fromtimestamp(value) for value in t]

    candles = [
        Candle(
            datetimes[i],
            Decimal(o[i]),
            Decimal(l[i]),
            Decimal(h[i]),
            Decimal(c[i]),
            Decimal(v[i]),
        )
        for i in range(length)
    ]

    return Candles(symbol, sorted(candles, key=lambda c: c.time))


def chunked_iterable(iterable: Iterable, size: int) -> Iterable:
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


async def chunked(name: str, items: List[Any], fn: Callable):
    async def assemble_batch(batch):
        started = datetime.utcnow()
        vms = await asyncio.gather(*[fn(item) for item in batch])
        elapsed = datetime.utcnow() - started
        log.info(f"{name} elapsed={elapsed} size={len(batch)}")
        return vms

    batched = chunked_iterable(items, size=10)
    return flatten([await assemble_batch(batch) for batch in batched])


def get_backup_path(path: str) -> str:
    full, extension = os.path.splitext(path)
    dated = datetime.now().strftime("%Y-%m-%d")
    return f"{full}-{dated}{extension}"


async def copy_file(from_path: str, to_path: str):
    # This is shitty but works for the files we're dealing with.
    log.info(f"copying {from_path} -> {to_path}")
    async with aiofiles.open(from_path, mode="r") as reading:
        async with aiofiles.open(to_path, mode="w") as writing:
            await writing.write(await reading.read())


async def backup_daily_file(path: str):
    if os.path.isfile(path):
        backup_path = get_backup_path(path)
        if not os.path.isfile(backup_path):
            await copy_file(path, backup_path)
        else:
            log.info(f"already have {backup_path}")


async def write_json_file(data: Any, path: str):
    json_path = os.path.join(os.path.join(MoneyCache, charts.PriceDirectory), path)
    await backup_daily_file(json_path)
    async with aiofiles.open(json_path, mode="w") as file:
        await file.write(json.dumps(data))


async def is_missing(path: str) -> bool:
    try:
        await aiofiles.os.stat(
            os.path.join(os.path.join(MoneyCache, charts.PriceDirectory), path)
        )
        return False
    except FileNotFoundError:
        return True


def finish_key(parts: Sequence[str]) -> str:
    return ",".join(parts)


def flatten(a):
    return [leaf for sl in a for leaf in sl]
