from typing import List, Dict, Sequence, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from dateutil.relativedelta import relativedelta
from decimal import Decimal
from pandas.core.frame import DataFrame
from charts import read_prices_csv_sync
from utils import is_market_open
import logging, asyncio, concurrent.futures, re, os
import archive, charts


log = logging.getLogger("feevee")
data_frames_cache_: Dict[str, DataFrame] = {}


@dataclass
class SymbolPriceFile:
    symbol: str
    path: str
    daily: bool
    modified: datetime
    changed: bool

    @property
    def df(self) -> Optional[DataFrame]:
        global data_frames_cache_
        return data_frames_cache_.get(self.path)


def price_range(df) -> Tuple[Decimal, Decimal]:
    lows = df[charts.DailyLowColumn]
    highs = df[charts.DailyHighColumn]
    return (lows.min(), highs.max())


@dataclass
class BasicPrice:
    time: datetime
    price: Decimal


@dataclass
class SymbolPrices:
    symbol: str
    daily_previous: Optional[BasicPrice] = None
    daily_end: Optional[BasicPrice] = None
    candle: Optional[BasicPrice] = None
    one_year_range: Optional[Tuple[Decimal, Decimal]] = None
    files: Dict[str, SymbolPriceFile] = field(default_factory=dict)

    @property
    def price(self) -> Optional[BasicPrice]:
        if is_market_open():
            if self.daily_end and self.candle:
                if self.candle.time > self.daily_end.time:
                    return self.candle
        return self.daily_end

    @property
    def key(self) -> Optional[str]:
        price = self.price
        return str(price.time) if price else None

    @property
    def previous_close(self) -> Optional[BasicPrice]:
        if is_market_open():
            if self.daily_end and self.candle:
                return self.daily_end
        return self.daily_previous

    def price_change(self) -> Optional[Decimal]:
        price = self.price
        previous_close = self.previous_close

        if price and previous_close:
            return price.price - previous_close.price

        return None

    def price_change_percentage(self) -> Optional[Decimal]:
        change = self.price_change()
        if change and self.previous_close:
            return (change / self.previous_close.price) * 100
        return None

    def update(self, file: SymbolPriceFile):
        self.files[file.path] = file

        if len(file.df.index) <= 1:  # type: ignore
            return

        def get_nth(n: int) -> BasicPrice:
            return BasicPrice(file.df.index[n], file.df[charts.DailyCloseColumn][n])  # type: ignore

        today = datetime.now()

        file_end = get_nth(-1)
        if file.daily:
            file_previous = get_nth(-2)

            self.daily_previous = file_previous
            self.daily_end = file_end

            one_year = file.df[today - relativedelta(months=12) : today]  # type: ignore
            self.one_year_range = price_range(one_year)

            log.info(f"{self.symbol:6} prices:daily = {self.daily_end}")
        else:
            self.candle = file_end
            log.info(f"{self.symbol:6} prices:intra = {self.candle}")

    def daily_prices(self) -> charts.Prices:
        for file in [f for f in self.files.values() if f.daily]:
            return charts.Prices(self.symbol, file.df)
        return charts.Prices(self.symbol, charts.make_empty_prices_df())

    def candle_prices(self) -> charts.Prices:
        for file in [f for f in self.files.values() if not f.daily]:
            return charts.Prices(self.symbol, file.df).market_hours_only()
        return charts.Prices(self.symbol, charts.make_empty_prices_df())


def _reload(cache: Dict[str, SymbolPrices], file: SymbolPriceFile):
    global data_frames_cache_
    try:
        started = datetime.utcnow()
        log.debug(f"{file.symbol:6} prices:loading {file.path}")
        df = read_prices_csv_sync(file.path)
        elapsed = datetime.utcnow() - started

        data_frames_cache_[file.path] = df
        symbol_prices = cache.setdefault(file.symbol, SymbolPrices(file.symbol))
        symbol_prices.update(file)

        log.debug(f"{file.symbol:6} prices:loaded elapsed={elapsed}")
    except:
        log.exception(f"{file.symbol:6} prices:error {file.path}")


def _get_symbol_price_files(
    entries: Sequence[archive.DirectoryEntry],
) -> List[SymbolPriceFile]:
    pattern = re.compile("([^/]+)-(daily|iday).csv")
    symbol_files: List[SymbolPriceFile] = []
    for entry in entries:
        m = pattern.search(entry.path)
        if m:
            symbol = m.group(1)
            daily = m.group(2) == "daily"
            symbol_files.append(
                SymbolPriceFile(
                    symbol, entry.path, daily, entry.modified, entry.changed
                )
            )
    if "FEEVEE_SYMBOLS" in os.environ:
        filtering = os.environ["FEEVEE_SYMBOLS"].split(" ")
        return [file for file in symbol_files if file.symbol in filtering]
    return symbol_files


async def _monitor(
    cache: Dict[str, SymbolPrices],
    pool: concurrent.futures.ThreadPoolExecutor,
    path: str,
):
    loop = asyncio.get_running_loop()

    while True:
        try:
            await asyncio.sleep(1)
            directory = await archive.get_directory(path)
            for symbol_file in _get_symbol_price_files(list(directory.values())):
                if symbol_file.changed:
                    loop.run_in_executor(pool, _reload, cache, symbol_file)
        except asyncio.CancelledError:
            log.info(f"prices:stopping")
            pool.shutdown()
            break
        except:
            log.exception(f"prices:error")
            await asyncio.sleep(5)


cache_: Dict[str, SymbolPrices] = {}
task_: Optional[asyncio.Task] = None


def get_symbol_prices_cache_key(symbol: str) -> str:
    global cache_
    if symbol in cache_:
        if key := cache_[symbol].key:
            return key
    return "prices:empty"


async def get_prices(symbol: str) -> SymbolPrices:
    global cache_
    if symbol in cache_:
        return cache_[symbol]
    return SymbolPrices(symbol)


async def initialize(path: str):
    global task_
    global cache_
    if task_:
        return
    assert task_ is None
    log.info(f"prices:initialize")
    pool = concurrent.futures.ThreadPoolExecutor()
    task_ = asyncio.create_task(_monitor(cache_, pool, path))
