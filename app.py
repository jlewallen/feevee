from typing import List, Dict, Optional, Any
from decimal import Decimal, ROUND_HALF_UP
from quart import Quart, send_from_directory, request, Response
from quart import json as quart_json
from quart_cors import cors
from datetime import datetime
from dateutil.relativedelta import relativedelta
from aiocache import cached
from dataclasses import dataclass, field
from ta.volume import AccDistIndexIndicator
import logging, json, re

from backend import (
    CheckSymbolMessage,
    Financials,
    Portfolio,
    chunked,
    ManageCandles,
    ManageDailies,
    ManageIndicators,
    SymbolChecker,
    MessagePublisher,
    MessageWorker,
    Caching,
    get_cache,
    SymbolMessage,
    RefreshChartsHandler,
    RefreshIndicatorsMessage,
    RefreshDailyMessage,
    RefreshCandlesMessage,
    RefreshChartsMessage,
    SymbolRepository,
    RefreshQueue,
    Stock,
    is_market_open,
    finish_key,
    load_days_of_symbol_candles,
    load_symbol_prices,
    get_user_symbols_key,
)
from loggers import setup_logging_queue
from storage import UserKey, get_db
import charts


@dataclass
class Theme:
    colors: charts.Colors
    noted_bg_color: str
    noted_text_color: str
    buy_bg_color: str
    buy_text_color: str
    basis_bg_color: str
    basis_text_color: str


Light = Theme(
    charts.Light,
    noted_bg_color="#d2b4de",
    noted_text_color="#000000",
    buy_bg_color="#aed6f1",
    buy_text_color="#000000",
    basis_bg_color="#c1e1c1",
    basis_text_color="#000000",
)

Dark = Theme(
    charts.Dark,
    noted_bg_color="#f4d03f",
    noted_text_color="#000000",
    buy_bg_color="#85c1e9",
    buy_text_color="#000000",
    basis_bg_color="#73c6b6",
    basis_text_color="#000000",
)

Paper = Theme(
    charts.Paper,
    noted_bg_color="#f4d03f",
    noted_text_color="#000000",
    buy_bg_color="#85c1e9",
    buy_text_color="#000000",
    basis_bg_color="#73c6b6",
    basis_text_color="#000000",
)


Themes = {"dark": Dark, "light": Light, "paper": Paper}


def get_theme(name: str):
    assert name in Themes
    return Themes[name]


def _assemble_stock_view_model_key(fn, stock: Stock) -> str:
    return finish_key(stock.key())


@cached(key_builder=_assemble_stock_view_model_key, **Caching)
async def assemble_stock_view_model(stock: Stock):
    # Sure, this method is hideous but at least it's all in one place!

    symbol_prices = stock.prices
    assert symbol_prices

    symbol_lots = [lot for lot in stock.lots.lots if lot.symbol == stock.symbol]
    basis_price = stock.lots.get_basis(stock.symbol)

    previous_close = symbol_prices.previous_close
    price_change = symbol_prices.price_change()
    percent_change = symbol_prices.price_change_percentage()
    last_price = symbol_prices.price.price if symbol_prices.price else None
    freshness: float = 0.0

    if symbol_prices.price:
        freshness = -symbol_prices.price.time.timestamp()

    def lot_to_json(lot):
        return dict(date=lot.date, price=float(lot.price), quantity=float(lot.quantity))

    def make_position():
        if not basis_price:
            return None

        assert symbol_prices
        total_value: Optional[Decimal] = None
        if symbol_prices.price:
            total_value = rounding(
                symbol_prices.price.price * sum([l.quantity for l in symbol_lots])
            )

        return dict(
            symbol=stock.symbol,
            basis_price=rounding(basis_price),
            total_value=total_value,
        )

    position = make_position()

    def is_nearby(
        target: Decimal, value: Decimal, epsilon: Decimal = Decimal(0.1)
    ) -> bool:
        s = value * Decimal(1.0) - epsilon
        e = value * Decimal(1.0) + epsilon
        return target >= s and target <= e

    virtual_tags = [] if stock.notes.tags else ["v:untagged"]

    if position:
        virtual_tags.append("v:hold")

    if price_change:
        virtual_tags.append("v:d" if price_change < 0 else "v:u")

    if last_price:
        if basis_price:
            if last_price > basis_price:
                virtual_tags.append("v:basis:above")
                if "exiting" in stock.notes.tags:
                    virtual_tags.append("V:sell")
            else:
                virtual_tags.append("v:basis:below")
                if "entering" in stock.notes.tags:
                    virtual_tags.append("V:buy")

        if year := symbol_prices.one_year_range:
            epsilon = Decimal(0.1)
            near_low = is_nearby(year[0], last_price, epsilon)
            near_high = is_nearby(year[1], last_price, epsilon)
            if near_low and not near_high:
                virtual_tags.append("v:year:low")
            if near_high and not near_low:
                virtual_tags.append("v:year:high")

        for np in stock.notes.prices:
            if is_nearby(np, last_price):
                virtual_tags.append("v:noted")
                break

    log.info(
        f"{stock.symbol:6} vm:done version={stock.version} freshness={freshness} price={last_price}"
    )

    return dict(
        symbol=stock.symbol,
        version=stock.version,
        meta=stock.meta,
        key=finish_key(stock.key()),
        info=json.loads(stock.info.data) if stock.info else None,
        position=position,
        change=price_change,
        price=last_price,
        previous_close=previous_close,
        tags=sorted(stock.notes.tags + virtual_tags),
        lots=[lot_to_json(lot) for lot in symbol_lots],
        percent_change=maybe_round(percent_change),
        negative=percent_change < 0 if percent_change else False,
        noted_prices=stock.notes.prices,
        has_candles=not not symbol_prices.candle,
        freshness=freshness,
        notes=[
            dict(
                symbol=n.symbol,
                ts=n.ts,
                noted_price=n.noted_price,
                future_price=n.future_price,
                body=n.body,
            )
            for n in stock.notes.rows
        ],
    )


async def _render_ohlc(
    stock: Stock,
    prices: charts.Prices,
    w: int,
    h: int,
    style: str,
    trading_hours_only: bool = False,
    show_last_buy: bool = False,
    **kwargs,
):
    symbol = stock.symbol
    theme = get_theme(style)
    basis_price = stock.lots.get_basis(symbol)
    last_buy_price = stock.lots.get_last_buy_price(symbol)

    marks = [
        charts.PriceMark(np, theme.noted_bg_color, theme.noted_text_color, False, True)
        for np in stock.notes.prices
    ]

    if show_last_buy and last_buy_price:
        marks.append(
            charts.PriceMark(
                rounding(last_buy_price),
                theme.buy_bg_color,
                theme.buy_text_color,
                False,
                True,
            )
        )

    if basis_price:
        marks.append(
            charts.PriceMark(
                rounding(basis_price),
                theme.basis_bg_color,
                theme.basis_text_color,
                False,
                True,
            )
        )

    data = await charts.ohlc(
        prices,
        symbol,
        size=(w, h),
        marks=marks,
        colors=theme.colors,
        trading_hours_only=trading_hours_only,
    )
    return Response(data, mimetype="image/png")


months_pattern = re.compile("(\d+)M")
days_pattern = re.compile("(\d+)D")
ma_pattern = re.compile("(\d+)MA")


async def load_months_of_symbol_prices(symbol: str, months: int, options: List[str]):
    original = await load_symbol_prices(symbol)

    prices = original.clone()

    for option in options:
        if option == "MACD":
            exp1 = (
                prices.daily[charts.DailyCloseColumn].ewm(span=12, adjust=False).mean()
            )
            exp2 = (
                prices.daily[charts.DailyCloseColumn].ewm(span=26, adjust=False).mean()
            )
            prices.daily["S:MACD"] = exp1 - exp2
            prices.daily["S:MACD-Signal"] = (
                prices.daily["S:MACD"].ewm(span=9, adjust=False).mean()
            )

        if option == "ADI":
            high = prices.high.astype(float)
            low = prices.low.astype(float)
            close = prices.closing.astype(float)
            volume = prices.volume.astype(float)
            indicator = AccDistIndexIndicator(
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
            prices.daily["S:ADI"] = indicator.acc_dist_index()

        if m := ma_pattern.match(option):
            days = int(m.group(1))
            prices.daily[f"I:{days}MA"] = (
                prices.daily[charts.DailyCloseColumn].rolling(window=f"{days}D").mean()
            )

    today = datetime.utcnow()
    start = today - relativedelta(months=months)
    return prices.history(start, today)


async def get_prices_for_duration(
    symbol: str, duration: str, options: List[str]
) -> charts.Prices:
    if m := months_pattern.match(duration):
        return await load_months_of_symbol_prices(symbol, int(m.group(1)), options)
    if m := days_pattern.match(duration):
        return await load_days_of_symbol_candles(symbol, int(m.group(1)))
    raise Exception(f"unknown duration: {duration}")


async def get_options(symbol: str, duration: str, options: List[str]) -> Dict[str, Any]:
    return dict(trading_hours_only="D" in duration)


def _render_ohlc_key(
    fn, stock: Stock, duration: str, w: int, h: int, style: str, options: List[str]
) -> str:
    return finish_key(
        ["ohlc"] + stock.key() + [str(v) for v in [duration, w, h, style]] + options
    )


@cached(key_builder=_render_ohlc_key, **Caching)
async def render_ohlc(
    stock: Stock, duration: str, w: int, h: int, style: str, options: List[str]
):
    prices = await get_prices_for_duration(stock.symbol, duration, options)
    kwargs = await get_options(stock.symbol, duration, options)
    log.info(
        f"{stock.symbol:6} rendering {stock.key()} {w}x{h} {duration} {style} {options} {kwargs}"
    )
    return await _render_ohlc(stock, prices, w, h, style, **kwargs)


@dataclass
class ChartTemplate:
    w: int
    h: int
    theme: str


@dataclass
class WebChartsCacheWarmer(RefreshChartsHandler):
    templates: List[ChartTemplate] = field(default_factory=list)
    capacity: int = 4

    async def include_template(self, w: int, h: int, theme: str):
        template = ChartTemplate(w, h, theme)
        if template in self.templates:
            return

        if len(self.templates) > self.capacity:
            log.info(f"web-charts:clearing-capacity")
            self.templates = self.templates[len(self.templates) - self.capacity :]

        log.info(f"web-charts:new {template}")
        self.templates.append(template)

    async def handle(self, messages: MessagePublisher, m: SymbolMessage):
        log.info(f"{m.symbol:6} web-charts:begin")
        stock = await repository.get_stock(m.user, m.symbol)
        for template in self.templates:
            for duration in ["3M", "12M"]:
                await render_ohlc(
                    stock, duration, template.w, template.h, template.theme, []
                )


class JSONEncoder(quart_json.JSONEncoder):
    def default(self, object_):
        if isinstance(object_, Decimal):
            return str(object_)
        elif isinstance(object_, datetime):
            return str(object_)
        else:
            return super().default(object_)


log = logging.getLogger("feevee")
app = cors(Quart(__name__))
app.json_encoder = JSONEncoder
repository = SymbolRepository()

DefaultTagPriorities = {
    "hf": 0,
    "v:noted": 0.40,
    "v:hold": 0.45,
    "v:untagged": 0.95,
    "lf": 1,
    "slow": 1,
}

financials = Financials()
candles = MessageWorker(
    ManageCandles(
        financials,
        tag_priorities=DefaultTagPriorities,
    )
)
indicators = MessageWorker(ManageIndicators())
dailies = MessageWorker(ManageDailies(financials, tag_priorities=DefaultTagPriorities))
web_charts = WebChartsCacheWarmer()
symbol_checker = MessageWorker(SymbolChecker())
refreshing = RefreshQueue(
    repository,
    candles,
    dailies,
    MessageWorker(web_charts, concurrency=5),
    indicators,
    symbol_checker,
)
AdministratorUserId = 1


async def get_user() -> UserKey:
    await refreshing.start()

    db = await get_db()
    maybe_user = await db.get_user_key_by_user_id(AdministratorUserId)
    assert maybe_user
    log.debug(f"web:user: {maybe_user}")
    return maybe_user


@cached(key_builder=get_user_symbols_key, **Caching)
async def get_user_symbols(portfolio: Portfolio):
    stocks = await repository.get_all_stocks(portfolio.user, portfolio=portfolio)
    return await chunked(
        "batch-vm", stocks, lambda stock: assemble_stock_view_model(stock)
    )


@app.route("/status")
async def status():
    user = await get_user()
    portfolio = await repository.get_portfolio(user)
    symbols = await get_user_symbols(portfolio)
    return dict(market=dict(open=is_market_open()), symbols=[s for s in symbols if s])


@app.route("/clear")
async def clear():
    user = await get_user()
    assert user.uid == AdministratorUserId

    cache = get_cache()
    log.info(f"clearing {cache}")
    await cache.clear()

    return dict()


@app.route("/render")
async def render():
    user = await get_user()
    assert user.uid == AdministratorUserId

    stocks = await repository.get_all_stocks(user)
    for stock in stocks:
        await refreshing.push(RefreshChartsMessage(user, stock.symbol))

    return dict()


async def _basic_refresh(user: UserKey, symbol: str):
    if request.args.get("candles"):
        await refreshing.push(RefreshCandlesMessage(user, symbol))
    if request.args.get("indicators"):
        await refreshing.push(RefreshIndicatorsMessage(user, symbol))
    if request.args.get("daily"):
        force = request.args.get("force")
        await refreshing.push(
            RefreshDailyMessage(user, symbol, maximum_age=0)
            if force
            else RefreshDailyMessage(user, symbol)
        )


@app.route("/symbols", methods=["POST"])
async def modify_symbols():
    user = await get_user()
    raw = await request.get_data()
    parsed: Dict[str, List[str]] = json.loads(raw)
    symbols = [s.upper() for s in parsed["symbols"]]

    if parsed["adding"]:
        for symbol in await repository.add_symbols(user, symbols):
            await refreshing.push(CheckSymbolMessage(user, symbol))
    else:
        await repository.remove_symbols(user, symbols)

    return dict()


@app.route("/lots", methods=["POST"])
async def modify_lots():
    user = await get_user()
    raw = await request.get_data()
    parsed: Dict[str, str] = json.loads(raw)
    lots_text = parsed["lots"]

    log.info(f"lots: {lots_text}")

    await repository.update_lots(user, lots_text)

    return dict()


@app.route("/symbols/refresh")
async def refresh_symbols():
    user = await get_user()
    stocks = await repository.get_all_stocks(user)
    for stock in stocks:
        await _basic_refresh(user, stock.symbol)

    return dict()


@app.route("/symbols/<symbol>/refresh")
async def refresh_symbol(symbol: str):
    user = await get_user()
    await _basic_refresh(user, symbol)

    return dict()


def parse_options() -> List[str]:
    options_raw = request.args.get("options")
    if options_raw:
        return options_raw.split(",")
    return []


@app.route("/symbols/<symbol>/ohlc/<duration>/<int:w>/<int:h>/<theme>")
async def get_chart(symbol: str, duration: str, w: int, h: int, theme: str):
    user = await get_user()
    await web_charts.include_template(w, h, theme)
    stock = await repository.get_stock(user, symbol)
    options = parse_options()
    return await render_ohlc(stock, duration, w, h, theme, options)


@app.route("/symbols/<symbol>/notes", methods=["POST"])
async def notes(symbol: str):
    user = await get_user()
    raw = await request.get_data()
    parsed = json.loads(raw)
    stock = await repository.get_stock(user, symbol)
    if parsed["body"]:
        log.info(f"{symbol:6} notes:saving {stock.key()}")
        stock = await repository.save_notes(
            user, symbol, parsed["notedPrice"], parsed["body"]
        )
        log.info(f"{symbol:6} notes:saved {stock.key()}")
    return await assemble_stock_view_model(stock)


@app.route("/css/<path:path>")
async def send_css(path: str):
    return await send_from_directory("dist/css", path)


@app.route("/js/<path:path>")
async def send_js(path: str):
    return await send_from_directory("dist/js", path)


@app.route("/<path:path>")
async def send_spa(path: str):
    return await send_from_directory("dist", path)


@app.route("/")
async def send_index():
    return await send_from_directory("dist", "index.html")


def rounding(v: Decimal) -> Decimal:
    return v.quantize(Decimal("0.01"), ROUND_HALF_UP)


def maybe_round(v: Optional[Decimal]) -> Optional[Decimal]:
    return rounding(v) if v else None


def factory():
    setup_logging_queue()
    return app
