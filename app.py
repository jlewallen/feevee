from typing import List, Dict, Optional, Sequence, Tuple
from decimal import Decimal, ROUND_HALF_UP
from quart import Quart, send_from_directory, request, Response
from quart import json as quart_json
from quart_cors import cors
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from aiocache import cached
from dataclasses import dataclass, field
import logging, json, asyncio, os, pandas

from backend import (
    ManageCandles,
    ManageDailies,
    ManageIndicators,
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
    load_symbol_prices,
    load_symbol_candles,
    load_months_of_symbol_prices,
)
from loggers import setup_logging_queue
from storage import UserKey, get_db
import charts, prices as pricing


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

    log.info(
        f"{stock.symbol:6} y={symbol_prices.yesterday} t={symbol_prices.today} c={symbol_prices.candle}"
    )

    price_change = symbol_prices.price_change()
    percent_change = symbol_prices.price_change_percentage()
    last_price = symbol_prices.price.price if symbol_prices.price else None

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

    virtual_tags = [] if stock.notes.tags else ["v:untagged"]
    virtual_tags.append("v:hold" if position else "v:watch")

    def is_nearby(target: Decimal, value: Decimal) -> bool:
        s = value * Decimal(0.9)
        e = value * Decimal(1.1)
        return target >= s and target <= e

    if price_change:
        virtual_tags.append("v:down" if price_change < 0 else "v:up")

    if last_price:
        if basis_price:
            if last_price > basis_price:
                virtual_tags.append("v:basis:above")
                if "exiting" in stock.notes.tags:
                    virtual_tags.append("v:sell")
            else:
                virtual_tags.append("v:basis:below")
                if "entering" in stock.notes.tags:
                    virtual_tags.append("v:buy")

        if symbol_prices.one_year_range:
            if is_nearby(symbol_prices.one_year_range[0], last_price):
                virtual_tags.append("v:year:low")

            if is_nearby(symbol_prices.one_year_range[1], last_price):
                virtual_tags.append("v:year:high")

        for np in stock.notes.prices:
            if is_nearby(np, last_price):
                virtual_tags.append("v:noted")
                break

    return dict(
        symbol=stock.symbol,
        version=stock.version,
        meta=stock.meta,
        key=finish_key(stock.key()),
        info=json.loads(stock.info.data) if stock.info else None,
        position=position,
        change=price_change,
        price=last_price,
        tags=stock.notes.tags + virtual_tags,
        lots=[lot_to_json(lot) for lot in symbol_lots],
        percent_change=maybe_round(percent_change),
        negative=percent_change < 0 if percent_change else False,
        noted_prices=stock.notes.prices,
        has_candles=not not symbol_prices.candle,
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


async def _render_ohlc(stock: Stock, prices: charts.Prices, w: int, h: int, style: str):
    symbol = stock.symbol
    theme = get_theme(style)
    basis_price = stock.lots.get_basis(symbol)
    last_buy_price = stock.lots.get_last_buy_price(symbol)

    marks = [
        charts.PriceMark(np, theme.noted_bg_color, theme.noted_text_color, False, True)
        for np in stock.notes.prices
    ]

    if last_buy_price:
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
        prices, symbol, size=(w, h), marks=marks, colors=theme.colors
    )
    return Response(data, mimetype="image/png")


def _render_candles_key(fn, stock: Stock, w: int, h: int, style: str) -> str:
    return finish_key(["candles"] + stock.key() + [str(v) for v in [w, h, style]])


@cached(key_builder=_render_candles_key, **Caching)
async def render_candles(stock: Stock, w: int, h: int, style: str):
    log.info(f"{stock.symbol:6} rendering {stock.key()} {w}x{h} {style}")
    candles = await load_symbol_candles(stock.symbol)
    return await _render_ohlc(stock, candles, w, h, style)


def _render_ohlc_key(fn, stock: Stock, months: int, w: int, h: int, style: str) -> str:
    return finish_key(["ohlc"] + stock.key() + [str(v) for v in [months, w, h, style]])


@cached(key_builder=_render_ohlc_key, **Caching)
async def render_ohlc(stock: Stock, months: int, w: int, h: int, style: str):
    log.info(f"{stock.symbol:6} rendering {stock.key()} {w}x{h} {months} {style}")
    prices = await load_months_of_symbol_prices(stock.symbol, months)
    return await _render_ohlc(stock, prices, w, h, style)


@dataclass
class ChartTemplate:
    w: int
    h: int


@dataclass
class WebChartsCacheWarmer(RefreshChartsHandler):
    templates: List[ChartTemplate] = field(default_factory=list)
    capacity: int = 6

    async def include_template(self, w: int, h: int):
        template = ChartTemplate(w, h)
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
            for theme in Themes:
                for months in [4, 12]:
                    log.debug(f"{m.symbol:6} web-charts:refresh {template} / {theme}")
                    await render_candles(stock, template.w, template.h, theme)
                    await render_ohlc(stock, months, template.w, template.h, theme)


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
candles = MessageWorker(
    ManageCandles(
        tag_priorities=DefaultTagPriorities,
    )
)
indicators = MessageWorker(ManageIndicators())
dailies = MessageWorker(ManageDailies(tag_priorities=DefaultTagPriorities))
web_charts = WebChartsCacheWarmer()
refreshing = RefreshQueue(
    repository, candles, dailies, MessageWorker(web_charts, concurrency=5), indicators
)
AdministratorUserId = 1


async def get_user() -> UserKey:
    await refreshing.start()

    db = await get_db()
    maybe_user = await db.get_user_key_by_user_id(AdministratorUserId)
    assert maybe_user
    log.debug(f"web:user: {maybe_user}")
    return maybe_user


@app.route("/status")
async def status():
    user = await get_user()
    stocks = await repository.get_all_stocks(user)
    view_models = [assemble_stock_view_model(stock) for stock in stocks]
    symbols = await asyncio.gather(*view_models)
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
    if request.args.get("daily"):
        await refreshing.push(RefreshDailyMessage(user, symbol))
    if request.args.get("candles"):
        await refreshing.push(RefreshCandlesMessage(user, symbol))
    if request.args.get("indicators"):
        await refreshing.push(RefreshIndicatorsMessage(user, symbol))


@app.route("/symbols", methods=["POST"])
async def add_symbols():
    user = await get_user()
    raw = await request.get_data()
    parsed: Dict[str, List[str]] = json.loads(raw)

    if parsed["adding"]:
        await repository.add_symbols(user, parsed["symbols"])
    else:
        raise NotImplementedError

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


@app.route("/symbols/<symbol>/ohlc/<int:months>/<int:w>/<int:h>/<style>")
async def get_chart(symbol: str, months: int, w: int, h: int, style: str):
    user = await get_user()
    await web_charts.include_template(w, h)
    stock = await repository.get_stock(user, symbol)
    return await render_ohlc(stock, months, w, h, style)


@app.route("/symbols/<symbol>/candles/<int:w>/<int:h>/<style>")
async def get_candles(symbol: str, w: int, h: int, style: str):
    user = await get_user()
    stock = await repository.get_stock(user, symbol)
    return await render_candles(stock, w, h, style)


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
