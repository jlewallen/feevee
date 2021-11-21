from typing import Optional, List, Sequence, Tuple
from decimal import Decimal
from dataclasses import dataclass
from pandas import DataFrame, read_csv
from datetime import datetime
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import plotly.io as pio
import os, logging
import asyncio, concurrent.futures
import aiofiles, io

log = logging.getLogger("feevee")
pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
DailyDateColumn = "date"
DailyOpenColumn = "1. open"
DailyHighColumn = "2. high"
DailyLowColumn = "3. low"
DailyCloseColumn = "4. close"
DailyVolumeColumns = ["5. volume", "6. volume"]


@dataclass
class Colors:
    bg: str
    text: str
    grid: str
    volume: str


Light = Colors(bg="#ffffff", text="#000000", grid="#e6e6e6", volume="#000000")
Dark = Colors(bg="#111111", text="#ffffff", grid="#2e2e2e", volume="#999999")
Paper = Colors(bg="#fff1e5", text="#000000", grid="#e6e6e6", volume="#000000")


@dataclass
class MarketPrice:
    symbol: str
    price: Decimal


@dataclass
class Prices:
    symbol: str
    daily: DataFrame

    @property
    def empty(self) -> bool:
        return self.daily.empty

    def date_of_minimum(self) -> datetime:
        # idxmin wasn't working for me.
        date: Optional[datetime] = None
        minimum: Optional[Decimal] = None
        for key, value in self.daily[DailyLowColumn].items():
            if date is None or value < minimum:
                minimum = value
                date = key
        assert date
        return date

    def date_range(self) -> Tuple[datetime, datetime]:
        return (self.daily.index.min(), self.daily.index.max())

    def price_range(self) -> Tuple[Decimal, Decimal]:
        lows = self.daily[DailyLowColumn]
        highs = self.daily[DailyHighColumn]
        return (lows.min(), highs.max())

    def volume_column(self) -> str:
        for column in DailyVolumeColumns:
            if column in self.daily:
                return column
        raise Exception("no volume column")

    def volume_range(self) -> Tuple[Decimal, Decimal]:
        vols = self.daily[self.volume_column()]
        return (vols.min(), vols.max())

    def within_range(self, v: Decimal) -> bool:
        pr = self.price_range()
        return v >= pr[0] and v < pr[1]

    def has(self, d: datetime) -> bool:
        s = self.daily.index[0]
        e = self.daily.index[-1]
        return d >= s and d < e

    def minimum(self) -> Decimal:
        return self.daily[DailyLowColumn].min()

    def maximum(self) -> Decimal:
        return self.daily[DailyHighColumn].max()

    def history(self, start: datetime, end: datetime) -> "Prices":
        sliced = self.daily[start:end]  # type: ignore
        return Prices(self.symbol, sliced)

    def price_at(self, ts: Optional[datetime] = None) -> MarketPrice:
        assert ts
        ts = ts if ts else datetime.today()
        sliced = self.daily[ts:]  # type: ignore
        assert not sliced.empty
        value = Decimal(sliced[DailyCloseColumn][0])
        return MarketPrice(self.symbol, value)

    def clone(self) -> "Prices":
        return Prices(self.symbol, self.daily.copy())


def get_file_mtime(path: str, zero_on_no_file=True) -> float:
    if zero_on_no_file and not os.path.isfile(path):
        return 0
    return os.stat(path).st_mtime


def get_relative_candles_path(symbol: str) -> str:
    return f".av/{symbol}-candles.json"


def get_relative_candles_csv_path(symbol: str) -> str:
    return f".av/{symbol}-candles.csv"


def get_recommendations_path(symbol: str) -> str:
    return f".av/{symbol}-recs.json"


def get_relative_intraday_prices_path(symbol: str) -> str:
    return f".av/{symbol}-iday.csv"


def get_relative_daily_prices_path(symbol: str) -> str:
    return f".av/{symbol}-daily.csv"


def decimal_from_value(value):
    return Decimal(value)


def datetime_from_value(value):
    if ":" in value:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return datetime.strptime(value, "%Y-%m-%d")


CsvConverters = {
    DailyDateColumn: datetime_from_value,
    DailyCloseColumn: decimal_from_value,
    DailyOpenColumn: decimal_from_value,
    DailyLowColumn: decimal_from_value,
    DailyHighColumn: decimal_from_value,
}


async def write_prices_csv(path: str, df: DataFrame):
    async with aiofiles.open(path, mode="w") as f:
        if df.isnull().any(axis=1).sum() > 0:
            raise Exception(f"writing null rows: {path}")
        await f.write(df.to_csv())


async def read_prices_csv(path: str) -> DataFrame:
    async with aiofiles.open(path, mode="r") as f:
        with io.StringIO(await f.read()) as text_io:
            df = read_csv(
                text_io,
                index_col=0,
                parse_dates=[0],
                header=0,
                infer_datetime_format=True,
                converters=CsvConverters,
            )
            return df.sort_index()


@dataclass
class PriceCache:
    base: str
    symbol: str

    async def load(self) -> Prices:
        path = os.path.join(self.base, get_relative_daily_prices_path(self.symbol))
        prices = await read_prices_csv(path)
        return Prices(self.symbol, prices)


@dataclass
class PriceMark:
    price: Decimal
    color: str
    text_color: str
    line: bool = True
    constrain: bool = False


def make_y_arrow(
    value: Decimal, text: str, opacity: float, color: str, text_color: str
):
    return dict(
        x=1.0,
        y=value,
        xref="x domain",
        yref="y2",
        xshift=5,
        opacity=opacity,
        showarrow=True,
        ax=4,
        ay=0,
        arrowsize=0.8,
        arrowcolor=color,
        xanchor="left",
        text=text + " ",
        bordercolor=color,
        borderwidth=2,
        borderpad=1,
        bgcolor=color,
        font=dict(size=11, color=text_color),
    )


def _render_ohlc(
    prices: Prices,
    title: str,
    size: Tuple[int, int],
    marks: List[PriceMark],
    colors: Colors,
):
    show_last_value_marker = True

    low, high = prices.price_range()
    min_volume, max_volume = prices.volume_range()

    axis_defaults = dict(
        showline=True,
        zeroline=True,
        tickfont=dict(size=12, color=colors.text),
        titlefont=dict(size=16, color=colors.text),
    )

    template = go.layout.Template()
    template_annotations = [
        dict(
            name=title,
            text=title,
            textangle=0,
            opacity=0.2,
            font=dict(color=colors.text, size=32),
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.50,
            y=0.50,
        ),
    ]

    have_data = len(prices.daily.index) > 1

    if have_data:
        template_annotations.append(
            dict(
                name="time",
                text=prices.daily.index[-1].strftime("%m/%d %H:%M:%S"),
                textangle=0,
                opacity=0.2,
                font=dict(color=colors.text, size=12),
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.17,
            ),
        )

    template.layout.annotations = template_annotations

    pio.templates["draft"] = template

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    volume = go.Bar(
        x=prices.daily.index,
        y=prices.daily[prices.volume_column()],
        visible=True,
        showlegend=False,
        name=title,
        xaxis="x",
        yaxis="y",
        marker_color=colors.volume,
        opacity=0.25,
    )

    fig.add_trace(volume)

    ohlc = go.Ohlc(
        x=prices.daily.index,
        open=prices.daily[DailyOpenColumn],
        high=prices.daily[DailyHighColumn],
        low=prices.daily[DailyLowColumn],
        close=prices.daily[DailyCloseColumn],
        line=go.ohlc.Line(width=1.5),
        name=title,
        visible=True,
        showlegend=False,
        xaxis="x2",
        yaxis="y2",
    )

    fig.add_trace(ohlc, secondary_y=True)

    if show_last_value_marker and have_data:
        price_range = prices.price_range()

        def constrain(value: Decimal) -> Decimal:
            return min(max(value, price_range[0]), price_range[1])

        value = prices.daily[DailyCloseColumn][-1]
        annotations = []
        annotations += [
            make_y_arrow(
                constrain(mark.price) if mark.constrain else mark.price,
                f"{mark.price}",
                1.0,
                mark.color,
                mark.text_color,
            )
            for mark in marks
            if not mark.line
        ]
        annotations += [make_y_arrow(value, f"{value}", 1, colors.text, colors.bg)]
        fig.update_layout(annotations=annotations)

    fig.update_layout(
        shapes=[
            dict(
                y0=mark.price,
                y1=mark.price,
                x0=0,
                x1=1,
                yref="y2",
                xref="paper",
                line_width=0.5,
                line_color=mark.color,
            )
            for mark in marks
            if mark.line and prices.within_range(mark.price)
        ],
    )

    fig.update_layout(
        template="plotly_dark+draft",
        autosize=False,
        width=size[0],
        height=size[1],
        margin=dict(l=10, r=40, b=50, t=10, pad=4),
        paper_bgcolor=colors.bg,
        plot_bgcolor=colors.bg,
        xaxis=dict(
            side="bottom",
            gridcolor=colors.grid,
            **axis_defaults,
        ),
        yaxis=dict(
            side="left",
            range=[min_volume, max_volume * 3],
            showticklabels=False,
            showline=False,
            zeroline=False,
            showgrid=False,
        ),
        xaxis2=dict(
            showticklabels=False,
            showline=False,
            zeroline=False,
            showgrid=False,
        ),
        yaxis2=dict(
            side="right",
            gridcolor=colors.grid,
            **axis_defaults,
        ),
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
    )

    return fig.to_image(format="png")


async def _shutdown_pool(pool: concurrent.futures.ProcessPoolExecutor):
    while True:
        try:
            await asyncio.sleep(1)
        except:
            log.info(f"stopping chart pool")
            pool.shutdown()
            break


async def ohlc(
    prices: Prices,
    title: str,
    size: Tuple[int, int],
    marks: List[PriceMark],
    colors: Colors,
):
    global pool
    if pool is None:
        pool = concurrent.futures.ProcessPoolExecutor()
        asyncio.create_task(_shutdown_pool(pool))

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        pool, _render_ohlc, prices, title, size, marks, colors
    )
