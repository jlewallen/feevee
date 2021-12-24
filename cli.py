#!env/bin/python3

import logging, asyncio, os, aiofiles.os, json
import QuantLib as ql
import matplotlib.pyplot as plt
import finnhub
import argparse
from pytz import timezone
from time import mktime
from dateutil.relativedelta import relativedelta
from asyncio_throttle import Throttler  # type: ignore
from datetime import datetime, timedelta
from backend import (
    MoneyCache,
    StockInfo,
    SymbolRepository,
    is_missing,
    write_json_file,
    Financials,
)
from charts import get_relative_daily_prices_path, get_relative_intraday_prices_path
from storage import Criteria, UserKey, get_db

FinnHubKey = os.environ["FINN_HUB_KEY"] if "FINN_HUB_KEY" in os.environ else None
throttler: Throttler = Throttler(rate_limit=10, period=60)
log = logging.getLogger("feevee")


async def get_earnings_calendar():
    start = datetime(2021, 10, 1)
    one_month_from_today = datetime.today() + relativedelta(months=1)

    while start < one_month_from_today:
        end = start + relativedelta(months=1)
        start_str = start.strftime("%Y-%m-%d")
        calendar_path = f"earnings-{start_str}.json"
        log.info(f"{start} -> {end} ({calendar_path})")

        if await is_missing(calendar_path) and FinnHubKey:
            async with throttler:
                fc = finnhub.Client(api_key=FinnHubKey)
                res = fc.earnings_calendar(
                    _from=start_str,
                    to=end.strftime("%Y-%m-%d"),
                    symbol="",
                )
                await write_json_file(res, calendar_path)

        start += relativedelta(months=1)


async def price_option():
    maturity_date = ql.Date(15, 1, 2016)
    spot_price = 170
    strike_price = 165
    volatility = 0.20  # the historical vols or implied vols
    dividend_rate = None  # 0.0063
    option_type = ql.Option.Call

    risk_free_rate = 0.001
    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates()

    calculation_date = ql.Date(8, 5, 2015)
    ql.Settings.instance().evaluationDate = calculation_date

    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    settlement = calculation_date

    am_exercise = ql.AmericanExercise(settlement, maturity_date)
    american_option = ql.VanillaOption(payoff, am_exercise)

    eu_exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, eu_exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, risk_free_rate, day_count)
    )
    flat_vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(calculation_date, calendar, volatility, day_count)
    )

    if dividend_rate:
        dividend_yield = ql.YieldTermStructureHandle(  # type: ignore
            ql.FlatForward(calculation_date, dividend_rate, day_count)
        )
        bsm_process = ql.BlackScholesMertonProcess(
            spot_handle, dividend_yield, flat_ts, flat_vol_ts
        )
    else:
        bsm_process = ql.BlackScholesProcess(spot_handle, flat_ts, flat_vol_ts)

    nsteps = 200
    american_option.setPricingEngine(
        ql.BinomialVanillaEngine(bsm_process, "crr", nsteps)
    )
    european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

    def binomial_price(option, bsm_process, steps):
        option.setPricingEngine(ql.BinomialVanillaEngine(bsm_process, "crr", steps))
        return option.NPV()

    steps = range(5, nsteps, 1)
    eu_prices = [binomial_price(european_option, bsm_process, step) for step in steps]
    am_prices = [binomial_price(american_option, bsm_process, step) for step in steps]
    bs_price = american_option.NPV()

    log.info(f"{bs_price}")

    # plt.plot(steps, eu_prices, label="European Option", lw=2, alpha=0.6)
    plt.plot(steps, am_prices, label="American Option", lw=2, alpha=0.6)
    plt.plot([5, 200], [bs_price, bs_price], "r--", label="BSM Price", lw=2, alpha=0.6)
    plt.xlabel("Steps")
    plt.ylabel("Price")
    plt.ylim(6, 20)
    plt.title("Binomial Tree Price For Varying Steps")
    plt.legend()
    plt.show()


async def main():
    parser = argparse.ArgumentParser(description="feevee cli tool")
    parser.add_argument("--prices", action="store_true", default=False)
    parser.add_argument("--options", action="store_true", default=False)
    parser.add_argument("--append", action="store_true", default=False)
    args = parser.parse_args()

    repository = SymbolRepository()
    db = await get_db()
    await db.open()

    try:
        log.info(f"querying user")
        user = await db.get_user_key_by_user_id(1)
        assert user

        log.info(f"user: {user}")

        stock_info = StockInfo(user)

        log.info(f"querying portfolio")
        portfolio = await repository.get_portfolio(user)

        log.info(f"querying stocks")
        stocks = await repository.get_all_stocks(
            user,
            portfolio,
            stock_info,
            Criteria(),
        )

        today = datetime.now().replace(minute=0, second=0)
        today_str = (
            today.strftime("%Y%m%d") if False else today.strftime("%Y%m%d_%H%M%S")
        )

        financials = Financials()

        for stock in stocks:
            if args.prices:
                daily_path = os.path.join(
                    MoneyCache, get_relative_daily_prices_path(stock.symbol)
                )
                if not os.path.isfile(daily_path) or args.append:
                    await financials.query(stock.symbol, daily_path, False)

                intraday_path = os.path.join(
                    MoneyCache, get_relative_intraday_prices_path(stock.symbol)
                )
                if not os.path.isfile(intraday_path) or args.append:
                    await financials.query(stock.symbol, intraday_path, True)

            if args.options and stock.info and stock.info.options:
                log.info(f"{stock.symbol:6} processing")

                todays_chain_path = f"chain-{today_str}-{stock.symbol}.json"
                if await is_missing(todays_chain_path):
                    async with throttler:
                        fc = finnhub.Client(api_key=FinnHubKey)
                        res = json.loads(fc.option_chain(symbol=stock.symbol))
                        await write_json_file(res, todays_chain_path)

    finally:
        log.info(f"closing")
        await db.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)7s %(name)s %(message)s",
    )

    logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
