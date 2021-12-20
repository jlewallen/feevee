#!env/bin/python3

import logging, asyncio, os, aiofiles.os, json
import finnhub
from pytz import timezone
from time import mktime
from dateutil.relativedelta import relativedelta
from asyncio_throttle import Throttler  # type: ignore
from datetime import datetime, timedelta
from backend import MoneyCache, SymbolRepository, is_missing, write_json_file
from storage import UserKey, get_db

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


async def main():
    repository = SymbolRepository()
    db = await get_db()
    await db.open()

    try:
        log.info(f"querying user")
        user = await db.get_user_key_by_user_id(1)
        assert user

        log.info(f"querying portfolio")
        portfolio = await repository.get_portfolio(user)

        log.info(f"querying stocks")
        stocks = await repository.get_all_stocks(user, portfolio=portfolio)

        await get_earnings_calendar()

        today = datetime.now().replace(minute=0, second=0)
        today_str = (
            today.strftime("%Y%m%d") if False else today.strftime("%Y%m%d_%H%M%S")
        )

        for stock in stocks:
            if stock.info and stock.info.options:
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

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
