import logging, asyncio, itertools, os, aiofiles, json
from typing import Optional, Iterable, Callable, List, Any, Sequence
from datetime import datetime, timedelta
from pytz import timezone

log = logging.getLogger("feevee")

PriceDirectory = ".prices"


def get_money_cache() -> str:
    value = os.environ["MONEY_CACHE"]
    assert value
    return value


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
    json_path = os.path.join(os.path.join(get_money_cache(), PriceDirectory), path)
    await backup_daily_file(json_path)
    async with aiofiles.open(json_path, mode="w") as file:
        await file.write(json.dumps(data))


async def is_missing(path: str) -> bool:
    try:
        await aiofiles.os.stat(
            os.path.join(os.path.join(get_money_cache(), PriceDirectory), path)
        )
        return False
    except FileNotFoundError:
        return True


def finish_key(parts: Sequence[str]) -> str:
    return ",".join(parts)


def flatten(a):
    return [leaf for sl in a for leaf in sl]
