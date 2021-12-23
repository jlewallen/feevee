from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass
import asyncio, concurrent.futures
import logging, os

log = logging.getLogger("feevee")
pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
times: Dict[str, datetime] = {}


@dataclass
class DirectoryEntry:
    path: str
    modified: datetime
    changed: bool


def _is_interesting(file_name: str) -> bool:
    if file_name[0] == ".":
        return False
    return "-daily.csv" in file_name or "-iday.csv" in file_name


def _stat_ignore_fnf(path: str):
    try:
        return os.stat(path)
    except FileNotFoundError:
        return None


def _get_directory(path: str) -> Dict[str, datetime]:
    try:
        log.debug(f"directory:get {path}")
        directory = {}
        for top, children, files in os.walk(path):
            for key, fs in {
                os.path.join(top, file): _stat_ignore_fnf(os.path.join(top, file))
                for file in files
                if _is_interesting(file)
            }.items():
                if fs:
                    directory[key] = datetime.fromtimestamp(fs.st_mtime)
        return directory
    except:
        log.exception(f"get-directory")
        raise


async def _shutdown_pool(pool: concurrent.futures.ThreadPoolExecutor):
    while True:
        try:
            await asyncio.sleep(1)
        except:
            log.info(f"stopping archive pool")
            pool.shutdown()
            break


def get_time(path: str) -> Optional[datetime]:
    if path in times:
        return times[path]
    return None


async def get_directory(path: str) -> Dict[str, DirectoryEntry]:
    global pool, times
    if pool is None:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        asyncio.create_task(_shutdown_pool(pool))

    loop = asyncio.get_running_loop()
    directory = await loop.run_in_executor(pool, _get_directory, path)

    entries: Dict[str, DirectoryEntry] = {}
    for key, time in directory.items():
        changed = key not in times or times[key] != time
        entries[key] = DirectoryEntry(key, time, changed)
        times[key] = time

    return entries
