from typing import Optional, Dict, List
from datetime import datetime
import asyncio, concurrent.futures
import logging, os

log = logging.getLogger("feevee")
pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
times: Dict[str, datetime] = dict()


def _get_directory(path: str) -> Dict[str, datetime]:
    try:
        log.info(f"directory:get {path}")
        directory = {}
        for top, children, files in os.walk(path):
            for key, fs in {
                os.path.join(top, file): os.stat(os.path.join(top, file))
                for file in files
            }.items():
                directory[key] = datetime.fromtimestamp(fs.st_mtime)
        return directory
    except:
        log.exception(f"get-directory")
        raise


async def _shutdown_pool(pool: concurrent.futures.ProcessPoolExecutor):
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


async def get_directory(path: str) -> Dict[str, datetime]:
    global pool
    if pool is None:
        pool = concurrent.futures.ProcessPoolExecutor()
        asyncio.create_task(_shutdown_pool(pool))

    loop = asyncio.get_running_loop()
    directory = await loop.run_in_executor(pool, _get_directory, path)

    for key, time in directory.items():
        times[key] = time

    return directory
