#!env/bin/python3

import logging, asyncio

log = logging.getLogger("feevee")


async def main():
    pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)7s %(message)s",
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
