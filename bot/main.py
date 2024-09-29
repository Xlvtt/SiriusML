import asyncio
from create_bot import bot, dispatcher
from handlers import router


async def main():
    dispatcher.include_router(router)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())