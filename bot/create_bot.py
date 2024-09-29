import logging
from aiogram import Bot, Dispatcher
from decouple import config
from aiogram.fsm.storage.memory import MemoryStorage


admins = [int(admin_id) for admin_id in config('ADMINS').split(',')]

logging.basicConfig(level=logging.INFO)

bot = Bot(token=config('API_TOKEN'))
dispatcher = Dispatcher(storage=MemoryStorage())