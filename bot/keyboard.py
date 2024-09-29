from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

main_keys = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text="Текущий контекст")],
    [KeyboardButton(text="Загрузить контекст")]
], resize_keyboard=True)