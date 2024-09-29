from aiogram import Router, F
from aiogram import types
from decouple import config
from aiogram.filters import Command, CommandStart
from aiogram.filters.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from transformers import AutoModelForQuestionAnswering
import keyboard
from model import inference

router = Router()

model_name = config("MODEL_NAME")
model = None
context = None  # TODO как хранить переменные в боте


class SetContextStates(StatesGroup):
    default_state = State()
    changing_context = State()


@router.message(CommandStart())
async def start_command(message: types.Message):
    text = "Привет! Я жеский лютый бот, который умеет искать ответы на любые вопросы в любом тексте, вот так вот!!\n\n"

    text += "Вот такие вот у меня есть команды:\n"
    text += "/start - вывести эту справочку еще раз\n"
    text += "/set_context - задать текст для поиска ответов\n"
    text += "/get_context - посмотреть текущий контекст\n"
    text += "\n Чтобы задать вопрос, просто наберите его в поле для сообщений)"
    global model
    if model is None:
        model = AutoModelForQuestionAnswering.from_pretrained(f"../{model_name}")
    await message.answer(text, reply_markup=keyboard.main_keys)


@router.message(Command("set_context"))
async def set_context_command(message: types.Message, state: FSMContext):
    if context is None:
        await message.answer("Отправьте первый контекст)")
    else:
        await message.answer("Отправьте новый контекст)")
    await state.set_state(SetContextStates.changing_context)


@router.message(SetContextStates.changing_context)
async def decided_to_change_context(message: types.Message, state: FSMContext):
    global context
    context = message.text
    await message.answer("Контекст был успешно обновлен!")
    await state.clear()


@router.message(F.text == "Загрузить контекст")
async def set_context_command_keyboard(message: types.Message, state: FSMContext):
    await set_context_command(message, state)


@router.message(Command("get_context"))
async def get_context_command(message: types.Message):
    global context
    if context is None:
        await message.answer("Вы еще не устанавливали контекст(((")
    else:
        await message.answer(context)


@router.message(F.text == "Текущий контекст")
async def get_context_command_keyboard(message: types.Message):
    await get_context_command(message)


@router.message()
async def request_command(message: types.Message):
    if model is None:
        await message.answer("Выполните команду /start, чтобы запустить бота!")
    elif context is None:
        await message.answer("Отличный вопрос! Но сначала нужно установить текст для поиска ответа!")
    else:
        data = {"context": [context], "question": [message.text], "id": [0]}
        answer = inference(model, data)[0]["text"]
        await message.answer(answer)
