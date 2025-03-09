import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from dotenv import load_dotenv
from chatgpt import ask

#загрузка данных из .env-файла
load_dotenv('key.env')
bot_key = os.getenv('API_TOKEN')

#Инициализация бота и диспетчера
bot = Bot(token = bot_key)
dp = Dispatcher()

#хэндлер на команду "start"
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.reply("Привет! Я бот, интегрированный с ChatGPT. Задай мне какой-нибудь вопрос, и я отвечу на него.")

#хендлер на команду "help"
@dp.message(Command("help"))     
async def cmd_start(message: types.Message):
    await message.answer("Бот разработан с поддержкой ChatGPT 3.5 Turbo. Тематика базы данных - чемпионат мира по футболу 2018 года, информация взята из Википедии. Бот обучен на основе базы знаний с 11514 строками. Поддерживаются запросы так и на русском, так и на английском языках.\n\n Команды бота:\n start — запуск бота\n help — справка, список комманд бота")

#хендлер на обработку сообщений
@dp.message()
async def echo(message: types.Message):
    #отправка сообщения в ChatGPT
    response = await ask(message.text)

    #отправка ответа пользователю
    await message.answer(response)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())