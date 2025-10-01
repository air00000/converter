import io
import zipfile
import aiohttp
import gdown
import tempfile
import os
import re
from typing import Optional, Callable
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import BufferedInputFile
from dotenv import load_dotenv

# Загружаем токен из .env
load_dotenv()
API_TOKEN = os.getenv("BOT_TOKEN")

if not API_TOKEN:
    raise ValueError("Не найден BOT_TOKEN в .env файле!")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

MAX_TG_FILE_SIZE = 49 * 1024 * 1024  # ~49 MB
user_sessions = {}


# ==== Утилиты ====

async def download_file_from_url(url: str) -> str:
    """Скачивание текстового файла по ссылке"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                return (await resp.read()).decode("utf-8", errors="ignore")
    return ""


def download_gdrive_folder(folder_url: str) -> dict[str, str]:
    """
    Скачивает все файлы из папки Google Drive (по публичной ссылке).
    Возвращает словарь {имя_файла: содержимое}.
    """
    tmpdir = tempfile.mkdtemp()
    gdown.download_folder(folder_url, output=tmpdir, quiet=True, use_cookies=False)

    files_data = {}
    for root, _, files in os.walk(tmpdir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    files_data[fname] = f.read()
            except Exception:
                pass
    return files_data


def build_parser(input_format: str, output_format: str) -> Callable[[str], Optional[str]]:
    """
    Генерирует быстрый парсер для строк по заданному input_format и output_format.
    Работает через split, без regex.
    """
    fields = re.findall(r"\{(.*?)\}", input_format)

    # определяем разделитель
    pattern = re.sub(r"\{.*?\}", "§", input_format)
    sep = None
    if "§" in pattern:
        idx = pattern.find("§")
        if idx + 1 < len(pattern):
            sep = pattern[idx + 1]
    if not sep:
        sep = ":"  # по умолчанию

    def parser(line: str) -> Optional[str]:
        parts = line.rstrip("\n").split(sep, maxsplit=len(fields)-1)
        if len(parts) < len(fields):
            return None
        values = dict(zip(fields, parts))
        try:
            return output_format.format(**values)
        except Exception:
            return None

    return parser


def split_and_zip(lines: list[str], lines_per_file: int, base_name: str):
    """
    Делим строки и упаковываем в ZIP архивы (в памяти), чтобы не превышать лимит Telegram
    """
    archives = []
    part_num = 1
    buffer = io.BytesIO()
    zipf = zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED)
    current_size = 0

    for i in range(0, len(lines), lines_per_file):
        chunk = lines[i:i + lines_per_file]
        file_content = "\n".join(chunk).encode("utf-8")
        fname = f"{base_name}_part_{i // lines_per_file + 1}.txt"
        zipf.writestr(fname, file_content)
        current_size += len(file_content)

        if current_size >= MAX_TG_FILE_SIZE:
            zipf.close()
            buffer.seek(0)
            archives.append((f"{base_name}_result_{part_num}.zip", buffer.read()))
            part_num += 1
            buffer = io.BytesIO()
            zipf = zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED)
            current_size = 0

    zipf.close()
    buffer.seek(0)
    archives.append((f"{base_name}_result_{part_num}.zip", buffer.read()))

    return archives


# ==== Хендлеры ====

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    user_sessions[message.from_user.id] = {}
    await message.answer("Привет! Отправь формат входных данных (например {mail}:{mailpass}:{username}:{password})")


@dp.message()
async def process_message(message: types.Message):
    uid = message.from_user.id
    session = user_sessions.get(uid, {})

    if "input_format" not in session:
        session["input_format"] = message.text.strip()
        await message.answer("Теперь введи желаемый формат (например {username}:{password})")
    elif "output_format" not in session:
        session["output_format"] = message.text.strip()
        await message.answer("Сколько строк должно быть в одном файле?")
    elif "lines_per_file" not in session:
        if not message.text.isdigit():
            await message.answer("Введи число!")
            return
        session["lines_per_file"] = int(message.text)
        await message.answer("Теперь отправь файл (txt) или ссылку на облако (Google Drive, обычная ссылка).")
    else:
        files_to_process = {}

        # Если документ
        if message.document:
            file = await bot.get_file(message.document.file_id)
            stream = await bot.download_file(file.file_path)
            file_text = stream.read().decode("utf-8", errors="ignore")
            files_to_process[message.document.file_name] = file_text

        # Если ссылка
        elif message.text.startswith("http"):
            if "drive.google.com/drive/folders" in message.text:
                # это папка Google Drive
                files_to_process = download_gdrive_folder(message.text)
            else:
                file_text = await download_file_from_url(message.text)
                files_to_process["input.txt"] = file_text

        if not files_to_process:
            await message.answer("Не удалось получить файл(ы)")
            return

        # Создаём быстрый парсер
        parser = build_parser(session["input_format"], session["output_format"])

        # Обрабатываем каждый файл отдельно
        for fname, file_text in files_to_process.items():
            processed = []
            for line in file_text.splitlines():
                new_line = parser(line)
                if new_line:
                    processed.append(new_line)

            if not processed:
                await message.answer(f"Файл {fname}: не удалось обработать (формат не совпадает).")
                continue

            archives = split_and_zip(processed, session["lines_per_file"], base_name=fname)

            for name, data in archives:
                await message.answer_document(
                    BufferedInputFile(data, filename=name)
                )

            await message.answer(f"Файл {fname} обработан ✅")

    user_sessions[uid] = session


if __name__ == "__main__":
    import asyncio
    asyncio.run(dp.start_polling(bot))
