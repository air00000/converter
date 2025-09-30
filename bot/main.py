import asyncio
import io
import logging
import os
import re
from typing import Iterable, List, Mapping, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, Message
from dotenv import load_dotenv


PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][\w-]*)\}")
URL_PATTERN = re.compile(r"https?://\S+")


class ConversionStates(StatesGroup):
    input_mask = State()
    output_mask = State()
    lines_per_file = State()
    waiting_file = State()


def _extract_placeholders(mask: str) -> List[str]:
    return [match.group(1) for match in PLACEHOLDER_PATTERN.finditer(mask)]


def _mask_to_regex(mask: str) -> re.Pattern[str]:
    placeholders = list(PLACEHOLDER_PATTERN.finditer(mask))
    if not placeholders:
        raise ValueError("Маска должна содержать хотя бы один плейсхолдер в фигурных скобках.")

    pattern_parts: List[str] = []
    last_index = 0

    for idx, placeholder in enumerate(placeholders):
        pattern_parts.append(re.escape(mask[last_index : placeholder.start()]))
        name = placeholder.group(1)
        quantifier = ".+" if idx == len(placeholders) - 1 else ".+?"
        pattern_parts.append(f"(?P<{name}>{quantifier})")
        last_index = placeholder.end()

    pattern_parts.append(re.escape(mask[last_index:]))
    pattern = "".join(pattern_parts)
    return re.compile(rf"^{pattern}$")


def _convert_lines(lines: Iterable[str], input_mask: str, output_mask: str) -> List[str]:
    regex = _mask_to_regex(input_mask)
    required_fields = set(_extract_placeholders(output_mask))
    converted: List[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        match = regex.match(line)
        if not match:
            continue

        values = match.groupdict()

        missing = required_fields.difference(values)
        if missing:
            raise ValueError(
                "В выходной маске используются отсутствующие поля: " + ", ".join(sorted(missing))
            )

        converted.append(output_mask.format_map(values))

    return converted


def _strip_trailing_punctuation(url: str) -> str:
    return url.rstrip(".,)>]}\n\r")


def _filename_from_headers(headers: Optional[Mapping[str, str]]) -> Optional[str]:
    content_disposition = headers.get("Content-Disposition") if headers else None
    if not content_disposition:
        return None

    match = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition)
    if match:
        return unquote(match.group(1))

    match = re.search(r'filename="?([^";]+)"?', content_disposition)
    if match:
        return match.group(1)

    return None


def _extract_drive_file_id(parsed_url) -> Optional[str]:
    if parsed_url.path.startswith("/file/d/"):
        parts = parsed_url.path.split("/")
        try:
            file_index = parts.index("d") + 1
            return parts[file_index]
        except (ValueError, IndexError):
            return None

    query = parse_qs(parsed_url.query)
    for key in ("id", "fid"):
        if key in query and query[key]:
            return query[key][0]

    path_parts = parsed_url.path.split("/")
    if "uc" in path_parts:
        try:
            idx = path_parts.index("uc")
            if idx + 1 < len(path_parts):
                return path_parts[idx + 1]
        except ValueError:
            pass

    return None


async def _download_google_drive_file(
    session: aiohttp.ClientSession, file_id: str
) -> Tuple[bytes, Optional[str]]:
    download_url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}

    async with session.get(download_url, params=params) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")

        if "text/html" not in content_type:
            content = await response.read()
            filename = _filename_from_headers(response.headers)
            return content, filename or f"google-drive-{file_id}"

        page = await response.text()
        token_match = re.search(r"confirm=([0-9A-Za-z_]+)", page)
        if not token_match:
            raise ValueError(
                "Не удалось получить файл из Google Drive. Проверь, что доступ по ссылке открыт."
            )

        params["confirm"] = token_match.group(1)

    async with session.get(download_url, params=params) as response:
        response.raise_for_status()
        content = await response.read()
        filename = _filename_from_headers(response.headers)

    return content, filename or f"google-drive-{file_id}"


async def _download_yandex_disk_file(session: aiohttp.ClientSession, public_key: str) -> Tuple[bytes, Optional[str]]:
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"

    async with session.get(api_url, params={"public_key": public_key}) as meta_response:
        meta_response.raise_for_status()
        meta = await meta_response.json()

    href = meta.get("href")
    if not href:
        raise ValueError("Не удалось получить ссылку для скачивания с Яндекс Диска.")

    async with session.get(href) as file_response:
        file_response.raise_for_status()
        content = await file_response.read()
        filename = _filename_from_headers(file_response.headers)

    return content, filename


async def _download_file_from_link(url: str) -> Tuple[bytes, Optional[str]]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Похоже, что ссылка имеет неподдерживаемый формат.")

    domain = parsed.netloc.lower()

    async with aiohttp.ClientSession() as session:
        if "drive.google.com" in domain:
            file_id = _extract_drive_file_id(parsed)
            if not file_id:
                raise ValueError("Не удалось определить идентификатор файла Google Drive.")
            return await _download_google_drive_file(session, file_id)

        if any(domain.endswith(part) for part in ("yadi.sk", "disk.yandex.ru")):
            return await _download_yandex_disk_file(session, url)

    raise ValueError("Поддерживаются только ссылки на Google Drive и Яндекс Диск.")


async def handle_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "👋 Привет! Отправь мне маску текущей структуры строк файла.\n"
        "Используй плейсхолдеры в фигурных скобках, например:\n"
        "`{protocol}://{domain}:{username}:{password}`",
        parse_mode=ParseMode.MARKDOWN,
    )
    await state.set_state(ConversionStates.input_mask)


async def handle_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Сбросил текущую операцию. Напиши /start, чтобы начать заново.")


async def handle_input_mask(message: Message, state: FSMContext) -> None:
    mask = message.text or ""
    try:
        _mask_to_regex(mask)
    except ValueError as exc:
        await message.answer(str(exc))
        return

    await state.update_data(input_mask=mask)
    await state.set_state(ConversionStates.output_mask)
    await message.answer(
        "Отлично! Теперь отправь маску нужной структуры.\n"
        "Используй те же имена плейсхолдеров, например: `{username}:{password}`",
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_output_mask(message: Message, state: FSMContext) -> None:
    mask = message.text or ""
    if not PLACEHOLDER_PATTERN.search(mask):
        await message.answer("В маске должен быть хотя бы один плейсхолдер в фигурных скобках.")
        return

    await state.update_data(output_mask=mask)

    await state.set_state(ConversionStates.lines_per_file)
    await message.answer(
        "Если хочешь разделить результат на несколько файлов, введи количество строк на один файл.\n"
        "Отправь 0 или слово 'skip', чтобы получить один файл.",
    )


async def handle_lines_per_file(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip().lower()

    if not text:
        await message.answer("Пожалуйста, введи число строк или 0, чтобы пропустить.")
        return

    if text in {"skip", "0", "нет", "no"}:
        lines_per_file = 0
    else:
        try:
            lines_per_file = int(text)
        except ValueError:
            await message.answer("Не удалось распознать число. Попробуй снова.")
            return

        if lines_per_file <= 0:
            lines_per_file = 0

    await state.update_data(lines_per_file=lines_per_file)
    await state.set_state(ConversionStates.waiting_file)
    await message.answer("Отлично! Теперь пришли файл в виде документа. Я верну его в нужном формате.")



async def _convert_and_send(message: Message, state: FSMContext, file_content: bytes) -> None:
    data = await state.get_data()
    input_mask = data.get("input_mask")
    output_mask = data.get("output_mask")

    lines_per_file = int(data.get("lines_per_file", 0) or 0)

    if not input_mask or not output_mask:
        await message.answer("Сначала отправь маски с помощью команды /start.")
        return

    text = file_content.decode("utf-8", errors="ignore")
    try:
        converted_lines = _convert_lines(text.splitlines(), input_mask, output_mask)
    except ValueError as exc:
        await message.answer(f"Не удалось преобразовать файл: {exc}")
        return

    if not converted_lines:
        await message.answer("Не удалось найти строки, подходящие под входную маску.")
        return

    chunks: List[List[str]]
    if lines_per_file > 0:
        chunks = [
            converted_lines[idx : idx + lines_per_file]
            for idx in range(0, len(converted_lines), lines_per_file)
        ]
    else:
        chunks = [converted_lines]

    if len(chunks) > 1:
        await message.answer(f"Готово! Разбил результат на {len(chunks)} файла(ов).")

    for idx, chunk in enumerate(chunks, start=1):
        output_bytes = "\n".join(chunk).encode("utf-8")
        filename = "converted.txt" if len(chunks) == 1 else f"converted_part_{idx}.txt"
        output_file = BufferedInputFile(output_bytes, filename=filename)

        try:
            await message.answer_document(
                output_file,
                caption=("Готово! Вот преобразованный файл." if len(chunks) == 1 else None),
            )
        except TelegramBadRequest:
            await message.answer("Не удалось отправить файл. Попробуй файл меньшего размера.")
            return


    await state.clear()


async def handle_file(message: Message, state: FSMContext, bot: Bot) -> None:
    document = message.document
    if not document:
        await message.answer("Пожалуйста, отправь файл как документ (не как фотографию).")
        return

    file = await bot.get_file(document.file_id)
    buffer = io.BytesIO()
    await bot.download_file(file.file_path, buffer)
    buffer.seek(0)

    await _convert_and_send(message, state, buffer.read())


async def handle_file_link(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    match = URL_PATTERN.search(text)
    if not match:
        await message.answer(
            "Пожалуйста, отправь документ или ссылку на файл в Google Drive или на Яндекс Диске."
        )
        return

    url = _strip_trailing_punctuation(match.group(0))

    try:
        file_content, _ = await _download_file_from_link(url)
    except ValueError as exc:
        await message.answer(str(exc))
        return
    except aiohttp.ClientError:
        await message.answer("Не удалось скачать файл по ссылке. Попробуй позже или отправь документ.")
        return

    await _convert_and_send(message, state, file_content)


async def main() -> None:
    load_dotenv()
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise RuntimeError("Укажите токен бота в переменной окружения BOT_TOKEN")

    logging.basicConfig(level=logging.INFO)

    bot = Bot(token=token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()

    dp.message.register(handle_start, CommandStart())
    dp.message.register(handle_cancel, Command(commands=["cancel"]))
    dp.message.register(handle_file, ConversionStates.waiting_file, F.document)
    dp.message.register(handle_file_link, ConversionStates.waiting_file, F.text)

    dp.message.register(handle_lines_per_file, ConversionStates.lines_per_file, F.text)

    dp.message.register(handle_output_mask, ConversionStates.output_mask, F.text)
    dp.message.register(handle_input_mask, ConversionStates.input_mask, F.text)

    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
