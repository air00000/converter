import asyncio
import io
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp
import gdown
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


def _extract_drive_folder_id(parsed_url) -> Optional[str]:
    parts = [part for part in parsed_url.path.split("/") if part]
    if "folders" in parts:
        idx = parts.index("folders")
        if idx + 1 < len(parts):
            return parts[idx + 1]

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


async def _download_google_drive_folder(folder_id: str) -> List[Tuple[bytes, Optional[str]]]:
    loop = asyncio.get_running_loop()

    with tempfile.TemporaryDirectory() as tmpdir:

        def _download_folder() -> List[str]:
            return gdown.download_folder(
                id=folder_id,
                output=tmpdir,
                quiet=True,
                use_cookies=False,
            ) or []

        try:
            downloaded_paths = await loop.run_in_executor(None, _download_folder)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "Не удалось скачать файлы из папки Google Drive. Проверь доступ по ссылке."
            ) from exc

        files: List[Tuple[bytes, Optional[str]]] = []
        for raw_path in downloaded_paths:
            path = Path(raw_path)
            if not path.is_file():
                continue
            try:
                relative_path = path.relative_to(tmpdir)
                relative_name = relative_path.as_posix()
            except ValueError:
                relative_name = path.name
            files.append((path.read_bytes(), relative_name))

        if not files:
            raise ValueError(
                "Не удалось найти файлы в папке Google Drive. Проверь, что доступ открыт для просмотра."
            )

        return files


async def _download_yandex_disk_file(
    session: aiohttp.ClientSession,
    public_key: str,
    path: Optional[str] = None,
    suggested_name: Optional[str] = None,
) -> Tuple[bytes, Optional[str]]:
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"

    params = {"public_key": public_key}
    if path:
        params["path"] = path

    async with session.get(api_url, params=params) as meta_response:
        meta_response.raise_for_status()
        meta = await meta_response.json()

    href = meta.get("href")
    if not href:
        raise ValueError("Не удалось получить ссылку для скачивания с Яндекс Диска.")

    async with session.get(href) as file_response:
        file_response.raise_for_status()
        content = await file_response.read()
        filename = _filename_from_headers(file_response.headers) or suggested_name

    return content, filename


async def _fetch_yandex_resource_meta(
    session: aiohttp.ClientSession,
    public_key: str,
    path: Optional[str] = None,
    offset: Optional[int] = None,
) -> Mapping[str, Any]:
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources"
    params: dict[str, Any] = {"public_key": public_key, "limit": 1000}
    if path:
        params["path"] = path
    if offset is not None:
        params["offset"] = offset

    async with session.get(api_url, params=params) as response:
        response.raise_for_status()
        return await response.json()


async def _download_yandex_disk_folder(
    session: aiohttp.ClientSession,
    public_key: str,
    root_meta: Mapping[str, Any],
) -> List[Tuple[bytes, Optional[str]]]:
    results: List[Tuple[bytes, Optional[str]]] = []

    async def _walk(path: Optional[str], prefix: Path, meta: Mapping[str, Any]) -> None:
        embedded = meta.get("_embedded", {}) if isinstance(meta, Mapping) else {}
        items = embedded.get("items", []) if isinstance(embedded, Mapping) else []

        if not items and prefix == Path():
            raise ValueError("Не удалось найти файлы в папке на Яндекс Диске.")

        async def _process_items(current_items: Iterable[Mapping[str, Any]]) -> None:
            for item in current_items:
                if not isinstance(item, Mapping):
                    continue
                item_type = item.get("type")
                item_name = item.get("name") or "file"
                item_path = item.get("path")
                relative = (prefix / item_name).as_posix()

                if item_type == "file":
                    content, _ = await _download_yandex_disk_file(
                        session,
                        public_key,
                        path=item_path,
                        suggested_name=item_name,
                    )
                    results.append((content, relative))
                elif item_type == "dir":
                    child_meta = await _fetch_yandex_resource_meta(
                        session, public_key, path=item_path
                    )
                    await _walk(item_path, Path(relative), child_meta)

        await _process_items(items)

        embedded_meta = embedded if isinstance(embedded, Mapping) else {}
        total = embedded_meta.get("total", len(items)) or 0
        limit = embedded_meta.get("limit", len(items)) or len(items)
        if limit <= 0:
            limit = len(items) or 1
        offset = embedded_meta.get("offset", 0) or 0

        while offset + limit < total:
            offset += limit
            next_meta = await _fetch_yandex_resource_meta(
                session, public_key, path=path, offset=offset
            )
            next_embedded = (
                next_meta.get("_embedded", {}) if isinstance(next_meta, Mapping) else {}
            )
            next_items = next_embedded.get("items", []) if isinstance(next_embedded, Mapping) else []
            limit = next_embedded.get("limit", len(next_items)) or len(next_items)
            if limit <= 0:
                limit = len(next_items) or 1
            offset = next_embedded.get("offset", offset) or offset
            await _process_items(next_items)

    await _walk(None, Path(), root_meta)

    if not results:
        raise ValueError("Не удалось найти файлы в папке на Яндекс Диске.")

    return results


async def _download_files_from_link(url: str) -> List[Tuple[bytes, Optional[str]]]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Похоже, что ссылка имеет неподдерживаемый формат.")

    domain = parsed.netloc.lower()

    if "drive.google.com" in domain:
        folder_id = _extract_drive_folder_id(parsed)
        if folder_id:
            return await _download_google_drive_folder(folder_id)

    async with aiohttp.ClientSession() as session:
        if "drive.google.com" in domain:
            file_id = _extract_drive_file_id(parsed)
            query_params = parse_qs(parsed.query)
            folder_id_fallback = query_params.get("id", [None])[0]

            if file_id:
                try:
                    content, filename = await _download_google_drive_file(session, file_id)
                    return [(content, filename)]
                except ValueError:
                    if folder_id_fallback and folder_id_fallback == file_id:
                        return await _download_google_drive_folder(folder_id_fallback)
                    raise

            if folder_id_fallback:
                return await _download_google_drive_folder(folder_id_fallback)

            raise ValueError("Не удалось определить идентификатор файла Google Drive.")

        if any(domain.endswith(part) for part in ("yadi.sk", "disk.yandex.ru")):
            root_meta = await _fetch_yandex_resource_meta(session, url)
            resource_type = root_meta.get("type") if isinstance(root_meta, Mapping) else None

            if resource_type == "file":
                content, filename = await _download_yandex_disk_file(
                    session, url, suggested_name=root_meta.get("name") if isinstance(root_meta, Mapping) else None
                )
                return [(content, filename)]

            if resource_type == "dir":
                return await _download_yandex_disk_folder(session, url, root_meta)

            raise ValueError("Не удалось определить тип ресурса Яндекс Диска по ссылке.")

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
    await message.answer(
        "Отлично! Теперь пришли файл в виде документа или ссылку на файл/папку. "
        "Я верну его в нужном формате."
    )



def _output_stem_from_source(source_name: Optional[str]) -> str:
    if not source_name:
        return "converted"

    candidate = source_name.replace("\\", "/").split("/")[-1]
    stem = Path(candidate).stem
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)
    return sanitized or "converted"


def _output_filename(stem: str, part_index: int, total_parts: int) -> str:
    if total_parts <= 1:
        return f"{stem}.txt"
    return f"{stem}_part_{part_index}.txt"


async def _convert_and_send(
    message: Message,
    state: FSMContext,
    file_content: bytes,
    source_name: Optional[str] = None,
) -> bool:
    data = await state.get_data()
    input_mask = data.get("input_mask")
    output_mask = data.get("output_mask")

    lines_per_file = int(data.get("lines_per_file", 0) or 0)

    if not input_mask or not output_mask:
        await message.answer("Сначала отправь маски с помощью команды /start.")
        return False

    text = file_content.decode("utf-8", errors="ignore")
    try:
        converted_lines = _convert_lines(text.splitlines(), input_mask, output_mask)
    except ValueError as exc:
        await message.answer(f"Не удалось преобразовать файл: {exc}")
        return False

    if not converted_lines:
        await message.answer("Не удалось найти строки, подходящие под входную маску.")
        return False

    if lines_per_file > 0:
        chunks = [
            converted_lines[idx : idx + lines_per_file]
            for idx in range(0, len(converted_lines), lines_per_file)
        ]
    else:
        chunks = [converted_lines]

    if len(chunks) > 1:
        details = f" для {source_name}" if source_name else ""
        await message.answer(
            f"Готово! Разбил результат на {len(chunks)} файла(ов){details}."
        )

    stem = _output_stem_from_source(source_name)
    for idx, chunk in enumerate(chunks, start=1):
        output_bytes = "\n".join(chunk).encode("utf-8")
        filename = _output_filename(stem, idx, len(chunks))
        output_file = BufferedInputFile(output_bytes, filename=filename)

        caption: Optional[str]
        if len(chunks) == 1:
            caption = (
                f"Готово! Вот преобразованный файл для {source_name}."
                if source_name
                else "Готово! Вот преобразованный файл."
            )
        else:
            caption = None

        try:
            await message.answer_document(output_file, caption=caption)
        except TelegramBadRequest:
            await message.answer(
                "Не удалось отправить файл. Попробуй файл меньшего размера."
            )
            return False

    return True


async def handle_file(message: Message, state: FSMContext, bot: Bot) -> None:
    document = message.document
    if not document:
        await message.answer("Пожалуйста, отправь файл как документ (не как фотографию).")
        return

    file = await bot.get_file(document.file_id)
    buffer = io.BytesIO()
    await bot.download_file(file.file_path, buffer)
    buffer.seek(0)

    success = await _convert_and_send(
        message,
        state,
        buffer.read(),
        source_name=document.file_name,
    )

    if success:
        await state.clear()


async def handle_file_link(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    match = URL_PATTERN.search(text)
    if not match:
        await message.answer(
            "Пожалуйста, отправь документ или ссылку на файл/папку в Google Drive или на Яндекс Диске."
        )
        return

    url = _strip_trailing_punctuation(match.group(0))

    try:
        files = await _download_files_from_link(url)
    except ValueError as exc:
        await message.answer(str(exc))
        return
    except aiohttp.ClientError:
        await message.answer("Не удалось скачать файл по ссылке. Попробуй позже или отправь документ.")
        return

    if not files:
        await message.answer("Не удалось найти файлы по указанной ссылке.")
        return

    if len(files) > 1:
        await message.answer(
            f"Нашёл {len(files)} файла(ов) по ссылке. Начинаю обработку..."
        )

    success_any = False
    for content, name in sorted(files, key=lambda item: item[1] or ""):
        success = await _convert_and_send(message, state, content, source_name=name)
        success_any = success_any or success

    if success_any:
        await state.clear()
        if len(files) > 1:
            await message.answer("Готово! Обработал все доступные файлы по ссылке.")


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
