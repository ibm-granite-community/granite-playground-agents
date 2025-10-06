from collections.abc import Generator
from typing import TypeVar

from pydantic import SecretStr
from pydantic_settings import BaseSettings

from granite_core.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


def batch(lst: list[T], batch_size: int) -> Generator[list[T], None, None]:
    """
    Yield successive batches of `batch_size` from `lst`.
    """
    length = len(lst)
    for i in range(0, length, batch_size):
        yield list(lst[i : i + batch_size])


def get_secret_value(setting: SecretStr | None) -> str | None:
    return None if setting is None else setting.get_secret_value()


def log_settings(settings: BaseSettings, name: str = "Core") -> None:
    logger.info(f"{name} Settings...")
    logger.info(f"  {'Name':<40}{'Default':<40}{'Value':<40}")
    logger.info(f"  {'====':<40}{'=======':<40}{'=====':<40}")
    for field_name, field_info in type(settings).model_fields.items():
        actual_value = getattr(settings, field_name)
        default_value_str = "(required)" if field_info.is_required() else repr(field_info.default)
        logger.info(f"  {field_name:<40}{default_value_str:<40}{actual_value!r:<40}")
