# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

from pydantic import SecretStr
from pydantic_settings import BaseSettings
from pytest import LogCaptureFixture

from granite_core.utils import batch, get_secret_value, log_settings


def test_batch() -> None:
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    batch_size = 3
    expected_batches = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
    assert list(batch(lst, batch_size)) == expected_batches

    # Test with empty list
    assert list(batch([], batch_size)) == []

    # Test with batch size larger than list length
    assert list(batch(lst, len(lst) + 1)) == [lst]


def test_get_secret_value() -> None:
    secret_str = SecretStr("my_secret")
    assert get_secret_value(secret_str) == "my_secret"
    assert get_secret_value(None) is None


def test_log_settings(caplog: LogCaptureFixture) -> None:
    class Settings(BaseSettings):
        field1: str = "value1"
        field2: int = 42
        field3: bool = True

    settings = Settings()
    log_settings(settings)

    assert "Core Settings..." in caplog.text
    assert "field1                                  'value1'                                'value1'" in caplog.text
    assert "field2                                  42                                      42" in caplog.text
    assert "field3                                  True                                    True " in caplog.text

    # Test with custom name
    caplog.clear()
    log_settings(settings, name="Custom")
    assert "Custom Settings..." in caplog.text
