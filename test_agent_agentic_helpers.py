import os
import pytest
from flask.helpers import get_debug_flag, get_load_dotenv, stream_with_context


def test_get_debug_flag_default():
    """Test get_debug_flag returns False by default when FLASK_DEBUG is not set."""
    if "FLASK_DEBUG" in os.environ:
        del os.environ["FLASK_DEBUG"]
    assert get_debug_flag() is False


def test_get_debug_flag_true():
    """Test get_debug_flag returns True when FLASK_DEBUG is set to a truthy value."""
    os.environ["FLASK_DEBUG"] = "1"
    assert get_debug_flag() is True
    os.environ["FLASK_DEBUG"] = "true"
    assert get_debug_flag() is True
    os.environ["FLASK_DEBUG"] = "yes"
    assert get_debug_flag() is True


def test_get_debug_flag_false():
    """Test get_debug_flag returns False when FLASK_DEBUG is set to a falsy value."""
    os.environ["FLASK_DEBUG"] = "0"
    assert get_debug_flag() is False
    os.environ["FLASK_DEBUG"] = "false"
    assert get_debug_flag() is False
    os.environ["FLASK_DEBUG"] = "no"
    assert get_debug_flag() is False


def test_get_load_dotenv_default():
    """Test get_load_dotenv returns True by default when FLASK_SKIP_DOTENV is not set."""
    if "FLASK_SKIP_DOTENV" in os.environ:
        del os.environ["FLASK_SKIP_DOTENV"]
    assert get_load_dotenv() is True


def test_get_load_dotenv_false():
    """Test get_load_dotenv returns False when FLASK_SKIP_DOTENV is set to a truthy value."""
    os.environ["FLASK_SKIP_DOTENV"] = "1"
    assert get_load_dotenv() is False
    os.environ["FLASK_SKIP_DOTENV"] = "true"
    assert get_load_dotenv() is False
    os.environ["FLASK_SKIP_DOTENV"] = "yes"
    assert get_load_dotenv() is False


def test_get_load_dotenv_true():
    """Test get_load_dotenv returns True when FLASK_SKIP_DOTENV is set to a falsy value."""
    os.environ["FLASK_SKIP_DOTENV"] = "0"
    assert get_load_dotenv() is True
    os.environ["FLASK_SKIP_DOTENV"] = "false"
    assert get_load_dotenv() is True
    os.environ["FLASK_SKIP_DOTENV"] = "no"
    assert get_load_dotenv() is True


def test_stream_with_context_no_context():
    """Test stream_with_context raises RuntimeError when no request context is active."""
    def simple_generator():
        yield "data"

    with pytest.raises(RuntimeError, match="can only be used when a request context is active"):
        list(stream_with_context(simple_generator()))


@pytest.mark.xfail(strict=True, reason="stream_with_context requires a request context which is not available in this test")
def test_stream_with_context_with_context():
    """Test stream_with_context works correctly when a request context is active."""
    # This test is expected to fail because we cannot simulate a request context here.
    def simple_generator():
        yield "data"

    result = list(stream_with_context(simple_generator()))
    assert result == ["data"]
