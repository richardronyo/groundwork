import pytest
import os
from datetime import timedelta
from flask.blueprints import Blueprint
from flask import Flask


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = None
    return app


def test_blueprint_initialization():
    blueprint = Blueprint(
        name='test',
        import_name=__name__,
        static_folder='static',
        static_url_path='/static',
        template_folder='templates',
        url_prefix='/prefix',
        subdomain='sub',
        url_defaults={'key': 'value'},
        root_path=os.getcwd(),
        cli_group='cli'
    )

    assert blueprint.name == 'test'
    assert blueprint.import_name == __name__
    assert blueprint.static_folder == os.path.join(os.getcwd(), 'static')
    assert blueprint.static_url_path == '/static'
    assert blueprint.template_folder == 'templates'
    assert blueprint.url_prefix == '/prefix'
    assert blueprint.subdomain == 'sub'
    assert blueprint.root_path == os.getcwd()
    assert blueprint.cli.name == 'test'


def test_get_send_file_max_age_default_none(app):
    blueprint = Blueprint(name='test', import_name=__name__)
    with app.app_context():
        assert blueprint.get_send_file_max_age('file.txt') is None


def test_get_send_file_max_age_with_timedelta(app):
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(hours=1)
    blueprint = Blueprint(name='test', import_name=__name__)
    with app.app_context():
        assert blueprint.get_send_file_max_age('file.txt') == 3600


def test_send_static_file_without_static_folder():
    blueprint = Blueprint(name='test', import_name=__name__)
    with pytest.raises(RuntimeError, match="'static_folder' must be set to serve static_files."):
        blueprint.send_static_file('file.txt')


def test_open_resource_valid_modes():
    blueprint = Blueprint(name='test', import_name=__name__, root_path=os.getcwd())
    # Create a temporary file to test reading
    test_file_path = os.path.join(os.getcwd(), 'test_resource.txt')
    with open(test_file_path, 'w') as f:
        f.write('test content')

    try:
        with blueprint.open_resource('test_resource.txt', 'r') as f:
            assert f.read() == 'test content'

        with blueprint.open_resource('test_resource.txt', 'rb') as f:
            assert f.read() == b'test content'
    finally:
        os.remove(test_file_path)


def test_open_resource_invalid_mode():
    blueprint = Blueprint(name='test', import_name=__name__, root_path=os.getcwd())
    with pytest.raises(ValueError, match="Resources can only be opened for reading."):
        blueprint.open_resource('test_resource.txt', 'w')
