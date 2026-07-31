import pytest
from flask import Flask, Blueprint, template_rendered
from jinja2 import DictLoader, Environment as Jinja2Environment, TemplateNotFound

from flask.templating import (
    _default_template_ctx_processor,
    Environment,
    DispatchingJinjaLoader,
    render_template,
    render_template_string,
    stream_template,
    stream_template_string,
)


def test_default_template_ctx_processor_without_request():
    app = Flask(__name__)
    with app.app_context():
        ctx = _default_template_ctx_processor()
        assert "g" in ctx
        assert "request" not in ctx


def test_default_template_ctx_processor_with_request():
    app = Flask(__name__)
    with app.test_request_context("/"):
        ctx = _default_template_ctx_processor()
        assert "g" in ctx
        assert "request" in ctx


def test_environment_sets_loader_and_app():
    app = Flask(__name__)
    env = Environment(app)
    # environment should retain a reference to the app
    assert env.app is app
    # loader should be set (Flask.create_global_jinja_loader provides a loader)
    assert env.loader is not None


def test_dispatching_jinja_loader_get_source_and_list_templates_merge_blueprints():
    app = Flask(__name__)
    app.jinja_loader = DictLoader({"a.html": "A content"})
    bp = Blueprint("bp", __name__)
    # give blueprint its own jinja loader
    bp.jinja_loader = DictLoader({"b.html": "B content"})
    app.register_blueprint(bp)

    loader = DispatchingJinjaLoader(app)
    env = Jinja2Environment()

    src, filename, uptodate = loader.get_source(env, "a.html")
    assert "A content" in src

    srcb, _, _ = loader.get_source(env, "b.html")
    assert "B content" in srcb

    templates = loader.list_templates()
    assert set(templates) == {"a.html", "b.html"}


def test_dispatching_jinja_loader_get_source_not_found_raises():
    app = Flask(__name__)
    # no loader on app and no blueprints -> nothing will be found
    app.jinja_loader = None
    loader = DispatchingJinjaLoader(app)
    env = Jinja2Environment()
    with pytest.raises(TemplateNotFound):
        loader.get_source(env, "missing.html")


def test_render_template_renders_and_emits_signal_and_updates_context():
    app = Flask(__name__)
    app.jinja_loader = DictLoader({"t.html": "Hello {{ name }}!"})

    received = {}

    def record(sender, template, context):
        # template is a jinja2 Template; name should be the template name
        received["template_name"] = template.name
        received["context"] = dict(context)

    with app.app_context():
        with template_rendered.connected_to(record, app):
            rv = render_template("t.html", name="World")
            assert rv == "Hello World!"
        # ensure signal handler ran and provided the template/context
        assert received["template_name"] in ("t.html",)
        assert received["context"]["name"] == "World"


def test_render_template_selects_first_existing_from_list():
    app = Flask(__name__)
    app.jinja_loader = DictLoader({"b.html": "B: {{ val }}"})

    with app.app_context():
        rv = render_template(["missing.html", "b.html"], val=42)
        assert "B: 42" in rv


def test_render_template_string_uses_context_and_emits_signal():
    app = Flask(__name__)
    received = {}

    def record(sender, template, context):
        received["name"] = context.get("name")

    with app.app_context():
        with template_rendered.connected_to(record, app):
            rv = render_template_string("Inline: {{ name }}", name="Inliner")
            assert "Inline: Inliner" in rv
        assert received["name"] == "Inliner"


def test_stream_template_string_yields_and_emits_signal_when_consumed():
    app = Flask(__name__)
    received = {"called": False}

    def record(sender, template, context):
        received["called"] = True

    with app.app_context():
        with template_rendered.connected_to(record, app):
            iterator = stream_template_string("Streamed: {{ name }}", name="S")
            parts = list(iterator)  # consume the stream; signal should be emitted after consumption
            text = "".join(parts)
            assert "Streamed: S" in text
        assert received["called"] is True


@pytest.mark.xfail(
    reason="Expected behavior: _default_template_ctx_processor should include 'current_app' in the context, "
    "but implementation only inserts 'g' and 'request'.",
    strict=True,
)
def test_default_ctx_processor_includes_current_app():
    """Probe for missing context entry: current_app should be available by default."""
    app = Flask(__name__)
    with app.app_context():
        ctx = _default_template_ctx_processor()
        # We assert the desirable behavior here; current implementation does not include it.
        assert "current_app" in ctx


@pytest.mark.xfail(
    reason="Signal is only emitted after the entire stream is consumed; this test expects it on partial consumption.",
    strict=True,
)
def test_stream_template_signal_on_partial_consumption():
    """Assert (but currently not true) that template_rendered signal fires even on partial stream consumption."""
    app = Flask(__name__)
    called = {"flag": False}

    def record(sender, template, context):
        called["flag"] = True

    with app.app_context():
        with template_rendered.connected_to(record, app):
            iterator = stream_template_string("Stream: {{ x }}", x="1")
            # consume only part of the iterator
            try:
                next(iterator)
            except StopIteration:
                # if the template was tiny, it may finish; that's fine for this probe
                pass
            # We expect the signal to have fired already; implementation sends it only after generation completes.
            assert called["flag"] is True