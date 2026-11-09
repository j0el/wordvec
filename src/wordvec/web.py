"""Web front end for both demos: the same commands as the terminal programs,
served as a small terminal-style page.

    /en/   English (GloVe vectors)
    /zh/   Chinese (Tencent vectors + CC-CEDICT + jieba)

The page is a thin client. Every command line goes to POST /api/<lang>/run,
which calls the same `execute()` the terminal loop uses, with stdout captured,
and sends the text back. Walkthrough position lives in the browser and is sent
with each request, so several people can use it at once without sharing a
"current step".
"""

import contextlib
import io
import os
import threading
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, request, send_from_directory

import wordvec
from wordvec import chinese

STATIC = Path(__file__).parent / "static"
MAX_LINE = 200

# Most of the program is plain Python with module-level state (the loaded
# vectors, each walkthrough's current step, redirected stdout), so one command
# runs at a time. Commands take milliseconds; the models are the slow part.
_run_lock = threading.Lock()


class English:
    code = "en"
    title = "How AI works - part 1"
    prompt = "how ai works>"
    banner = wordvec.BANNER
    examples = ["king - man + woman", "paris - france + italy",
                "walking - walk + swim", "doctor - man + woman",
                "near dog", "compare dog cat", "show dog"]

    def __init__(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):  # silence "Loading..." progress
            self.model = wordvec.load_model()
        self.walks = wordvec.make_walkthroughs()

    def execute(self, active, line):
        return wordvec.execute(self.model, self.walks, active, line)


class Chinese:
    code = "zh"
    title = "How AI works - part 1 (中文)"
    prompt = "how ai works>"
    banner = chinese.BANNER
    examples = ["king - man + woman", "beijing - china + japan",
                "国王 - 男人 + 女人", "doctor - man + woman",
                "near dog", "compare dog cat", "translate 我喜欢吃苹果"]

    def __init__(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.model, self.dictionary, self.walks = chinese.load_all()

    def execute(self, active, line):
        return chinese.execute(self.model, self.dictionary, self.walks, active, line)


# (button label, command). Same on both pages; the panels differ by language.
BUTTONS = [("Full story", "full"), ("Tour", "tour"), ("Build", "build"),
           ("Generate", "generate"), ("Twins", "twins"), ("Context", "context"),
           ("Attend", "attend")]


def create_app(languages=None):
    """Build the Flask app. `languages` is injectable so tests can skip the
    multi-hundred-MB model loads."""
    if languages is None:
        languages = {lang.code: lang for lang in (English(), Chinese())}
    app = Flask(__name__, static_folder=None)

    def language_or_404(code):
        if code not in languages:
            abort(404)
        return languages[code]

    @app.get("/")
    def root():
        return redirect("/en/")

    @app.get("/healthz")
    def healthz():
        return "ok"

    @app.get("/<code>/")
    def page(code):
        language_or_404(code)
        response = send_from_directory(STATIC, "index.html")
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/static/<path:name>")
    def static_file(name):
        response = send_from_directory(STATIC, name)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/<code>/info")
    def info(code):
        language = language_or_404(code)
        return jsonify(code=language.code, title=language.title,
                       prompt=language.prompt, banner=language.banner,
                       examples=language.examples,
                       buttons=[{"label": l, "command": c} for l, c in BUTTONS])

    @app.post("/api/<code>/run")
    def run(code):
        language = language_or_404(code)
        body = request.get_json(silent=True) or {}
        line = body.get("line", "")
        before = body.get("state") or {}
        if not isinstance(line, str) or len(line) > MAX_LINE:
            return jsonify(error=f"Keep it under {MAX_LINE} characters."), 400

        with _run_lock:
            active = language.walks.get(before.get("walk"))
            if active is not None:
                try:
                    active.step = int(before.get("step", 0))
                except (TypeError, ValueError):
                    active.step = 0
            buffer = io.StringIO()
            # Steps may ask for pictures (wordvec.figure); collect them here.
            wordvec.FIGURE_SINK = figures = []
            try:
                with contextlib.redirect_stdout(buffer):
                    try:
                        active = language.execute(active, line)
                    except Exception as error:  # a demo shouldn't show a stack trace
                        print(f"  Something went wrong: {error}\n")
                        app.logger.exception("command failed: %r", line)
            finally:
                wordvec.FIGURE_SINK = None
            if active is wordvec.QUIT:
                active = None

        after = ({"walk": active.name, "step": active.step, "total": len(active.steps)}
                 if active is not None else None)
        # The terminal clears the screen on each walkthrough step; the page does
        # the same when the step changed.
        clear = after is not None and (
            after["walk"] != before.get("walk") or after["step"] != before.get("step"))
        return jsonify(output=buffer.getvalue(), state=after, clear=clear,
                       figures=figures)

    return app


def build_app():
    """Entry point for gunicorn: wordvec.web:build_app()"""
    return create_app()


def main() -> None:
    """Dev server: `uv run wordvec-web`."""
    port = int(os.environ.get("PORT", "8000"))
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=port)


if __name__ == "__main__":
    main()
