import pathlib
import sys

import invoke


@invoke.task
def format(ctx):
    """
    Apply automatic code formating tools
    """
    ctx.run("black src")
    ctx.run("isort .")


@invoke.task
def typing(ctx):
    """
    Check type annotations
    """
    ctx.run("mypy src --html-report htmlmypy")

@invoke.task
def lint(ctx):
    """
    Check linting in the src folder
    """
    ctx.run("flake8 src")

@invoke.task
def test(ctx, name=""):
    """
    Run pytest
    """
    if name:
        ctx.run(f"pytest --no-cov -sk {name}", pty=True)
    else:
        ctx.run("pytest --cov=lib --cov-report=term --no-cov-on-fail --cov-context=test")
        ctx.run("coverage html --show-contexts")

@invoke.task(format, lint, test, typing)
def check(ctx):
    """
    Runs all the code checking tools
    """
    ...

@invoke.task
def lab(ctx):
    """
    Start Jupyter Lab
    """
    ctx.run("jupyter lab")
