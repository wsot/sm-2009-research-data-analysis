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
    ctx.run("mypy src")


@invoke.task
def lint(ctx):
    """
    Check linting in the src folder
    """
    ctx.run("flake8 src")

@invoke.task
def test(ctx):
    """
    Run pytest
    """
    ctx.run("pytest")


@invoke.task(format, typing, lint, test)
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
