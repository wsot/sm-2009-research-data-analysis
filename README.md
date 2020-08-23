# Simeon Morgan's 2009 Research Data Extractor

In 2009, I did some research. Now I need to get data from that for publication.
All the original data analysis was written in VBA, which was a nightmarish hellscape
of bad cod.

This time, it will be better.

## Summary

This is very much work-in-progress. As in, it doesn't work yet.
There's a publication being written and reviewers wanted some additional data.
I wanted to use the opportunity to write some decent code and get my head
around the TDT library. (Maybe numpy, matplotlib, etc a bit too) and so
here we are.

The intention is to create a backing library that does all the hard work, and
that can be used from simple interface scripts, or from Jupyter Lab.

## Getting Started

This project uses Poetry for dependency management and Invoke for task execution.

### Prerequisites

- Python 3.8+
- Poetry

### Installation and use

- Set up your virtual environment using your preferred method (mine is `pyenv`)
- Install dependencies with `poetry install` (or `poetry install --nodev` you do don't want dev tools)
- Start JupyterLab with `poetry run jupyter lab`
- ?
- Profit!