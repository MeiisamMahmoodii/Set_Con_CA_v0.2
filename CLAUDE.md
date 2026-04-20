# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

- Run the main application: `uv run python main.py`
- Check CUDA availability: `uv run python tes.py`
- Add a new dependency: `uv add <package>`
- Update dependencies: `uv lock --upgrade`

## Project Structure

- `main.py`: The entry point of the application.
- `tes.py`: A utility script to verify GPU/CUDA availability.
- `pyproject.toml`: Project configuration and dependency management via `uv`.
- `test.ipynb`: Jupyter notebook for interactive testing and experimentation.

## Architecture

This project is a Python-based project leveraging `torch` for GPU-accelerated computations. It uses `uv` for high-performance dependency management and environment orchestration.
