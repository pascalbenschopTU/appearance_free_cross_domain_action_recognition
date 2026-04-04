"""Backward-compatible public training entrypoint."""

from cli.train_cli import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
