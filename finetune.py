"""Backward-compatible public finetuning entrypoint."""

from cli.finetune_cli import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
