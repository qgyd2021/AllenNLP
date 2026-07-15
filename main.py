#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os

import gradio as gr

import log
from project_settings import log_directory, time_zone_info

log.setup_size_rotating(log_directory=log_directory, tz_info=time_zone_info)

from tabs.text_classification import get_text_classification_tab

logger = logging.getLogger("main")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_port",
        default=7860,
        type=int,
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Gradio launch 后会用 httpx 请求本机 startup-events 做自检。
    # Windows 若开了系统代理，httpx 会把 localhost 也走代理，导致 WinError 10054 启动失败。
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")

    with gr.Blocks(title="AllenNLP") as blocks:
        gr.Markdown(value="# AllenNLP")
        with gr.Tabs():
            _ = get_text_classification_tab()

    blocks.queue().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
