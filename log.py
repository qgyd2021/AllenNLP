#!/usr/bin/python3
# -*- coding: utf-8 -*-
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import os
from zoneinfo import ZoneInfo


def get_converter(tz_info: str = "Asia/Shanghai"):
    def converter(timestamp):
        dt = datetime.fromtimestamp(timestamp, ZoneInfo(tz_info))
        return dt.timetuple()
    return converter


def setup_size_rotating(log_directory: str, tz_info: str = "Asia/Shanghai"):
    fmt = "%(asctime)s|%(name)s|%(levelname)s|%(filename)s|%(lineno)d|%(message)s"

    formatter = logging.Formatter(
        fmt=fmt,
        datefmt="%Y-%m-%d %H:%M:%S %z"
    )
    formatter.converter = get_converter(tz_info)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    main_logger = logging.getLogger("main")
    main_logger.addHandler(stream_handler)
    main_info_file_handler = RotatingFileHandler(
        filename=os.path.join(log_directory, "main.log"),
        maxBytes=100 * 1024 * 1024,
        encoding="utf-8",
        backupCount=2,
    )
    main_info_file_handler.setLevel(logging.INFO)
    main_info_file_handler.setFormatter(formatter)
    main_logger.addHandler(main_info_file_handler)

    api_logger = logging.getLogger("api")
    api_file_handler = RotatingFileHandler(
        filename=os.path.join(log_directory, "api.log"),
        maxBytes=10 * 1024 * 1024,
        encoding="utf-8",
        backupCount=2,
    )
    api_file_handler.setLevel(logging.DEBUG)
    api_file_handler.setFormatter(formatter)
    api_logger.addHandler(api_file_handler)

    toolbox_logger = logging.getLogger("toolbox")
    toolbox_logger.addHandler(stream_handler)
    toolbox_file_handler = RotatingFileHandler(
        filename=os.path.join(log_directory, "toolbox.log"),
        maxBytes=10 * 1024 * 1024,
        encoding="utf-8",
        backupCount=2,
    )
    toolbox_file_handler.setLevel(logging.DEBUG)
    toolbox_file_handler.setFormatter(formatter)
    toolbox_logger.addHandler(toolbox_file_handler)

    debug_file_handler = RotatingFileHandler(
        filename=os.path.join(log_directory, "debug.log"),
        maxBytes=1 * 1024 * 1024,
        encoding="utf-8",
        backupCount=2,
    )
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(formatter)

    info_file_handler = RotatingFileHandler(
        filename=os.path.join(log_directory, "info.log"),
        maxBytes=1 * 1024 * 1024,
        encoding="utf-8",
        backupCount=2,
    )
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(formatter)

    error_file_handler = RotatingFileHandler(
        filename=os.path.join(log_directory, "error.log"),
        maxBytes=1 * 1024 * 1024,
        encoding="utf-8",
        backupCount=2,
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.DEBUG,
        datefmt="%a, %d %b %Y %H:%M:%S",
        handlers=[
            debug_file_handler,
            info_file_handler,
            error_file_handler,
        ]
    )
