#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path


project_path = os.path.abspath(os.path.dirname(__file__))
project_path = Path(project_path)

time_zone_info = "Asia/Shanghai"

log_directory = project_path / "logs"
log_directory.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    pass
