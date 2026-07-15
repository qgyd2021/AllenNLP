#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Any, Dict

from jinja2 import Template


def jinja2_render_template(template: str, context: Dict[str, Any]) -> Template:
    overrides_text = str(template or "").strip()
    if not overrides_text:
        return overrides_text
    return Template(overrides_text).render(**context)


if __name__ == "__main__":
    pass
