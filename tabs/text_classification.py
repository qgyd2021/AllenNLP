#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from allennlp.models.archival import load_archive
from allennlp.predictors.text_classifier import TextClassifierPredictor

from project_settings import project_path
from toolbox.modelscope.download import ensure_model_downloaded
from toolbox.torch.model_manager import TTLModelManager
from toolbox.jinja2.utils import jinja2_render_template


logger = logging.getLogger(__name__)

MODEL_DIR = project_path / "trained_models/modelscope"
EMPTY_PROB_TABLE = [["", ""]]


def _notify_model_download_start(model_id: str) -> None:
    logger.info(
        "正在从 ModelScope 下载模型：{}，首次下载可能较慢，请耐心等待…".format(model_id)
    )


def _build_model_key(archive_dir: str, cuda_device: int, overrides: dict) -> str:
    spec = {
        "archive_dir": str(Path(archive_dir).resolve()),
        "cuda_device": cuda_device,
        "overrides": overrides,
    }
    return json.dumps(spec, ensure_ascii=False, sort_keys=True)


def _load_predictor(key: str):
    spec = json.loads(key)
    archive = load_archive(
        archive_file=spec["archive_dir"],
        cuda_device=spec["cuda_device"],
        overrides=spec.get("overrides") or {},
    )
    predictor = TextClassifierPredictor(
        model=archive.model,
        dataset_reader=archive.dataset_reader,
    )
    runtime_device = "cuda:0" if spec["cuda_device"] >= 0 else "cpu"
    return predictor, runtime_device


PREDICTOR_MANAGER = TTLModelManager(
    loader=_load_predictor,
    gpu_idle_seconds=300,
    cpu_idle_seconds=900,
    sweep_interval=30,
    prefer_gpu=False,
    relocate_on_acquire=False,
)


def _detect_gpu_info() -> Dict[str, Any]:
    if torch.cuda.is_available():
        return {
            "available": True,
            "device": "cuda:0",
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
        }
    return {"available": False}


def _resolve_device(device_choice: str) -> int:
    if str(device_choice).startswith("GPU"):
        return 0
    return -1


def _format_prob_table(predictor: TextClassifierPredictor, outputs: dict) -> List[List[Any]]:
    index_to_label = predictor._model.vocab.get_index_to_token_vocabulary("labels")
    labels = [index_to_label[index] for index in range(len(index_to_label))]
    probs = outputs.get("probs", [])
    rows = []
    for label, prob in sorted(zip(labels, probs), key=lambda item: item[1], reverse=True):
        rows.append([label, round(float(prob), 4)])
    return rows or EMPTY_PROB_TABLE


def parse_archive_overrides(overrides_text: str) -> Dict[str, Any]:
    overrides_text = str(overrides_text or "").strip()
    if not overrides_text:
        return {}
    overrides = json.loads(overrides_text)
    if not isinstance(overrides, dict):
        raise ValueError("overrides 必须是 JSON 对象")
    return overrides


def when_predict_button_click(
    model_id: str,
    device_choice: str,
    overrides_text: str,
    text: str,
) -> Tuple[str, float, List[List[Any]], str]:
    text = str(text or "").strip()
    if not text:
        message = {"error": "请输入待分类文本"}
        return "", 0.0, EMPTY_PROB_TABLE, json.dumps(message, ensure_ascii=False, indent=2)

    model_id = str(model_id).strip()
    cuda_device = _resolve_device(device_choice)

    try:
        archive_dir, downloaded = ensure_model_downloaded(model_id, MODEL_DIR)
        overrides_text = jinja2_render_template(overrides_text, {
            "archive_dir": archive_dir.as_posix()
        })
        overrides = parse_archive_overrides(overrides_text)
        model_key = _build_model_key(
            archive_dir.as_posix(),
            cuda_device=cuda_device,
            overrides=overrides,
        )
        start = time.perf_counter()
        with PREDICTOR_MANAGER.use(model_key) as (predictor, runtime_device):
            outputs = predictor.predict_json({"sentence": text})
        elapsed = time.perf_counter() - start

        label = outputs["label"]
        probs = outputs["probs"]
        max_prob = round(float(max(probs)), 4)
        prob_table = _format_prob_table(predictor, outputs)
        message = {
            "elapsed_seconds": round(elapsed, 3),
            "device": runtime_device,
            "device_choice": device_choice,
            "model_id": model_id,
            "archive_dir": archive_dir.as_posix(),
            "downloaded": downloaded,
            "overrides": overrides,
            "model_manager": {
                "gpu_idle_seconds": PREDICTOR_MANAGER.gpu_idle_seconds,
                "cpu_idle_seconds": PREDICTOR_MANAGER.cpu_idle_seconds,
                "cached_models": PREDICTOR_MANAGER.state(),
            },
        }
        if runtime_device.startswith("cuda") and torch.cuda.is_available():
            message["gpu_name"] = torch.cuda.get_device_name(0)
        return label, max_prob, prob_table, json.dumps(message, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as exc:
        message = {"error": "overrides JSON 格式错误: {}".format(exc)}
        return "", 0.0, EMPTY_PROB_TABLE, json.dumps(message, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("帖子领域分类失败")
        message = {"error": "分类失败: {}".format(exc)}
        return "", 0.0, EMPTY_PROB_TABLE, json.dumps(message, ensure_ascii=False, indent=2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tab_settings_file",
        default=(project_path / "data/tab_settings/voc_post_domain_tab_settings.json").as_posix(),
        type=str,
    )
    return parser.parse_args()


def get_text_classification_tab():
    import gradio as gr

    args = get_args()

    with open(args.tab_settings_file, "r", encoding="utf-8") as f:
        settings = json.load(f)

    examples = settings["examples"]
    init_params = settings["init_params"]
    model_choices = init_params["model_choices"]

    gpu_info = _detect_gpu_info()
    device_choices = ["CPU"]
    if gpu_info["available"]:
        device_choices.append("GPU ({})".format(gpu_info["name"]))

    with gr.TabItem("文本分类"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 模型配置")
                model_id = gr.Dropdown(
                    label="ModelScope 模型",
                    choices=model_choices,
                    value=model_choices[0],
                    interactive=True,
                    allow_custom_value=True,
                )
                overrides = gr.Textbox(
                    label="Archive Overrides (JSON)",
                    placeholder='{"dataset_reader.tokenizer.model_name": "{{ archive_dir }}/tokenizer"}',
                )
                gr.Markdown("### 推理设备")
                device = gr.Radio(
                    label="推理设备",
                    choices=device_choices,
                    value="CPU",
                    interactive=gpu_info["available"],
                )
                if gpu_info["available"]:
                    gpu_device_info = gr.Textbox(
                        label="GPU 设备",
                        value=gpu_info.get("name", ""),
                        interactive=False,
                    )

                gr.Markdown("### 文本输入")
                text = gr.Textbox(
                    label="帖子文本",
                    lines=8,
                    placeholder="请输入帖子标题、描述或正文内容",
                )
                predict_button = gr.Button("开始分类", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 分类输出")
                result_label = gr.Textbox(label="预测标签")
                result_prob = gr.Number(label="最高概率", precision=4)
                result_probs = gr.Dataframe(
                    label="各类别概率",
                    headers=["label", "prob"],
                    datatype=["str", "number"],
                    interactive=False,
                )
                result_message = gr.Textbox(label="Message", lines=10)

        gr.Examples(
            label="Examples",
            examples=examples,
            inputs=[model_id, device, overrides, text],
            outputs=[result_label, result_prob, result_probs, result_message],
            fn=when_predict_button_click,
            cache_examples=False,
        )

        predict_button.click(
            when_predict_button_click,
            inputs=[model_id, device, overrides, text],
            outputs=[result_label, result_prob, result_probs, result_message],
        )

    return locals()
