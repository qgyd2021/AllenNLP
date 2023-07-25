#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Tuple
from allennlp.training.optimizers import Registrable, Optimizer, make_parameter_groups
from pytorch_pretrained_bert.optimization import BertAdam
import torch


@Optimizer.register("bert_adam")
class BertAdamOptimizer(Optimizer, BertAdam):

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 5e-5,
        warmup: float = 0.1,
        t_total: int = 50000,
        schedule: str = 'warmup_linear',
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            warmup=warmup,
            t_total=t_total,
            schedule=schedule,
        )


if __name__ == '__main__':
    pass
