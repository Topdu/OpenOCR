import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import torch

from tools.engine import trainer as trainer_module


class _ScalarModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, image, data=None):
        return self.weight * image


class _Loss:

    def __call__(self, prediction, batch):
        return {"loss": prediction.sum()}


class _FakeScaler:

    def __init__(self):
        self.step_count = 0

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()
        self.step_count += 1

    def update(self):
        pass


class _Logger:

    def info(self, message):
        pass


def _install_detector_builders(monkeypatch):
    builders = {
        "opendet.losses": ("build_loss", lambda config: _Loss()),
        "opendet.metrics": (
            "build_metric",
            lambda config: SimpleNamespace(main_indicator="metric"),
        ),
        "opendet.modeling": ("build_model", lambda config: _ScalarModel()),
        "opendet.postprocess": (
            "build_post_process",
            lambda config, global_config: object(),
        ),
    }
    for module_name, (builder_name, builder) in builders.items():
        module = ModuleType(module_name)
        setattr(module, builder_name, builder)
        monkeypatch.setitem(sys.modules, module_name, module)


def test_detector_amp_training_uses_standard_model_path(monkeypatch):
    """Detector AMP should not read the recognizer-only transformer setting."""
    _install_detector_builders(monkeypatch)

    trainer = trainer_module.Trainer.__new__(trainer_module.Trainer)
    trainer.cfg = {
        "Global": {
            "cal_metric_during_train": False,
            "log_smooth_window": 1,
            "epoch_num": 1,
            "print_batch_step": 10,
            "eval_epoch_step": 1,
            "eval_batch_step": [0, 1],
            "save_epoch_step": [10, 1],
            "save_iter_step": [10, 1],
        },
        "Architecture": {},
        "PostProcess": {},
        "Loss": {},
        "Metric": {},
        "Train": {},
    }
    trainer._init_det_model()

    trainer.task = "det"
    trainer.device = torch.device("cpu")
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lr_lambda=lambda _: 1.0
    )
    trainer.train_dataloader = [[torch.ones(1)]]
    trainer.valid_dataloader = None
    trainer.scaler = _FakeScaler()
    trainer.accumulation_steps = 1
    trainer.grad_clip_val = 0
    trainer.status = {"epoch": 1, "global_step": 0, "metrics": {}}
    trainer.logger = _Logger()
    trainer.writer = None

    monkeypatch.setattr(trainer_module, "save_ckpt", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.amp, "autocast", lambda **kwargs: nullcontext())

    trainer.train()

    assert trainer.use_transformers is False
    assert trainer.scaler.step_count == 1
    assert trainer.model.weight.grad is None
