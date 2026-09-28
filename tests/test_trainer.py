from types import SimpleNamespace

import torch

from tools.engine import trainer as trainer_module


class _ScalarModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, image, data=None):
        return self.weight * image


class _RecordingSGD(torch.optim.SGD):

    def __init__(self, params):
        super().__init__(params, lr=0.0)
        self.step_gradients = []

    def step(self, closure=None):
        self.step_gradients.append(
            [
                parameter.grad.detach().clone()
                for group in self.param_groups
                for parameter in group['params']
            ])
        return super().step(closure)


class _Logger:

    def info(self, message):
        pass


class _Loss:

    def __call__(self, prediction, batch):
        return {'loss': prediction.sum()}


def test_non_amp_training_clears_gradients_between_optimizer_steps(monkeypatch):
    """Each non-AMP batch should backpropagate only its own gradient."""
    model = _ScalarModel()
    optimizer = _RecordingSGD(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                   lr_lambda=lambda _: 1.0)

    trainer = trainer_module.Trainer.__new__(trainer_module.Trainer)
    trainer.cfg = {
        'Global': {
            'cal_metric_during_train': False,
            'log_smooth_window': 1,
            'epoch_num': 1,
            'print_batch_step': 10,
            'eval_epoch_step': 1,
            'eval_batch_step': [0, 1],
            'save_epoch_step': [10, 1],
            'save_iter_step': [10, 1],
        },
        'Train': {},
    }
    trainer.task = 'det'
    trainer.device = torch.device('cpu')
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.loss_class = _Loss()
    trainer.train_dataloader = [[torch.ones(1)], [torch.ones(1)]]
    trainer.valid_dataloader = None
    trainer.scaler = None
    trainer.accumulation_steps = 1
    trainer.grad_clip_val = 0
    trainer.status = {'epoch': 1, 'global_step': 0, 'metrics': {}}
    trainer.eval_class = SimpleNamespace(main_indicator='metric')
    trainer.logger = _Logger()
    trainer.writer = None
    trainer.use_transformers = False

    monkeypatch.setattr(trainer_module, 'save_ckpt', lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)

    trainer.train()

    assert len(optimizer.step_gradients) == 2
    torch.testing.assert_close(optimizer.step_gradients[0][0], torch.tensor(1.0))
    torch.testing.assert_close(optimizer.step_gradients[1][0], torch.tensor(1.0))
    assert model.weight.grad is None
