from catalyst.dl import Callback, RunnerState, CallbackOrder, CriterionCallback

import torch
from torch.nn import Module


class MyDebugCallback(Callback):
    def __init__(self):
        super(MyDebugCallback, self).__init__(order=CallbackOrder.Metric + 1)

    def on_epoch_end(self, state: RunnerState) -> None:
        print('Input:')
        print(state.input.keys())
        print('Output')
        print(state.output.keys())
        print('-' * 40)


class MyDebugCriterion(Module):
    def __init__(self):
        super(MyDebugCriterion, self).__init__()

    def forward(self, *args, **kwargs):
        print('Args:')
        print(', '.join(list(map(str, map(type, args)))))
        print('Kwargs:')
        print(', '.join(kwargs.keys()))
        print('*' * 40)
        return torch.zeros((1, ), dtype=torch.float32)
