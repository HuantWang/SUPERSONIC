import os
import tempfile
import itertools as it
import tabulate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def uniquify(path, sep=''):
    def name_sequence():
        count = it.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s = sep, n = next(count))
    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir = dirname, prefix = filename, suffix = ext)
        tempfile._name_sequence = orig
    return filename


def plot_grad_flow(named_parameters, flags):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    fig = plt.figure(figsize=(5, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.9, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()

    path = os.path.expandvars(os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "gradients.png")))
    plt.savefig(uniquify(path))


class GradientTracker:

    def __init__(self):
        self.avg_grad = {}
        self.max_grad = {}

        self.learning_step_count = 0

    def process_backward_pass(self, named_parameters, verbose=False):
        current_grad = []
        for n, p in named_parameters:
            if p.requires_grad and "bias" not in n:
                current_grad.append([n])

                if n not in self.avg_grad:
                    self.avg_grad[n] = []
                if n not in self.max_grad:
                    self.max_grad[n] = []

                if p.grad is None and verbose:
                    print("Layer '{}' has gradient None!".format(n))

                self.avg_grad[n].append(p.grad.abs().mean())
                self.max_grad[n].append(p.grad.abs().max())

                current_grad[-1].append(self.avg_grad[n][-1])
                current_grad[-1].append(p.grad.abs().std())
                current_grad[-1].append(self.max_grad[n][-1])

        if verbose:
            print("\nCurrent gradients at learning step {:d}:".format(self.learning_step_count))
            print(tabulate.tabulate(current_grad, headers=["layer", "mean", "std", "max"], tablefmt="presto"), "\n")

        self.learning_step_count += 1

    def print_total(self):
        grad = []
        for n in self.avg_grad:
            grad.append([n])
            grad[-1].append(np.mean(self.avg_grad[n]))
            grad[-1].append(np.max(self.max_grad[n]))

        print("\nTotal gradients at learning step {:d}:".format(self.learning_step_count))
        print(tabulate.tabulate(grad, headers=["layer", "mean", "max"], tablefmt="presto"), "\n")