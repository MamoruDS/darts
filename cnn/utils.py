import os
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable

from typing import Union
Number = Union[float, int]


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val: Number, n: int = 1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)) -> list[torch.Tensor]:
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length: int):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(cutout: bool = False, cutout_length: int = 16):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    trans = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
    if cutout:
        trans.append(Cutout(cutout_length))
    train_transform = transforms.Compose(trans)

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def count_parameters_in_MB(model) -> Union[np.ndarray, Number]:
    return np.sum(
        np.prod(v.size())
        for name, v in model.named_parameters()
        if "auxiliary" not in name
    ) / 1e6


def save_checkpoint(state, is_best, save) -> None:
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model: nn.Module, model_path: str) -> None:
    torch.save(model.state_dict(), model_path)


def load(model: nn.Module, model_path: str) -> None:
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1,
                                 1).bernoulli_(keep_prob).cuda()
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
