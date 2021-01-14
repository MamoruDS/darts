import torch
import numpy as np
import torch.autograd
import torch.optim
import torch.nn as nn

from model_search import Network


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    def __init__(self, model: Network, args):
        self.network_momentum: float = args.momentum
        self.network_weight_decay: float = args.weight_decay
        self.model = model
        print(
            f's:{type(self.model.arch_parameters)}; s():{type(self.model.arch_parameters())}; s()[0]:{type(self.model.arch_parameters()[0])}'
        )
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=args.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay
        )

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta: torch.Tensor = _concat(self.model.parameters()).data
        try:
            moment = _concat(
                network_optimizer.state[v]['momentum_buffer']
                for v in self.model.parameters()
            ).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(
            torch.autograd.grad(loss, list(self.model.parameters()))
        ).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(
            torch.sub(
                input=theta, other=(moment + dtheta), alpha=eta, out=theta
            )
        )
        return unrolled_model

    def step(
        self, input_train, target_train, input_valid, target_valid, eta,
        network_optimizer, unrolled
    ):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid, eta,
                network_optimizer
            )
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(
        self, input_train, target_train, input_valid, target_valid, eta,
        network_optimizer
    ):
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer
        )
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha: list[torch.Tensor
                    ] = [v.grad for v in unrolled_model.arch_parameters()]
        vector: list[torch.Tensor
                    ] = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(
            vector, input_train, target_train
        )

        for g, ig in zip(dalpha, implicit_grads):
            torch.sub(g.data, ig.data, alpha=eta, out=g.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset:offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(
        self, vector: list[torch.Tensor], input, target, r=1e-2
    ):
        R: torch.Tensor = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            torch.add(p.data, v, alpha=R.item(), out=p.data) # TBD
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            torch.sub(p.data, v, alpha=R.item(), out=p.data) # TBD
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            torch.add(p.data, v, alpha=R.item(), out=p.data) # TBD

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
