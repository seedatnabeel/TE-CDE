# Adapted from: https://github.com/patrick-kidger/torchcde
import logging

import torch
import torch.nn as nn
import torchcde
from torch.nn import functional as F


######################
# A CDE model is defined as
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(
            hidden_channels,
            128,
        )  # torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(
            128,
            input_channels * hidden_channels,
        )  # torch.nn.Linear(hidden_channels, input_channels * hidden_channels)

        self.W = torch.nn.Parameter(torch.Tensor(input_channels))
        self.W.data.fill_(1)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)

        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        # z = torch.matmul(z,torch.diag(self.W))

        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(
        self,
        input_channels_x,
        hidden_channels_x,
        output_channels,
        interpolation="linear",
    ):
        super(NeuralCDE, self).__init__()

        self.embed_x = torch.nn.Linear(input_channels_x, hidden_channels_x)

        self.cde_func = CDEFunc(input_channels_x, hidden_channels_x)
        self.combine_z = torch.nn.Linear(
            (hidden_channels_x + hidden_channels_x) // 2,
            hidden_channels_x * hidden_channels_x,
        )
        self.outcome = torch.nn.Linear(hidden_channels_x, output_channels)
        self.treatment = torch.nn.Linear(hidden_channels_x, 4)
        self.softmax = torch.nn.Softmax(dim=4)
        self.interpolation = interpolation

        self.dropout_layer = torch.nn.Dropout(0.1)

        logging.info(f"Interpolation type: {self.interpolation}")

    def forward(self, coeffs_x, device, mcd=True):
        if self.interpolation == "cubic":
            x = torchcde.NaturalCubicSpline(coeffs_x)
        elif self.interpolation == "linear":
            x = torchcde.LinearInterpolation(coeffs_x)
        else:
            raise ValueError(
                "Only 'linear' and 'cubic' interpolation methods are implemented.",
            )

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        z_x = torch.tensor(
            self.embed_x(x.evaluate(x.interval[0])),
            dtype=torch.float,
            device=device,
        )

        ######################
        # Solve the CDE.
        ######################
        # t =x.interval adds the time tracking component to the CDE
        z_hat = torch.tensor(
            torchcde.cdeint(
                X=x,
                z0=z_x,
                func=self.cde_func,
                t=x.interval,
                backend="torchdiffeq",
                method="dopri5",
                options=dict(jump_t=x.grid_points),
            ),
            dtype=torch.float,
            device=device,
        )

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################

        z_hat = z_hat[:, 1]

        if mcd == True:
            z_hat = self.dropout_layer(z_hat)

        pred_y = self.outcome(z_hat) * 1150

        pred_a = self.treatment(z_hat)

        pred_a_softmax = F.log_softmax(pred_a, dim=1)

        return pred_y, pred_a_softmax, pred_a, z_hat
