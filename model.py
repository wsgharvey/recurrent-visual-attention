import math

import torch
import torch.nn as nn

from torch.distributions import Normal

from modules import baseline_network
from modules import glimpse_network, core_network
from modules import action_network, discrete_location_network, location_network
from attention_target_dataset import normalize_attention_loc


class RecurrentAttention(nn.Module):
    """
    A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References
    ----------
    - Minh et. al., https://arxiv.org/abs/1406.6247
    """
    def __init__(self,
                 g,
                 k,
                 s,
                 c,
                 h_g,
                 h_l,
                 std,
                 constrain_mu,
                 hidden_size,
                 num_classes):
        """
        Initialize the recurrent attention model and its
        different components.

        Args
        ----
        - g: size of the square patches in the glimpses extracted
          by the retina.
        - k: number of patches to extract per glimpse.
        - s: scaling factor that controls the size of successive patches.
        - c: number of channels in each image.
        - h_g: hidden layer size of the fc layer for `phi`.
        - h_l: hidden layer size of the fc layer for `l`.
        - std: standard deviation of the Gaussian policy.
        - hidden_size: hidden size of the rnn.
        - num_classes: number of classes in the dataset.
        - num_glimpses: number of glimpses to take per image,
          i.e. number of BPTT steps.
        """
        super(RecurrentAttention, self).__init__()
        self.std = std

        self.sensor = glimpse_network(h_g, h_l, g, k, s, c)
        self.rnn = core_network(hidden_size, hidden_size)
        self.locator = discrete_location_network(hidden_size, 784)
        self.classifier = action_network(hidden_size, num_classes)
        self.baseliner = baseline_network(hidden_size, 1)

    def forward(self, x, h_t_prev, last=False, replace_l_t=None, new_l_t=None):
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l_t_prev: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the previous
          timestep `t-1`.
        - h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the previous timestep `t-1`.
        - last: a bool indicating whether this is the last timestep.
          If True, the action network returns an output probability
          vector over the classes and the baseline `b_t` for the
          current timestep `t`. Else, the core network returns the
          hidden state vector for the next timestep `t+1` and the
          location vector for the next timestep `t+1`.

        Returns
        -------
        - h_t: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`.
        - mu: a 2D tensor of shape (B, 2). The mean that parametrizes
          the Gaussian policy.
        - l_t: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the
          current timestep `t`.
        - b_t: a vector of length (B,). The baseline for the
          current time step `t`.
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the classes.
        - log_pi: a vector of length (B,).
        """
        unnormed_l_t, loc_dist = self.locator(h_t_prev)

        # replace l_t_s
        if replace_l_t is not None:
            unnormed_l_t = torch.mul(unnormed_l_t, 1-replace_l_t)
            unnormed_l_t = unnormed_l_t + torch.mul(new_l_t, replace_l_t)

        l_t = normalize_attention_loc(unnormed_l_t)
        g_t = self.sensor(x, l_t)
        h_t = self.rnn(g_t, h_t_prev)
        b_t = self.baseliner(h_t).squeeze()

        if last:
            log_probas = self.classifier(h_t)
            return h_t, unnormed_l_t, b_t, log_probas, loc_dist

        return h_t, unnormed_l_t, b_t, loc_dist
