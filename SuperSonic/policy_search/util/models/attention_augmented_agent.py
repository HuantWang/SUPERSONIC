"""
Adapted from https://github.com/cjlovering/Towards-Interpretable-Reinforcement-Learning-Using-Attention-Augmented-Agents-Replication
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from policy_search.util.core.popart import PopArtLayer


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        """From the original implementation:
        Paper
        -----
        https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf

        Referenced code
        ---------------
        https://github.com/automan000/Convolution_LSTM_PyTorch/blob/master/convolution_lstm.py
        """
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whi = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxf = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whf = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxc = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whc = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxo = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Who = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def initial_state(self, batch_size, hidden, height, width):
        return self.init_hidden(batch_size, hidden, height, width)

    def init_hidden(self, batch_size, hidden, height, width):
        if self.Wci is None:
            self.Wci = torch.zeros(1, hidden, height, width, requires_grad=True)
            self.Wcf = torch.zeros(1, hidden, height, width, requires_grad=True)
            self.Wco = torch.zeros(1, hidden, height, width, requires_grad=True)
        return (
            torch.zeros(batch_size, hidden, height, width),
            torch.zeros(batch_size, hidden, height, width),
        )

    def forward(self, x, prev_hidden=()):
        if self.Wci is None:
            _, _, height, width = x.shape
            hidden = self.hidden_channels
            self.Wci = torch.zeros(1, hidden, height, width, requires_grad=True).to(
                x.device
            )
            self.Wcf = torch.zeros(1, hidden, height, width, requires_grad=True).to(
                x.device
            )
            self.Wco = torch.zeros(1, hidden, height, width, requires_grad=True).to(
                x.device
            )

        h, c = prev_hidden

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)

        return ch, cc


class VisionNetwork(nn.Module):
    def __init__(self, frame_height, frame_width, in_channels=3, hidden_channels=128):
        super(VisionNetwork, self).__init__()
        self._frame_height = frame_height
        self._frame_width = frame_width
        self._in_channels = in_channels
        self._hidden_channels = hidden_channels

        # padding s.t. the output shapes match the paper.
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._in_channels,
                out_channels=32,
                kernel_size=(8, 8),
                stride=4,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding=2
            ),
        )
        self.vision_lstm = ConvLSTMCell(
            input_channels=64, hidden_channels=self._hidden_channels, kernel_size=3
        )

    def initial_state(self, batch_size, dummy_frame):
        cnn_output = self.vision_cnn(dummy_frame)
        height, width = tuple(cnn_output.shape[2:])
        return self.vision_lstm.initial_state(
            batch_size, self._hidden_channels, height, width
        )

    def forward(self, x, prev_vision_core_state):
        x = x.permute(0, 3, 1, 2)
        vision_core_output, vision_core_state = self.vision_lstm(
            self.vision_cnn(x), prev_vision_core_state
        )
        return (
            vision_core_output.permute(0, 2, 3, 1),
            (vision_core_output, vision_core_state),
        )


class QueryNetwork(nn.Module):
    def __init__(self, num_queries, c_k, c_s):
        super(QueryNetwork, self,).__init__()
        self._num_queries = num_queries
        self._c_o = c_k + c_s
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self._num_queries * self._c_o),
            nn.ReLU(),
            nn.Linear(self._num_queries * self._c_o, self._num_queries * self._c_o),
        )

    def forward(self, query):
        out = self.model(query)
        return out.reshape(-1, self._num_queries, self._c_o)


class SpatialBasis:
    def __init__(self, height=27, width=20, channels=64):
        self._height = height
        self._width = width
        self._channels = channels
        self._s = None

        self.init()

    def __call__(self, x):
        batch_size, x_height, x_width, *_ = x.size()
        re_init = False
        if self._height != x_height:
            self._height = x_height
            re_init = True
        if self._width != x_width:
            self._width = x_width
            re_init = True
        if re_init:
            self.init()

        # Stack the spatial bias (for each batch) and concat to the input.
        s = torch.stack([self._s] * batch_size).to(x.device)
        return torch.cat([x, s], dim=3)

    def init(self):
        h, w, d = self._height, self._width, self._channels

        p_h = torch.mul(
            torch.arange(1, h + 1).unsqueeze(1).float(), torch.ones(1, w).float()
        ) * (np.pi / h)
        p_w = torch.mul(
            torch.ones(h, 1).float(), torch.arange(1, w + 1).unsqueeze(0).float()
        ) * (np.pi / w)

        # NOTE: I didn't quite see how U,V = 4 made sense given that the authors form the spatial
        # basis by taking the outer product of the values. Still, I think what I have is aligned with what
        # they did, but I am less confident in this step.
        U = V = 8  # size of U, V.
        u_basis = v_basis = torch.arange(1, U + 1).unsqueeze(0).float()
        a = torch.mul(p_h.unsqueeze(2), u_basis)
        b = torch.mul(p_w.unsqueeze(2), v_basis)
        out = torch.einsum("hwu,hwv->hwuv", torch.cos(a), torch.cos(b)).reshape(h, w, d)
        self._s = out


def spatial_softmax(A):
    # A: batch_size x h x w x d
    b, h, w, d = A.size()
    # Flatten A s.t. softmax is applied to each grid (not over queries)
    A = A.reshape(b, h * w, d)
    A = F.softmax(A, dim=1)
    # Reshape A to original shape.
    A = A.reshape(b, h, w, d)
    return A


def apply_alpha(A, V):
    b, h, w, c = A.size()
    A = A.reshape(b, h * w, c).transpose(1, 2)

    _, _, _, d = V.size()
    V = V.reshape(b, h * w, d)

    return torch.matmul(A, V)


class AttentionAugmentedAgent(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_actions,
        hidden_size: int = 256,
        c_v: int = 120,
        c_k: int = 8,
        c_s: int = 64,
        num_queries: int = 4,
        rgb_last: bool = False,
        num_tasks: int = 1,
        use_popart: bool = False,
        **kwargs
    ):
        super(AttentionAugmentedAgent, self).__init__()
        self.hidden_size = hidden_size
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.rgb_last = rgb_last
        self.num_tasks = num_tasks
        self.use_popart = use_popart
        self.c_v, self.c_k, self.c_s, self.num_queries = c_v, c_k, c_s, num_queries
        if self.rgb_last:
            self.observation_shape = (3,) + tuple(self.observation_shape[1:])

        self.config = {
            "observation_shape": observation_shape,
            "num_actions": num_actions,
            "hidden_size": hidden_size,
            "c_v": c_v,
            "c_k": c_k,
            "c_s": c_s,
            "num_queries": num_queries,
            "rgb_last": rgb_last,
        }
        self.config.update(kwargs)

        self.vision = VisionNetwork(
            self.observation_shape[1],
            self.observation_shape[2],
            in_channels=self.observation_shape[0],
        )
        self.query = QueryNetwork(num_queries, c_k, c_s)
        self.spatial = SpatialBasis()

        self.answer_processor = nn.Sequential(
            # 1031 x 512
            nn.Linear(
                (c_v + c_s) * num_queries + (c_k + c_s) * num_queries + 1 + 1, 512
            ),
            nn.ReLU(),
            nn.Linear(512, hidden_size),
        )

        self.policy_core = nn.LSTM(hidden_size, hidden_size)

        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.baseline_head = PopArtLayer(
            hidden_size, num_tasks if self.use_popart else 1
        )

    def initial_state(self, batch_size):
        with torch.no_grad():
            dummy_frame = torch.zeros(1, *self.observation_shape)
            vision_core_initial_state = tuple(
                s.unsqueeze(0)
                for s in self.vision.initial_state(batch_size, dummy_frame)
            )
        # unsqueeze() here as well as in forward() is necessary because some of the code in monobeast.py assumes that
        # the first dimension of the returned state tensors are layers of the RNN, so we need this "dummy dimension"

        policy_core_initial_state = tuple(
            torch.zeros(
                self.policy_core.num_layers, batch_size, self.policy_core.hidden_size
            )
            for _ in range(2)
        )
        return vision_core_initial_state + policy_core_initial_state

    def forward(self, inputs, state=(), return_attention_maps=False):
        # input frames are formatted: (time_steps, batch_size, frame_stack, height, width)
        # the original network is designed for (batch_size, height, width, num_channels)
        # there are a couple options to solve this:
        # - use grayscale, stack frames, use those as channels
        # - use full colour, stack frames, resulting in 4 * 3 = 12 channels
        # - use full colour, don't stack frames (similar to original paper)
        # IMPORTANT NOTE: for the latter, the original paper still sends the same action 4 times,
        # so the following might be a better option (as far as implementation goes)
        # - use full colour, stack frames, use only the last one
        # => for now, I'm just going to use the first method

        # (time_steps, batch_size, frame_stack, height, width)
        x: torch.Tensor = inputs["frame"]
        time_steps, batch_size, *_ = x.shape
        # (time_steps, batch_size, frame_stack, height, width)
        x = x.float() / 255.0
        # (time_steps, batch_size, height, width, frame_stack) to match the design of the network
        x = x.permute(0, 1, 3, 4, 2)
        # frames are RGB and only the first should be used
        if self.rgb_last:
            x = x[:, :, :, :, -3:]

        # (time_steps, batch_size, 1)
        prev_reward = inputs["reward"].view(time_steps, batch_size, 1)
        # (time_steps, batch_size, num_actions)
        # prev_action = F.one_hot(inputs["last_action"].view(time_steps, batch_size), self.num_actions).float()
        prev_action = inputs["last_action"].view(time_steps, batch_size, 1)
        # (time_steps, batch_size)
        not_done = (~inputs["done"]).float()

        vision_core_output_list = []
        vision_core_state = tuple(
            s.squeeze(0) for s in state[:2]
        )  # see comment in initial_state()
        for x_batch, not_done_batch in zip(x.unbind(), not_done.unbind()):

            # (batch_size, 1, 1, 1) => expanded to be broadcastable for the multiplication
            not_done_batch = not_done_batch.view(-1, 1, 1, 1)
            # (batch_size, c_k + c_v, height_ac=height_after_cnn, width_ac=width_after_cnn) * 2
            vision_core_state = tuple(not_done_batch * s for s in vision_core_state)

            # 1 (a). Vision.
            # --------------
            # (batch_size, height_ac, width_ac, c_k + c_v)
            vision_core_output, vision_core_state = self.vision(
                x_batch, vision_core_state
            )
            vision_core_output_list.append(vision_core_output)
            # for clarity vision_core_output.unsqueeze(0) might be better, because it would be clear that this
            # is the result for one time step, but since we merge time and batch in the following steps anyway,
            # we can also just "discard" the time dimension and get the same result when we concatenate
            # the results for each time step

        vision_core_state = tuple(
            s.unsqueeze(0) for s in vision_core_state
        )  # see comment in initial_state()

        # (time_steps * batch_size, height_ac, width_ac, c_k + c_v)
        vision_core_output = torch.cat(vision_core_output_list)

        # (batch_size, height_ac, width_ac, c_k), (batch_size, height_ac, width_ac, c_v)
        keys, values = vision_core_output.split([self.c_k, self.c_v], dim=3)
        # (batch_size, height_ac, width_ac, c_k + c_s), (batch_size, height_ac, width_ac, c_v + c_s)
        keys, values = self.spatial(keys), self.spatial(values)

        # reshape the keys and values tensors so that they can be separated in the time dimension
        # (time_steps, batch_size, height_ac, width_ac, c_k + c_s)
        keys = keys.view(time_steps, batch_size, *keys.shape[1:])
        # (time_steps, batch_size, height_ac, width_ac, c_v + c_s)
        values = values.view(time_steps, batch_size, *values.shape[1:])

        policy_core_output_list = []
        attention_map_list = []
        policy_core_state = state[2:]
        for (
            keys_batch,
            values_batch,
            prev_reward_batch,
            prev_action_batch,
            not_done_batch,
        ) in zip(
            keys.unbind(),
            values.unbind(),
            prev_reward.unbind(),
            prev_action.unbind(),
            not_done.unbind(),
        ):

            # (1, batch_size, 1)
            not_done_batch = not_done_batch.view(1, -1, 1)
            # (lstm_layers, batch_size, hidden_size) * 2
            policy_core_state = tuple(not_done_batch * s for s in policy_core_state)

            # 1 (b). Queries.
            # --------------
            # (batch_size, num_queries, c_k + c_s)
            queries = self.query(policy_core_state[0])

            # 2. Answer.
            # ----------
            # (batch_size, height_ac, width_ac, num_queries)
            answer = torch.matmul(keys_batch, queries.transpose(2, 1).unsqueeze(1))
            # (batch_size, height_ac, width_ac, num_queries)
            answer = spatial_softmax(answer)
            attention_map_list.append(answer)
            # (batch_size, num_queries, c_v + c_s)
            answer = apply_alpha(answer, values_batch)

            # (batch_size, (c_v + c_s) * num_queries + (c_k + c_s) * num_queries + 1 + num_actions)
            answer = torch.cat(
                torch.chunk(answer, self.num_queries, dim=1)
                + torch.chunk(queries, self.num_queries, dim=1)
                + (
                    prev_reward_batch.unsqueeze(1).float(),
                    prev_action_batch.unsqueeze(1).float(),
                ),
                dim=2,
            ).squeeze(1)
            # (batch_size, hidden_size)
            answer = self.answer_processor(answer)

            # 3. Policy.
            # ----------
            # (batch_size, hidden_size)
            policy_core_output, policy_core_state = self.policy_core(
                answer.unsqueeze(0), policy_core_state
            )
            policy_core_output_list.append(policy_core_output.squeeze(0))
            # squeeze() is needed because the LSTM input has an "extra" dimensions for the layers of the LSTM,
            # of which there is only one in this case; therefore, the concatenated input vector has an extra
            # dimension and the output as well

        # (time_steps * batch_size, hidden_size)
        output = torch.cat(policy_core_output_list)
        attention_maps = torch.cat(attention_map_list)

        # 4, 5. Outputs.
        # --------------
        # (time_steps * batch_size, num_actions)
        policy_logits = self.policy_head(output)
        # (time_steps * batch_size, num_tasks)
        baseline, normalized_baseline = self.baseline_head(output)

        # (time_steps * batch_size, 1)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        # (time_steps, batch_size, num_actions)
        policy_logits = policy_logits.view(time_steps, batch_size, self.num_actions)
        # (time_steps, batch_size, num_tasks)
        baseline = baseline.view(time_steps, batch_size, self.num_tasks)
        normalized_baseline = normalized_baseline.view(
            time_steps, batch_size, self.num_tasks
        )
        # (time_steps, batch_size, 1)
        action = action.view(time_steps, batch_size, 1)

        if return_attention_maps:
            return (
                dict(
                    policy_logits=policy_logits,
                    baseline=baseline,
                    action=action,
                    normalized_baseline=normalized_baseline,
                ),
                vision_core_state + policy_core_state,
                attention_maps,
            )

        return (
            dict(
                policy_logits=policy_logits,
                baseline=baseline,
                action=action,
                normalized_baseline=normalized_baseline,
            ),
            vision_core_state + policy_core_state,
        )
