import torch
from torch import nn
from torch.nn import functional as F

from policy_search.util.core.popart import PopArtLayer


class ResNet(nn.Module):
    def __init__(
        self,
        observation_shape,  # not used in this architecture
        num_actions,
        num_tasks=1,
        use_lstm=False,
        use_popart=False,
        reward_clipping="abs_one",
        **kwargs
    ):

        super(ResNet, self).__init__()
        self.num_actions = num_actions
        self.num_tasks = num_tasks
        self.use_lstm = use_lstm
        self.use_popart = use_popart
        self.reward_clipping = reward_clipping

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 4
        for num_ch in [16, 32, 32]:
            feats_convs = [
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = [
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ]
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(3872, 256)

        # FC output size + last reward.
        core_output_size = self.fc.out_features + 1

        if use_lstm:
            self.core = nn.LSTM(core_output_size, 256, num_layers=1)
            core_output_size = 256

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = PopArtLayer(
            core_output_size, num_tasks if self.use_popart else 1
        )

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=(), run_to_conv=-1):
        if run_to_conv >= 0:
            x = inputs
        else:
            x = inputs["frame"]

        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        conv_counter = 0
        for i, f_conv in enumerate(self.feat_convs):
            x = f_conv(x)
            conv_counter += 1
            if 0 <= run_to_conv < conv_counter:
                return x

            res_input = x
            x = self.resnet1[i](x)
            conv_counter += 2
            if 0 <= run_to_conv < conv_counter:
                return x
            x += res_input

            res_input = x
            x = self.resnet2[i](x)
            conv_counter += 2
            if 0 <= run_to_conv < conv_counter:
                return x
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = None
        if self.reward_clipping == "abs_one":
            clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        elif self.reward_clipping == "none":
            clipped_reward = inputs["reward"].view(T * B, 1)

        core_input = torch.cat([x, clipped_reward], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            not_done = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), not_done.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                # core_state = nest.map(nd.mul, core_state)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input

        policy_logits = self.policy(core_output)
        baseline, normalized_baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)

        baseline = baseline.view(T, B, self.num_tasks)
        normalized_baseline = normalized_baseline.view(T, B, self.num_tasks)
        action = action.view(T, B, 1)

        return (
            dict(
                policy_logits=policy_logits,
                baseline=baseline,
                action=action,
                normalized_baseline=normalized_baseline,
            ),
            core_state,
        )
