"""This module defines how to search policy strategy"""

import argparse
import datetime
import SuperSonic.policy_definition.policy_define as policy_define
import policy_search.util.policy_search as policy_search
from SuperSonic.utils.engine.tasks_engine import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "--xpid", default="MultiTaskPopArt", help="Experiment id (default: None)."
)

parser.add_argument(
    "--task", default="Stoke", help="The task you want to optimize"
)

# Training settings.
parser.add_argument(
    "--disable_checkpoint", action="store_true", help="Disable saving checkpoint."
)
parser.add_argument(
    "--savedir",
    default="~/logs/supersonic",
    help="Root dir where experiment data will be saved.",
)
parser.add_argument(
    "--num_actors",
    default=1,
    type=int,
    metavar="N",
    help="Number of actors per environment (default: 4).",
)

parser.add_argument(
    "--batch_size", default=1, type=int, metavar="B", help="Learner batch size."
)  # steps
parser.add_argument(
    "--unroll_length",
    default=1,
    type=int,
    metavar="T",
    help="The unroll length (time dimension).",
)  # steps
parser.add_argument(
    "--num_learner_threads",
    "--num_threads",
    default=1,
    type=int,
    metavar="N",
    help="Number learner threads.",
)  # steps
parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA.")
parser.add_argument(
    "--num_actions", default=6, type=int, metavar="A", help="Number of actions."
)
parser.add_argument("--use_lstm", action="store_true", help="Use LSTM in agent model.")
parser.add_argument(
    "--agent_type",
    type=str,
    default="resnet",
    help="The type of network to use for the agent.",
)
parser.add_argument(
    "--frame_height", type=int, default=84, help="Height to which frames are rescaled."
)
parser.add_argument(
    "--frame_width", type=int, default=84, help="Width to which frames are rescaled."
)
parser.add_argument(
    "--aaa_input_format",
    type=str,
    default="gray_stack",
    choices=["gray_stack", "rgb_last", "rgb_stack"],
    help="Color format of the frames as input for the AAA.",
)
parser.add_argument("--use_popart", action="store_true", help="Use PopArt Layer.")

# Loss settings.
parser.add_argument(
    "--entropy_cost", default=0.0006, type=float, help="Entropy cost/multiplier."
)
parser.add_argument(
    "--baseline_cost", default=0.5, type=float, help="Baseline cost/multiplier."
)
parser.add_argument(
    "--discounting", default=0.99, type=float, help="Discounting factor."
)
parser.add_argument(
    "--reward_clipping",
    default="abs_one",
    choices=["abs_one", "none"],
    help="Reward clipping.",
)

# Optimizer settings.
parser.add_argument(
    "--learning_rate", default=0.00048, type=float, metavar="LR", help="Learning rate."
)
parser.add_argument(
    "--alpha", default=0.99, type=float, help="RMSProp smoothing constant."
)
parser.add_argument("--momentum", default=0, type=float, help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float, help="RMSProp epsilon.")
parser.add_argument(
    "--grad_norm_clipping", default=40.0, type=float, help="Global gradient norm clip."
)

# Misc settings.
parser.add_argument(
    "--write_profiler_trace",
    action="store_true",
    help="Collect and write a profiler trace " "for chrome://tracing/.",
)
parser.add_argument(
    "--save_model_every_nsteps", default=0, type=int, help="Save model every n steps"
)
parser.add_argument("--beta", default=0.0001, type=float, help="PopArt parameter")

# Test settings.
parser.add_argument(
    "--num_episodes", default=100, type=int, help="Number of episodes for Testing."
)
parser.add_argument("--actions", help="Use given action sequence.")


# yapf: enable
parser.add_argument("--num_buffers", type=int, default="32", help="num_buffers")

parser.add_argument(
    "--env", type=str, default="BanditCSREnv-v0", help="Task environments"
)
parser.add_argument("--steps", type=int, default=100000, help="Number of steps")
parser.add_argument(
    "--total_steps",
    default=50,
    type=int,
    metavar="T",
    help="Total environment steps to train for.",
)
parser.add_argument("--Policy", help="Policy")
parser.add_argument("--Dataset", help="Dataset")

class PolSearch_main:
    """:class:
    This is a inferface to call start_engine to search policy.

        """

    def __init__(self,flags):
        self.flags=flags

    def start_engine(self):
        """Calling to policy_search() invokes SuperSonic meta-optimizer, where the developer can also limit the number of trials spent on client RL searching.
          using policy_define.SuperOptimizer() to define policy strategies and split dataset.
          using policy_search.train() to search best policy for objective task.
          finally, using CSR(bestpolicy).main() to deploy best policy strategy and optimize the program."""

        cwd = os.getcwd()
        directory = os.path.join(
            cwd, "logs", datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )
        os.makedirs(directory)

        self.flags.Policy = policy_define.SuperOptimizer(
            StateFunctions=["Word2vec", "Doc2vec", "Bert", "Actionhistory"],
            RewardFunctions=["weight"],
            RLAlgorithms=["MCTS", "PPO", "DQN", "QLearning"],
            ActionFunctions=["init"],
        ).PolicyDefined()

        self.flags.Dataset = policy_define.SuperOptimizer(
            datapath="../../tasks/stoke/example/hacker"
        ).cross_valid()

        arguments = (
            f"--env {self.flags.env} "
            f"--savedir {directory} "
            f"--total_steps {self.flags.steps} "
            "--batch_size 32 "
        )

        bestpolicy = policy_search.train(self.flags)
        print("The best policy strategy is", bestpolicy)

        #TODO:tune parameters

        print("starting to run the best policy for your task")

        TaskEngine(bestpolicy,tasks_name='Stoke').run()


if __name__ == "__main__":
    flags = parser.parse_args()
    PolSearch_main(flags).start_engine()
