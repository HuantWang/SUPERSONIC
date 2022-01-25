import os
import argparse
import datetime
import compiler_gym.mdp_search.policy_define as policy_define
import PolSear_MTL.util.PolSear_engine as policy_search
from SuperSonic.bin.utils.engine.tasks_engine import *

parser = argparse.ArgumentParser()


# parser.add_argument("--mode", default="train",
#                     choices=["train", "test", "test_render"],
#                     help="Training or test mode.")
parser.add_argument("--xpid", default='MultiTaskPopArt',
                    help="Experiment id (default: None).")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=1, type=int, metavar="N",
                    help="Number of actors per environment (default: 4).")

parser.add_argument("--batch_size", default=1, type=int, metavar="B",
                    help="Learner batch size.") #steps
parser.add_argument("--unroll_length", default=1, type=int, metavar="T",
                    help="The unroll length (time dimension).")#steps
parser.add_argument("--num_learner_threads", "--num_threads", default=1, type=int,
                    metavar="N", help="Number learner threads.")#steps
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--num_actions", default=6, type=int, metavar="A",
                    help="Number of actions.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument("--agent_type", type=str, default="resnet",
                    help="The type of network to use for the agent.")
parser.add_argument("--frame_height", type=int, default=84,
                    help="Height to which frames are rescaled.")
parser.add_argument("--frame_width", type=int, default=84,
                    help="Width to which frames are rescaled.")
parser.add_argument("--aaa_input_format", type=str, default="gray_stack", choices=["gray_stack", "rgb_last", "rgb_stack"],
                    help="Color format of the frames as input for the AAA.")
parser.add_argument("--use_popart", action="store_true",
                    help="Use PopArt Layer.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.0006,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")

# Misc settings.
parser.add_argument("--write_profiler_trace", action="store_true",
                    help="Collect and write a profiler trace "
                    "for chrome://tracing/.")
parser.add_argument("--save_model_every_nsteps", default=0, type=int,
                    help="Save model every n steps")
parser.add_argument("--beta", default=0.0001, type=float,
                    help="PopArt parameter")

# Test settings.
parser.add_argument("--num_episodes", default=100, type=int,
                    help="Number of episodes for Testing.")
parser.add_argument("--actions",
                    help="Use given action sequence.")


# yapf: enable
parser.add_argument("--num_buffers", type=int, default="32",
                    help="num_buffers")
parser.add_argument("--env", type=str, default="BanditCSREnv-v0",
                    help="Gym environments")
parser.add_argument("--steps", type=int, default=100000, help="Number of steps")
parser.add_argument("--total_steps", default=50, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--Policy",
                    help="Policy")
parser.add_argument("--Dataset",
                    help="Dataset")



def main(flags):
    cwd = os.getcwd()
    directory = os.path.join(cwd, "logs", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(directory)

    flags.Policy = policy_define.SuperOptimizer(
        StateFunctions =["Word2vec","Doc2vec","Bert","Actionhistory"],
        RewardFunctions = ["weight"],
        RLAlgorithms = ["MCTS","PPO","DQN","QLearning"],
        ActionFunctions = ["init"],).PolicyDefined()

    flags.Dataset = policy_define.SuperOptimizer(
        datapath='../../tasks/CSR/DATA'
    ).cross_valid()


    arguments = (
        f'--env {flags.env} '
        f'--savedir {directory} '
        f'--total_steps {flags.steps} '
        '--batch_size 32 '
    )

    bestpolicy=policy_search.train(flags)
    print("The best policy is", bestpolicy)
    print("start run the best policy")

    # CSR(bestpolicy).main()
    # print(f'python -m torchbeast.monobeast {arguments}')




if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
