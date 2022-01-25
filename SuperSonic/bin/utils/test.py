import gym
import compiler_gym                      # imports the CompilerGym environments


if __name__ == '__main__':
    env = gym.make("Stoke-v0")
    env.reset()
    env.step(env.action_space.sample())
    print(env.state)
    env.step(env.action_space.sample())
    print(env.state)

# env = gym.make(                          # creates a new environment
#      "llvm-v0",                           # selects the compiler to use
#      benchmark="cbench-v1/qsort",         # selects the program to compile
#      observation_space="Autophase",       # selects the observation space
#      reward_space="IrInstructionCountOz", # selects the optimization target
# )
# env.reset()                              # starts a new compilation session
# env.render()                             # prints the IR of the program
# env.step(env.action_space.sample())      # applies a random optimization, updates state/reward/actions
