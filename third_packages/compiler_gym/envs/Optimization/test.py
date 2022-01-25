import gym
from gym import spaces
from gym.utils import seeding
import grpc
from . import schedule_pb2
from . import schedule_pb2_grpc
import numpy as np
# from StringIO import StringIO
from io import StringIO


class HalideEnv(gym.Env):
    metadata = {'render.modes': ['ansi']}
    target = 'localhost:50051'



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _unique(self, action):
        count = self.action_count.get(action, 0) + 1
        return count == 1

