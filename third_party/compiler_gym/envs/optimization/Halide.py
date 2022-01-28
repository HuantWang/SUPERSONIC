"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.spaces import Discrete, Dict, Box
import random
from gensim.models.doc2vec import Doc2Vec

from SuperSonic.policy_definition.action import action_functions
from SuperSonic.policy_definition.observation import observation_function
from SuperSonic.policy_definition.reward import reward_function

# add by zc
import csv
import time
import os
import json

# import shutilobservation_function
import re
import grpc
from SuperSonic.service import schedule_pb2
from SuperSonic.service import schedule_pb2_grpc
from io import StringIO
import sqlite3

# add for logging
import logging

stuball = None
f = None


class HalideEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    def __init__(
        self,
        algorithm_id,
        input_image,
        max_stage_directive,
        log_path,
        state_function,
        action_function,
        reward_function,
    ):  # get
        """ Defines the reinforcement leaning environment.
        Modify to match different task shape.

        """

        channel = grpc.insecure_channel("localhost:50051")
        global stuball, f, csv_writer
        stuball = schedule_pb2_grpc.ScheduleServiceStub(channel)

        f = open(
            os.path.join(log_path, "result.csv"), "w+", encoding="utf-8", newline=""
        )
        csv_writer = csv.writer(f)
        csv_writer.writerow(["t", "best_exec", "best_schedule"])
        f.flush()

        request = schedule_pb2.ScheduleInitRequest(
            algorithm_id=algorithm_id,
            input_image=input_image,
            max_stage_directive=max_stage_directive,
        )

        response = stuball.init(request)

        assert response.max_stage > 0 and response.max_directive > 0
        assert response.schedule_map_range > 1 and response.init_time_sec > 0

        self.init_exec_time_sec = response.init_time_sec
        self.min_exec_time_sec = None
        self.best_exec_time_sec = response.init_time_sec
        self.max_stage = response.max_stage
        self.max_stage_directive = max_stage_directive
        self.action_count = dict()
        self.map_count = None
        self.np_random = None
        # find mdp interface
        self.state_function = state_function
        self.action_function = action_function
        self.reward_function = reward_function

        self.reward_scale = 100.0 / response.init_time_sec
        self.error_reward = -1.0
        self.timeout_error_reward = self.error_reward * response.max_stage
        self.noop_reward = 0.0

        self.obsv_low = 0
        self.obsv_high = 1000

        # obsv_size = response.max_stage * max_stage_directive * (2 + response.max_param)
        self.obsv_size = 128
        self.action_space, self.observation_space = action_functions().init_actions(
            response.schedule_map_range,
            self.obsv_low,
            self.obsv_high,
            self.obsv_size,
            self.action_function,
        )
        # self.action_space = spaces.Discrete(response.schedule_map_range)
        # self.observation_space = spaces.Box(low=obsv_low, high=obsv_high, shape=(self.obsv_size,), dtype=np.int32)
        self.map = np.empty(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )

        self.state = np.random.random(128)
        self.history = np.zeros(self.obsv_size)
        self.input = [self.state, self.history]
        self.actions = []

        self.np_random, seed = seeding.np_random(None)
        # #add for grpc
        self.request = None
        self.response = None
        self.reset()
        self.tstart = time.time()

    def get_reward(self, action):
        reward, self.min_exec_time_sec = reward_function().get_reward(
            self.response.exec_time_sec,
            self.min_exec_time_sec,
            weight=self.reward_scale,
            method=self.reward_function,
        )

        return reward
        # if self.response.exec_time_sec < self.min_exec_time_sec:
        #     exec_diff = self.min_exec_time_sec - self.response.exec_time_sec
        #     self.min_exec_time_sec = self.response.exec_time_sec
        #     return exec_diff * self.reward_scale
        # return 0.0

    def get_map(self, request):
        # collect the map number of self.map
        p = self.map_count * len(self.response.op.elem_id)
        for i, id in enumerate(self.response.op.elem_id):
            self.map[p + i] = id
        action = request.op.map_code
        self.action_count[action] = self.action_count.get(action, 0) + 1
        self.map_count += 1

        # return observation

    def get_obs(self, reward, done, error):
        ek = "best_exec"
        sk = "best_schedule"
        info = {ek: self.min_exec_time_sec, sk: "n/a"}
        code = re.sub(r"\$[0-9]*", "", self.render())
        if not done and not error and self.min_exec_time_sec < self.best_exec_time_sec:
            self.best_exec_time_sec = self.min_exec_time_sec
            done = True
            info[sk] = re.sub(r"\$[0-9]*", "", self.render())
            epinfo = {
                "t": round(time.time() - self.tstart, 6),
                "best_exec": info[ek],
                "best_schedule": info[sk],
            }
            print(epinfo)
            if csv_writer:
                print("enter write")
                csv_writer.writerow(
                    [round(time.time() - self.tstart, 6), info[ek], info[sk]]
                )
                f.flush()

        conn = sqlite3.connect(
            "/home/SuperSonic/tasks/src/opt_test/MCTS/examples/supersonic.db"
        )
        c = conn.cursor()
        sql = "INSERT INTO HALIDE (TIME, RESULT,ACTIONS,REWARD,LOG) \
                                      VALUES (?, ?, ?, ?, ?)"
        c.execute(
            sql,
            (
                round(time.time() - self.tstart),
                code,
                str(self.actions),
                reward,
                self.min_exec_time_sec,
            ),
        )
        # c.execute("INSERT INTO HALIDE (RESULT,ACTIONS,REWARD,LOG) \
        #                               VALUES (code, self.actions, reward, self.min_exec_time_sec )")
        conn.commit()
        conn.close()
        # select dbl

        # cursor = c.execute("SELECT RESULT,ACTIONS,REWARD,LOG  from HALIDE")
        # for row in cursor:
        #     print("RESULT = ", row[0])
        #     print("ACTIONS = ", row[1])
        #     print("REWARD = ", row[2])
        #     print("LOG = ", row[3], "\n")
        # conn.close()

        # embedding obs
        self.input = [code, self.actions]
        self.state_function = "Actionhistory"
        self.state = observation_function().get_observation(
            self.input, self.obsv_size, self.state_function
        )

        if (
            not done
            and not error
            and self.map_count >= (self.max_stage * self.max_stage_directive)
        ):
            done = True
        if self.map_count >= (self.max_stage * self.max_stage_directive):
            self.restart()

        # print(np.array(self.map))
        # done = True
        # print(f"done:f{done}")
        return np.array(self.state), reward, done, info

    def step(self, action):  # get
        self.actions.append(action)
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        if action == 0:
            return self.get_obs(self.noop_reward, True, False)

        if not (self.action_count.get(action, 0) + 1 == 1):
            return self.get_obs(self.error_reward, False, True)

        request = schedule_pb2.ScheduleStepRequest()
        request.op.map_code = action
        response = stuball.step(request)

        if response.exec_timeout and response.exec_error:
            return self.get_obs(self.timeout_error_reward, True, True)

        if response.exec_error:
            return self.get_obs(self.error_reward, False, True)

        self.get_map(request)

        return self.get_obs(self.get_reward(action), False, False)

    def reset(self):  # get
        request = schedule_pb2.ScheduleResetRequest()
        response = stuball.reset(request)

        self.min_exec_time_sec = self.init_exec_time_sec
        self.action_count.clear()
        self.map_count = 0
        self.map[:] = 0
        self.actions = []

        action = self.np_random.randint(self.action_space.n)  # random a num in actions

        # get_request
        request = schedule_pb2.ScheduleStepRequest()
        request.op.map_code = action
        response = stuball.step(request)  # reset->step
        if not response.exec_error:
            self.response = response
            self.get_map(request)  # 对比此处_state函数应该对比get_obs函数
            self.get_reward(action)
        return np.array(self.map)

    def render(self, mode="human"):  # get

        request = schedule_pb2.ScheduleRenderRequest()
        response = stuball.render(request)

        out = StringIO()
        for line_content in response.schedule_str:
            out.write(line_content)
            out.write(" ")
        return out.getvalue()

    def close(self):  # get

        request = schedule_pb2.ScheduleCloseRequest()
        response = self.stub.close(request)


# if __name__ == '__main__':
#     channel = grpc.insecure_channel("localhost:50051")
#     stub = schedule_pb2_grpc.ScheduleServiceStub(channel)
#     env = HalideEnv(algorithm_id=11,input_image='alfaromeo_gray_64x.png',max_stage_directive=8,log_path="result",stub=stub)
#     env.reset()
#     env.step(env.action_space.sample())
#     print(env.state)
#     env.step(env.action_space.sample())
#     print(env.state)
