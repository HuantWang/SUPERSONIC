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
    """A :class:
        Halide is a domain-specific language and compiler for image processing pipelines (or graphs) with multiple computation stages.
        This task builds upon Halide version 10. Our evaluation uses ten Halide applications that have been heavily tested on the Halide
        compiler. We measure the execution time of each benchmark when processing three image datasets provided by the Halide benchmark suite.
        The benchmarks are optimized to run on a multi-core CPU.

    Source:
        This environment corresponds to the version of the Halide. (https://github.com/halide/Halide)
        paper link: https://people.csail.mit.edu/jrk/halide-pldi13.pdf

    Observation:
        Type: Box(100)
        Pipeline’s schedule will be convert to vectors by different embedding approaches,
        e.g. Word2vec, Doc2vec, CodeBert ...

    Actions:
        Type: Discrete(4)
        Num      Action      Description
        0        adding      Adds an optimization to the stage.
        1        removing    Removes an optimization to the stage.
        2        decreasing  Decreases the value (by one) of an enabled parameterized option.
        3        increasing  Increases the value (by one) of an enabled parameterized option.


    Reward:
        In all cases, lower cost is better. We measure the execution time of each benchmark when processing three image
        datasets provided by the Halide benchmark suite.

    Starting State:
        All observations are assigned a uniform random value in [-1..1]

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
        """ Defines the reinforcement leaning environment. Initialise with an environment.

        :param algorithm_id: Encoding of halide benchmark.
        :param input_image: The image datasets to measure halide benchmark.
        :param max_stage_directive:  Max scheduling directives that will be applied to the pipeline stages.
        :param log_path:The path to save result.
        :param state_function:  a state function that can summarize the program after each action as a
                                finite feature vector.
        :param action_function: an action function that can discrete set of actions or transformations that can be applied
                                to a program, such as passes in a compiler
        :param reward_function: a reward function that reports the quality of the actions taken so far.
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
        self.obsv_size = 100
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

        self.state = np.random.random(100)
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
        """ Calculate reward with method "reward_function".

                    :param action: What will agent do in next step.
                    :return: return a reward score after calculating.
                """
        reward = reward_function().get_rew(
           self.response.exec_time_sec,
           self.min_exec_time_sec,
           weight=self.reward_scale,
           reward_function=self.reward_function,
        )


        return reward
        # if self.response.exec_time_sec < self.min_exec_time_sec:
        #     exec_diff = self.min_exec_time_sec - self.response.exec_time_sec
        #     self.min_exec_time_sec = self.response.exec_time_sec
        #     return exec_diff * self.reward_scale
        # return 0.0

    def get_map(self, request):
        """ encode the schedule as observation.
                    :param request: This is requset from grpc.
                """
        # collect the map number of self.map
    
        p = self.map_count * len(self.response.op.elem_id)
        for i, id in enumerate(self.response.op.elem_id):
            self.map[p + i] = id
        action = request.op.map_code
        self.action_count[action] = self.action_count.get(action, 0) + 1
        self.map_count += 1

        # return observation

    def get_obs(self, reward, done, error):
        """ feedback the observation with method "observation_function".

                    :param reward:A reward, to describe how the agent "ought" to behave.
                    :param done: The RL engine's state,ture or false.
                    :return: A tuple of observation, observation_mask, score, done, and info.
                """
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
            "/home/sys/SUPERSONIC/SuperSonic/SQL/supersonic.db"
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
        #self.state_function = "Actionhistory"
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
        """Take a step.

                    :param action: An action, or a sequence of actions. When multiple
                            actions are provided the observation and reward are returned after
                            running all of the actions.
                    :return: A tuple of observation, reward, done, and info. Observation and
                            reward are None if default observation/reward is not set. If done is
                            True, observation and reward may also be None (e.g. because the
                            service failed).
                """
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
        """ reset the RL environment.
                """
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
        """ record the schedule.
                """
        request = schedule_pb2.ScheduleRenderRequest()
        response = stuball.render(request)

        out = StringIO()
        for line_content in response.schedule_str:
            out.write(line_content)
            out.write(" ")
        return out.getvalue()

    def close(self):  # get
        """ close grpc connection.
                """
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
