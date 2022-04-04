import os
import threading
import time
import timeit
import traceback
import typing
import torch
import sqlite3
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from SuperSonic.policy_search.util import atari_wrappers
from SuperSonic.policy_search.util.core import environment
from SuperSonic.policy_search.util.core import prof
from SuperSonic.policy_search.util.core import vtrace
from SuperSonic.policy_search.util.models.attention_augmented_agent import (
    AttentionAugmentedAgent,
)
from SuperSonic.policy_search.util.models.resnet_monobeast import ResNet
from SuperSonic.policy_search.util.models.atari_net_monobeast import AtariNet
from SuperSonic.policy_search.util.analysis.gradient_tracking import GradientTracker

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.


# logging.basicConfig(
#     format="[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] "
#     "%(message)s",
#     level=0,
# )
# logging.getLogger("matplotlib.font_manager").disabled = True

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

gradient_tracker = GradientTracker()


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages**2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        # target=torch.flatten(actions, 0, 1),
        target=torch.flatten(actions, 0, 2),
        reduction="none",
    )
    # cross_entropy = cross_entropy.view_as(advantages)
    cross_entropy = cross_entropy.view_as(actions)
    return torch.sum(cross_entropy * advantages.detach())


def act(
    flags,
    env: str,
    task: int,
    full_action_space: bool,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        # logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        policy = flags.Policy
        trainset, testset = flags.Dataset[0], flags.Dataset[1]

        # create the environment from command line parameters
        # => could also create a special one which operates on a list of games (which we need)
        gym_env = create_env(
            env,
            policy_all=policy,
            dataset=trainset,
            frame_height=flags.frame_height,
            frame_width=flags.frame_width,
            gray_scale=(flags.aaa_input_format == "gray_stack"),
            full_action_space=full_action_space,
            task=task,
        )

        # generate a seed for the environment (NO HUMAN STARTS HERE!), could just
        # use this for all games wrapped by the environment for our application
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)

        # wrap the environment, this is actually probably the point where we could
        # use multiple games, because the other environment is still one from Gym
        env = environment.Environment(gym_env)

        # get the initial frame, reward, done, return, step, last_action
        env_output = env.initial()

        # perform the first step
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            # get a buffer index from the queue for free buffers (?)
            index = free_queue.get()
            # termination signal (?) for breaking out of this loop
            if index is None:
                break

            # Write old rollout end.
            # the keys here are (frame, reward, done, episode_return, episode_step, last_action)
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            # here the keys are (policy_logits, baseline, action)
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            # I think the agent_state is just the RNN/LSTM state (which will be the "initial" state for the next step)
            # not sure why it's needed though because it really just seems to be the initial state before starting to
            # act; however, it might be randomly initialised, which is why we might want it...
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout
            for t in range(flags.unroll_length):
                timings.reset()

                # forward pass without keeping track of gradients to get the agent action
                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")

                # agent acting in the environment
                env_output = env.step(agent_output["action"])

                timings.time("step")

                # writing the respective outputs of the current step (see above for the list of keys)
                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")

            # after finishing a trajectory put the index in the "full queue",
            # presumably so that the data can be processed/sent to the learner
            full_queue.put(index)

        # if actor_index == 0:
        #     logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        # logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock(),
):
    # need to make sure that we wait until batch_size trajectories/rollouts have been put into the queue
    with lock:
        timings.time("lock")
        # get the indices of actors "offering" trajectories/rollouts to be processed by the learner
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")

    # create the batch as a dictionary for all the data in the buffers (see act() function for list of
    # keys), where each entry is a tensor of these values stacked across actors along the first dimension,
    # which I believe should be the "batch dimension" (see _format_frame())
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }

    # similar thing for the initial agent states, where I think the tuples are concatenated to become torch tensors
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")

    # once the data has been "transferred" into batch and initial_agent_state,
    # signal that the data has been processed to the actors
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")

    # move the data to the right device (e.g. GPU)
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")

    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    stats,
    lock=threading.Lock(),
    envs=None,
):
    """Performs a learning (optimization) step."""
    with lock:
        # forward pass with gradients
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        # if specified, clip rewards between -1 and 1
        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        # the "~"/tilde operator is apparently kind of a complement or # inverse, so maybe this just reverses
        # the "done" tensor? in that case would discounting only be applied when the game was NOT done?
        discounts = (~batch["done"]).float() * flags.discounting

        # prepare tensors for computation of the loss
        task = F.one_hot(batch["task"].long(), flags.num_tasks).float()
        clipped_rewards = clipped_rewards[:, :, None]
        discounts = discounts[:, :, None]

        mu = model.baseline.mu[None, None, :]
        sigma = model.baseline.sigma[None, None, :]

        # get the V-trace returns; I hope nothing needs to be changed about this, but I think
        # once one has the V-trace returns it can just be plugged into the PopArt equations
        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
            normalized_values=learner_outputs["normalized_baseline"],
            mu=mu,
            sigma=sigma,
        )

        # PopArt normalization
        with torch.no_grad():
            normalized_vs = (vtrace_returns.vs - mu) / sigma

        # policy gradient loss
        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages * task,
        )

        # value function/baseline loss (1/2 * squared difference between V-trace and value function)
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            # vtrace_returns.vs - learner_outputs["baseline"]
            (normalized_vs - learner_outputs["normalized_baseline"])
            * task
        )

        # entropy loss for getting a "diverse" action distribution (?), "normal entropy" over action distribution
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        # do the backward pass (WITH GRADIENT NORM CLIPPING) and adjust hyperparameters (scheduler, ?)
        optimizer.zero_grad()
        total_loss.backward()
        # plot_grad_flow(model.named_parameters(), flags)
        gradient_tracker.process_backward_pass(model.named_parameters())
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        # update the PopArt parameters, which the optimizer does not take care of
        if flags.use_popart:
            model.baseline.update_parameters(vtrace_returns.vs, task)

        # update the actor model with the new parameters
        actor_model.load_state_dict(model.state_dict())

        # get the returns only for finished episodes (where the game was played to completion)
        episode_returns = batch["episode_return"][batch["done"]]
        stats["step"] = stats.get("step", 0) + flags.unroll_length * flags.batch_size
        stats["episode_returns"] = tuple(episode_returns.cpu().numpy())
        stats["mean_episode_return"] = torch.mean(episode_returns).item()
        stats["total_loss"] = total_loss.item()
        stats["pg_loss"] = pg_loss.item()
        stats["baseline_loss"] = baseline_loss.item()
        stats["entropy_loss"] = entropy_loss.item()
        stats["mu"] = mu[0, 0, :]
        stats["sigma"] = sigma[0, 0, :]
        if "env_step" not in stats:
            stats["env_step"] = {}
        for task in batch["task"][0].cpu().numpy():
            stats["env_step"][envs[task]] = (
                stats["env_step"].get(envs[task], 0) + flags.unroll_length
            )

        return stats


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(  # seems like these "inner" dicts could also be something else...
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1, flags.num_tasks), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1, 1), dtype=torch.int64),
        normalized_baseline=dict(size=(T + 1, flags.num_tasks), dtype=torch.float32),
        task=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}

    # basically create a bunch of empty torch tensors according to the sizes in the specs dicts above
    # and do this for the specified number of buffers, so that there will be a list of length flags.num_buffers
    # for each key
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    print("Start client RL training...")
    # prepare for logging and saving models
    # conn = sqlite3.connect("/home/huanting/supersonic/SUPERSONIC/SuperSonic/SQL/supersonic.db")
    conn = sqlite3.connect("./SuperSonic/SQL/supersonic.db")
    c = conn.cursor()
    # result,action history,reward,execution outputs
    try:
        c.execute(
            """CREATE TABLE SUPERSONIC
                       (
                       ID           FLOAT       NOT NULL,
                       TASK         TEXT       NOT NULL,
                       ACTION        INTEGER      NOT NULL,
                       REWARD        FLOAT  NOT NULL,
                       LOG TEXT );"""
        )
        print("Table created successfully")
    except:
        pass

    conn.commit()
    conn.close()

    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    # plogger = file_writer.FileWriter(
    #     xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    # )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )
    if flags.save_model_every_nsteps > 0:
        os.makedirs(checkpointpath.replace("model.tar", "intermediate"), exist_ok=True)
    # get policy and dataset wht
    policy = flags.Policy
    trainset, testset = flags.Dataset[0], flags.Dataset[1]

    # get a list and determine the number of environments
    environments = flags.env.split(",")
    flags.num_tasks = len(environments)

    # set the number of buffers
    if flags.num_buffers is None:
        flags.num_buffers = max(
            2 * flags.num_actors * flags.num_tasks, flags.batch_size
        )
    if flags.num_actors * flags.num_tasks >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    # set the device to do the training on
    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        # logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        # logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    # set the right agent class
    if flags.agent_type.lower() in [
        "aaa",
        "attention_augmented",
        "attention_augmented_agent",
    ]:
        Net = AttentionAugmentedAgent
        # logging.info("Using the Attention-Augmented Agent architecture.")
    elif flags.agent_type.lower() in ["rn", "res", "resnet", "res_net"]:
        Net = ResNet
        # logging.info("Using the ResNet architecture (monobeast version).")
    else:
        Net = AtariNet
        # logging.warning("No valid agent type specified. Using the default agent.")

    # create a dummy environment, mostly to get the observation and action spaces from
    # TODO:
    gym_env = create_env(
        environments[0],
        policy_all=policy,
        dataset=trainset,
        frame_height=flags.frame_height,
        frame_width=flags.frame_width,
        gray_scale=(flags.aaa_input_format == "gray_stack"),
    )

    observation_space_shape = gym_env.observation_space.shape
    action_space_n = gym_env.action_space.n
    full_action_space = False

    for environment in environments:
        gym_env = create_env(environment, policy_all=policy, dataset=trainset)
        if gym_env.action_space.n != action_space_n:
            # logging.warning("Action spaces don't match, using full action space.")
            full_action_space = True
            action_space_n = 18
            break

    # create the model and the buffers to pass around data between actors and learner
    model = Net(
        observation_space_shape,
        action_space_n,
        use_lstm=flags.use_lstm,
        num_tasks=flags.num_tasks,
        use_popart=flags.use_popart,
        reward_clipping=flags.reward_clipping,
        rgb_last=(flags.aaa_input_format == "rgb_last"),
    )
    buffers = create_buffers(flags, observation_space_shape, model.num_actions)

    # I'm guessing that this is required (similarly to the buffers) so that the
    # different threads/processes can all have access to the parameters etc. (?)
    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    # create stuff to keep track of the actor processes
    actor_processes = []
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    # create and start actor threads (the same number for each environment)

    for i, environment in enumerate(environments):
        for j in range(flags.num_actors):

            actor = ctx.Process(
                target=act,
                args=(
                    flags,
                    environment,
                    i,
                    full_action_space,
                    i * flags.num_actors + j,
                    free_queue,
                    full_queue,
                    model,
                    buffers,
                    initial_agent_state_buffers,
                ),
            )
            actor.start()
            actor_processes.append(actor)

    learner_model = Net(
        observation_space_shape,
        action_space_n,
        use_lstm=flags.use_lstm,
        num_tasks=flags.num_tasks,
        use_popart=flags.use_popart,
        reward_clipping=flags.reward_clipping,
        rgb_last=(flags.aaa_input_format == "rgb_last"),
    ).to(device=flags.device)

    # the hyperparameters in the paper are found/adjusted using population-based training,
    # which might be a bit too difficult for us to do; while the IMPALA paper also does
    # some experiments with this, it doesn't seem to be implemented here
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "mu",
        "sigma",
    ] + ["{}_step".format(e) for e in environments]
    # print("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        # step in particular needs to be from the outside scope, since all learner threads can update
        # it and all learners should stop once the total number of steps/frames has been processed
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )
            learn(
                flags,
                model,
                learner_model,
                batch,
                agent_state,
                optimizer,
                scheduler,
                stats,
                envs=environments,
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys if "_step" not in k})
                for e in stats["env_step"]:
                    to_log["{}_step".format(e)] = stats["env_step"][e]
                # plogger.log(to_log)
                step += (
                    T * B
                )  # so this counts the number of frames, not e.g. trajectories/rollouts

        # if i == 0:
        # logging.info("Batch and learn: %s", timings.summary())

    # populate the free queue with the indices of all the buffers at the start
    for m in range(flags.num_buffers):
        free_queue.put(m)

    # start as many learner threads as specified => could in principle do PBT
    threads = []

    for i in range(5):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def save_latest_model():
        if flags.disable_checkpoint:
            return
        # logging.info("Saving checkpoint to %s", checkpointpath)
        # print("Saving checkpoint to %s", checkpointpath)

        torch.save(
            {
                "model_state_dict": learner_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )
        # print("model has been saved in ", checkpointpath)

    def save_intermediate_model():
        save_model_path = checkpointpath.replace(
            "model.tar",
            "intermediate/model." + str(stats.get("step", 0)).zfill(9) + ".tar",
        )
        # print("Saving checkpoint to %s", checkpointpath)
        # logging.info("Saving model to %s", save_model_path)
        torch.save(
            {
                "model_state_dict": learner_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "stats": stats,
                "flags": vars(flags),
            },
            save_model_path,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        last_savemodel_nsteps = 0

        num = 0
        while step < flags.total_steps:
            num = num + 1
            start_step = stats.get("step", 0)
            start_time = timer()
            time.sleep(3)
            end_step = stats.get("step", 0)

            if timer() - last_checkpoint_time > 10 * 60:
                # save every 10 min.
                save_latest_model()
                last_checkpoint_time = timer()

            if (
                flags.save_model_every_nsteps > 0
                and end_step >= last_savemodel_nsteps + flags.save_model_every_nsteps
            ):
                # save model every save_model_every_nsteps steps
                save_intermediate_model()
                last_savemodel_nsteps = end_step

            sps = (end_step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
    except KeyboardInterrupt:
        gradient_tracker.print_total()
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        # print("Saving checkpoint to %s", checkpointpath)
        # logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)  # send quit signal to actors
        for actor in actor_processes:
            actor.join(timeout=10)
        gradient_tracker.print_total()

    save_latest_model()
    # GIVE ME POLICY
    conn = sqlite3.connect("./SuperSonic/SQL/supersonic.db")
    c = conn.cursor()
    cursor = c.execute("SELECT ACTION  from SUPERSONIC")
    a = cursor.fetchall()
    policynum = int(max(a[-20:], key=a.count)[0])
    bestpolicy = policy[0][policynum]

    # waiting for thread stop
    time.sleep(300)

    return bestpolicy


def create_env(
    env,
    policy_all,
    dataset,
    frame_height=84,
    frame_width=84,
    gray_scale=True,
    full_action_space=False,
    task=0,
):
    return atari_wrappers.wrap_pytorch_task(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(env, policy_all, dataset),
            clip_rewards=False,
            frame_stack=True,
            frame_height=frame_height,
            frame_width=frame_width,
            gray_scale=gray_scale,
            scale=False,
        ),
        task=task,
    )
