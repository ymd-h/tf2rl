import time
import logging
import multiprocessing

from cpprb import ReplayBuffer, MPReplayBuffer


def import_tf():
    import tensorflow as tf
    if tf.config.experimental.list_physical_devices('GPU'):
        for cur_device in tf.config.experimental.list_physical_devices("GPU"):
            print(cur_device)
            tf.config.experimental.set_memory_growth(cur_device, enable=True)
    return tf



def explorer(global_rb, queue, trained_steps, is_training_done,
             env_fn, policy_fn, set_weights_fn, nsteps=100):
    """

    """
    import_tf()
    logger = logger.getLogger("tf2rl")

    env = env_fn()

    policy = policy_fn(env=env, name="Explorer")
    kwargs = get_default_rb_dict(nsteps)
    kwargs["env_dict"]["logp"] = {}
    local_rb = ReplayBuffer(**kwargs)
    local_idx = np.arange(local_rb.get_buffer_size(), dtype=np.int)

    obs = env.reset()
    while not is_training_done.is_set():
        if not queue.empty():
            w = queue.get()
            set_weights_fn(policy_fn,w)

        for _ in range(nsteps):
            action, logp, v = policy.get_action_and_val(obs)
            next_obs, rew, done, _ = env.step(action)

            local_rb.add(obs=obs,
                         act=action,
                         rew=rew,
                         next_obs=next_obs,
                         done=done,
                         logp=logp)

            if done:
                local_rb.on_episode_end()
                obs = env.reset()
                break
            else:
                obs = next_obs


        rollout = local_rb.get_all_transitions()
        rollout["LSTM"] = LSTM
        global_rb.add(**rollout)
        local_rb.clear()


def learner():
    pass


def run(args, env_fn, policy_fn, get_weights_fn, set_weights_fn):
    initialize_logger(logging_level=logging.getLevelName(args.logging_level))
