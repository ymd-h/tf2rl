import time
import logging
import multiprocessing

from cpprb import ReplayBuffer, MPReplayBuffer

from tf2rl.algos.policy_base import OffPolicyAgent


def import_tf():
    import tensorflow as tf
    if tf.config.experimental.list_physical_devices('GPU'):
        for cur_device in tf.config.experimental.list_physical_devices("GPU"):
            print(cur_device)
            tf.config.experimental.set_memory_growth(cur_device, enable=True)
    return tf



def explorer(global_rb, queue, trained_steps, is_training_done,
             env_fn, policy_fn, set_weights_fn, unroll_length=100):
    """

    """
    import_tf()
    logger = logger.getLogger("tf2rl")

    env = env_fn()

    policy = policy_fn(env=env, name="Explorer")
    kwargs = get_default_rb_dict(unroll_length)
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


def learner(global_rb, trained_steps, is_training_done, policy_fn, get_weights_fn,
            env, n_training, update_freq, evaluation_freq, gpu, queues,
            c_bar = 1, rho_bar = 1):
    tf.import_tf()
    logger = logging.getLogger("tf2rl")

    policy = policy_fn(env, "Learner", gpu=gpu)

    output_dir = prepare_output_dir(args=None,
                                    user_specified_dir="./results",
                                    suffix="learner")
    writer = tf.summary.create_file_writer(output_dir)
    writer.set_as_default()

    while not global_rb.get_stored_size() < policy.n_warmup:
        time.sleep(1)

    start_time = time.perf_counter()
    while not is_training_done.is_set():
        with trained_steps.get_lock():
            trained_steps.value += 1
        n_trained_steps = trained_steps.value

        samples = global_rb.sample(policy.batch_size)
        ploicy.train(samples["obs"], samples["act"], samples["next_obs"],
                     samples["rew"], samples["done"],
                     samples["logp"], samples["LSTM"])

        # Put updated weights to queue
        if n_trained_steps % update_freq == 0:
            w = get_weights_fn(policy)
            for q in queues[:-1]:
                q.put(w)

        # Periodically do evaluation
        if n_trained_steps % evaluation_freq == 0:
            queues[-1].put((get_weights_fn(policy), n_trained_steps))

        if n_trained_steps >= n_training:
            is_training_done.set()

def run(args, env_fn, policy_fn, get_weights_fn, set_weights_fn):
    initialize_logger(logging_level=logging.getLevelName(args.logging_level))


class IMPALA(OffPolicyAgent):
    def __init__(self,
                 unroll_length=100,
                 memory_capacity=int(1e+6),
                 c_bar = 1,
                 rho_bar = 1,
                 lr = 0.00048,
                 momentum = 0.0,
                 rms_decay = 0.99,
                 rms_epsilon = 0.1,
                 optimizer = None,
                 **kwargs):
        super().__init__(memory_capacity=memory_capacity, **kwargs)
        self.c_bar = c_bar
        self.rho_bar = rho_bar

        self.lr = lr
        self.momentum = momentum
        self.rms_decay = rms_decay
        self.rms_epsilon = rms_epsilon
        self.optimizer = (optimizer
                          or tf.keras.optimizers.RMSProp(learning_rate = self.lr,
                                                         rho = self.rms_decay,
                                                         momentum = self.momentum,
                                                         epsilon = self.rms_epsilon))


    def get_action(self, state, test=False):
        pass

    def get_action_and_val(self, state, test=False):
        pass

    @staticmethod
    def get_argument(parser=None):
        parser = OffPolicyAgent.get_argument(parser)
        parser.add_argument('--unroll-length', default=100)
        parser.add_argument('--c-bar', default=1)
        parser.add_argument('--rho-bar', default=1)
        return parser
