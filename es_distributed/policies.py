import logging
import pickle
import os
import subprocess
import retro
import gym_remote.client as grc
import gym_remote.exceptions as gre

import h5py
import numpy as np
import tensorflow as tf

from . import tf_util as U


import random
import copy

#ppo requirements
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as ppo2_policies
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines import logger

from sonic_util_train import AllowBacktracking, SonicDiscretizer, RewardScaler, FrameStack, WarpFrame, make_env
import traceback
import threading
import joblib
import time
import shutil
import csv


logger = logging.getLogger(__name__)


class Policy:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.scope = self._initialize(*args, **kwargs)
        #print('scope is {}'.format(self.scope))
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope.name)
        #print('all variables', self.all_variables)

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
        print('trainable variables', self.trainable_variables)
        self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        print('num_params', self.num_params)
        self._setfromflat = U.SetFromFlat(self.trainable_variables)
        self._getflat = U.GetFlat(self.trainable_variables)

        logger.info('Trainable variables ({} parameters)'.format(self.num_params))
        for v in self.trainable_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))
        logger.info('All variables')
        for v in self.all_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))

        placeholders = [tf.placeholder(v.value().dtype, v.get_shape().as_list()) for v in self.all_variables]
        self.set_all_vars = U.function(
            inputs=placeholders,
            outputs=[],
            updates=[tf.group(*[v.assign(p) for v, p in zip(self.all_variables, placeholders)])]
        )

    def _initialize(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, filename):
        assert filename.endswith('.h5')
        with h5py.File(filename, 'w') as f:
            for v in self.all_variables:
                f[v.name] = v.eval()
            # TODO: it would be nice to avoid pickle, but it's convenient to pass Python objects to _initialize
            # (like Gym spaces or numpy arrays)
            f.attrs['name'] = type(self).__name__
            f.attrs['args_and_kwargs'] = np.void(pickle.dumps((self.args, self.kwargs), protocol=-1))

    @classmethod
    def Load(cls, filename, extra_kwargs=None):
        with h5py.File(filename, 'r') as f:
            args, kwargs = pickle.loads(f.attrs['args_and_kwargs'].tostring())
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            policy = cls(*args, **kwargs)
            policy.set_all_vars(*[f[v.name][...] for v in policy.all_variables])
        return policy

    # === Rollouts/training ===

    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        if timestep_limit is None:
            timestep_limit = 4500
        rews = []
        t = 0
        if save_obs:
            obs = []
        #ob = env.reset()
        for game, state in self.get_training_envs():
            self.env.close()
            print('launching {} {}'.format(game, state))
            self.env = self.launch_env(game, state)
            ob = self.env.reset()
            for _ in range(timestep_limit):
                embedding = self.session.run([self.output_layer], {self.model.X: ob[None]})[0].reshape(512,)
                ac = self.act([embedding], random_stream=random_stream)
                if save_obs:
                    #print('appending embedding', embedding.shape)
                    #obs.append(ob)
                    obs.append(embedding)
                ob, rew, done, _ = self.env.step(ac)
                rews.append(rew)
                t += 1
                if render:
                    self.env.render()
                if done:
                    break
        rews = np.array(rews, dtype=np.float32)
        if save_obs:
            return rews, t, np.array(obs)
        print('completed rollout')
        return rews, t

    def act(self, ob, random_stream=None):
        raise NotImplementedError

    def set_trainable_flat(self, x):
        self._setfromflat(x)

    def get_trainable_flat(self):
        return self._getflat()

    @property
    def needs_ob_stat(self):
        raise NotImplementedError

    def set_ob_stat(self, ob_mean, ob_std):
        raise NotImplementedError


def bins(x, dim, num_bins, name):
    scores = U.dense(x, dim * num_bins, name, U.normc_initializer(0.01))
    scores_nab = tf.reshape(scores, [-1, dim, num_bins])
    return tf.argmax(scores_nab, 2)  # 0 ... num_bins-1


class SonicPolicy(Policy):
    def _initialize(self, ac_noise_std):
        import numpy as np
        BUTTONS = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        COMBOS = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        valid_actions = []
        for action in COMBOS:
            arr = np.array([False] * 12)
            for button in action:
                arr[BUTTONS.index(button)] = True
            valid_actions.append(arr)
        self.ACTIONS = valid_actions

        self.session = tf.get_default_session()

        #just the first one in the list
        self.env = self.launch_env('SonicTheHedgehog-Genesis', 'SpringYardZone.Act1')
        self.model = ppo2_policies.CnnPolicy(
            self.session,
            self.env.observation_space,
            self.env.action_space,
            1,
            1)
        self.a0 = self.model.pd.sample()
        #params = tf.trainable_variables()
        self.output_layer = tf.get_default_graph().get_tensor_by_name('model/Relu_3:0')
        
        self.input_shape = (512,)
        self.ac_noise_std = ac_noise_std

        with tf.variable_scope(type(self).__name__) as scope:
            # Observation normalization
            ob_mean = tf.get_variable(
                'ob_mean', self.input_shape, tf.float32, tf.constant_initializer(0.1), trainable=False)
            ob_std = tf.get_variable(
                'ob_std', self.input_shape, tf.float32, tf.constant_initializer(0.001), trainable=False)
            in_mean = tf.placeholder(tf.float32, self.input_shape)
            in_std = tf.placeholder(tf.float32, self.input_shape)
            self._set_ob_mean_std = U.function([in_mean, in_std], [], updates=[
                tf.assign(ob_mean, in_mean),
                tf.assign(ob_std, in_std),
            ])

            # Policy network
            o = tf.placeholder(tf.float32, [None] + list(self.input_shape))
            #o = self.output_layer

            # normalize
            a = tf.clip_by_value((o - ob_mean) / ob_std, -5.0, 5.0)
            a = U.dense(a, 7, 'out', U.normc_initializer(0.01))

            self._act = U.function([o], a)
        return scope

    def get_training_envs(self):
        envs = []
        with open('./sonic-train.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                game, state = row
                if game == 'game':
                    continue
                else:
                    envs.append(row)
        return envs

    def launch_env(self, game, state):
        #game, state = random.choice(env_data)
        # retro-contest-remote run -s tmp/sock -m monitor -d SonicTheHedgehog-Genesis GreenHillZone.Act1
        
        #base_dir = './remotes/'
        #if os.path.exists(base_dir):
        #    shutil.rmtree(base_dir)
        #os.makedirs(base_dir, exist_ok=True)
        #os.makedirs(base_dir + state, exist_ok=True)
        #socket_dir = base_dir + "{}/sock".format(state)
        #os.makedirs(socket_dir, exist_ok=True)
        #monitor_dir = base_dir + "{}/monitor".format(state)
        #os.makedirs(monitor_dir, exist_ok=True)
        #subprocess.Popen(["retro-contest-remote", "run", "-s", socket_dir, '-m', monitor_dir, '-d', game, state], stdout=subprocess.PIPE)
        #env = grc.RemoteEnv(socket_dir)
        env = retro.make(game, state)

        env = SonicDiscretizer(env)
        env = RewardScaler(env)
        env = WarpFrame(env)
        env = FrameStack(env, 4)

        return env

    def act(self, ob, random_stream=None):
        #embedding = self.session.run([self.output_layer], {self.model.X: ob})[0].reshape(512,)
        a = self._act(ob)
        if random_stream is not None and self.ac_noise_std != 0:
            a += random_stream.randn(*a.shape) * self.ac_noise_std
        action = np.argmax(a[0])
        env_action = self.ACTIONS[action]
        #print('ACTIONS, action, env_action', self.ACTIONS, action, env_action)
        return env_action

    @property
    def needs_ob_stat(self):
        return True

    @property
    def needs_ref_batch(self):
        return False

    def set_ob_stat(self, ob_mean, ob_std):
        #if ob_mean.shape != (512,):
        #    print('ob_mean', ob_mean)
        #    print('ob_std', ob_std)
        #    pass
        if ob_mean.shape != (512,):
            print('bad ob_mean_shape', ob_mean.shape)
            return
        self._set_ob_mean_std(ob_mean, ob_std)

    def initialize_from(self, filename, ob_stat=None):
        """
        Initializes weights from another policy, which must have the same architecture (variable names),
        but the weight arrays can be smaller than the current policy.
        """
        with h5py.File(filename, 'r') as f:
            f_var_names = []
            f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'

            init_vals = []
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                f_shp = f[v.name].shape
                assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
                    'This policy must have more weights than the policy to load'
                init_val = v.eval()
                # ob_mean and ob_std are initialized with nan, so set them manually
                if 'ob_mean' in v.name:
                    init_val[:] = 0
                    init_mean = init_val
                elif 'ob_std' in v.name:
                    init_val[:] = 0.001
                    init_std = init_val
                # Fill in subarray from the loaded policy
                init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
                init_vals.append(init_val)
            self.set_all_vars(*init_vals)

        if ob_stat is not None:
            ob_stat.set_from_init(init_mean, init_std, init_count=1e5)
