from player.player import Player
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
import itertools
from collections import deque
import random
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from player.prioritized_memory import Memory
from player.deep_neural_network_model import DnnModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy


def saveCSV(result: list, path):
    sc = pd.DataFrame(result, columns=['reward', 'success', 'failure', 'sensing'])
    sc.to_csv(path, index=False)


class DqnPlayer(Player):
    def __init__(self, identifier: str, max_num_unit_packet: int, observation_history_length: int,
                 sensing_unit_packet_length_ratio: int, unit_packet_success_reward: float,
                 unit_packet_failure_reward: float, dnn_layers_list: List[Dict], random_sensing_prob: float,
                 sensing_discount_factor: float, dnn_learning_rate: float, scenario: int, modelNumber: int,
                 dueling: bool = False, double: bool = False, PER: bool = False, noisy: bool = False,
                 distribution: bool = False, num_support: int = 1, v_max: float = 0, v_min: float = 0, n_step: int = 1):
        super(DqnPlayer, self).__init__(identifier)
        self._freq_channel_list: List[int] = []
        self._num_freq_channel = 0
        self._max_num_unit_packet = max_num_unit_packet
        self._freq_channel_combination = []
        self._num_freq_channel_combination = 0
        self._num_action = 0
        self._observation_history_length = observation_history_length
        self._sensing_unit_packet_length_ratio = sensing_unit_packet_length_ratio
        self._observation_history = np.empty(0)
        self._cca_thresh = -70
        self._replay_memory = []
        self._latest_observation_dict = None
        self._unit_packet_success_reward = unit_packet_success_reward
        self._unit_packet_failure_reward = unit_packet_failure_reward
        self._main_dnn: Optional[tf.keras.Model] = None
        self._target_dnn: Optional[tf.keras.Model] = None
        self._dnn_layers_list = dnn_layers_list
        self._random_sensing_prob = random_sensing_prob
        self._sensing_discount_factor = sensing_discount_factor
        self._dnn_learning_rate = dnn_learning_rate
        self._scenario = scenario
        self._modelNumber = modelNumber
        self._result = []
        self._tensorboard_callback = self.tensorboard_save()
        self._cce = CategoricalCrossentropy()
        self._dueling = dueling
        self._double = double
        self._PER = PER
        self._noisy = noisy
        self._distribution = distribution
        if distribution is True:
            self._num_support = num_support
        else:
            self._num_support = 1
        self._v_max = v_max
        self._v_min = v_min
        if self._distribution is True:
            self._dz = float(self._v_max - self._v_min) / (self._num_support - 1)
            self._z = [self._v_min + i * self._dz for i in range(self._num_support)]
        self._n_step = n_step
        self._n_step_buffer = deque(maxlen=self._n_step)

    def connect_to_server(self, server_address: str, server_port: int):
        super(DqnPlayer, self).connect_to_server(server_address, server_port)
        self._freq_channel_list = self.operator_info['freq channel list']
        self._num_freq_channel = len(self._freq_channel_list)
        self._freq_channel_combination = [np.where(np.flip(np.array(x)))[0].tolist()
                                          for x in itertools.product((0, 1), repeat=self._num_freq_channel)][1:]
        self._num_freq_channel_combination = 2 ** self._num_freq_channel - 1
        self._num_action = self._num_freq_channel_combination * self._max_num_unit_packet + 1
        self._observation_history = np.zeros((self._observation_history_length, self._num_freq_channel, 2))
        initial_action = {'type': 'sensing'}
        self._latest_observation_dict = self.step(initial_action)
        self.update_observation_history(initial_action, self._latest_observation_dict)
        self._main_dnn = DnnModel(conv_layers_list=self._dnn_layers_list, num_action=self._num_action,
                                  num_support=self._num_support, dueling=self._dueling, noisy=self._noisy,
                                  distribution=self._distribution)
        self._target_dnn = DnnModel(conv_layers_list=self._dnn_layers_list, num_action=self._num_action,
                                    num_support=self._num_support, dueling=self._dueling, noisy=self._noisy,
                                    distribution=self._distribution)
        if self._distribution is True:
            self._main_dnn.compile(optimizer=Adam(lr=self._dnn_learning_rate), loss='categorical_crossentropy')
        else:
            self._main_dnn.compile(optimizer=Adam(lr=self._dnn_learning_rate), loss="mse")

    def train_dnn(self, num_episodes: int, replay_memory_size: int, mini_batch_size: int, initial_epsilon: float,
                  epsilon_decay: float, min_epsilon: float, dnn_epochs: int, progress_report: bool,
                  test_run_length: int):
        epsilon = initial_epsilon
        if self._PER is True:
            self._replay_memory = Memory(replay_memory_size)
        for episode in range(1, num_episodes + 1):
            if progress_report:
                print(f"Episode: {episode} (epsilon: {epsilon})")
            if self._PER is True:
                self.accumulate_prioritized_replay_memory(replay_memory_size, epsilon, progress_report)
                for i in range(dnn_epochs):
                    self.prioritized_batch_replay(batch_size=mini_batch_size)
            else:
                self.accumulate_replay_memory(replay_memory_size, epsilon, progress_report)
                for i in range(dnn_epochs):
                    self.mini_batch_replay(batch_size=mini_batch_size)
            epsilon *= epsilon_decay
            epsilon = max(epsilon, min_epsilon)
            self._target_dnn.set_weights(self._main_dnn.get_weights())
            if episode % 1 == 0:
                if test_run_length > 0:
                    self.test_run(test_run_length)
            # self.model_save(scenario=self._scenario, modelNumber=self._modelNumber, episode=episode)
        self.model_save(scenario=self._scenario, modelNumber=self._modelNumber, episode=num_episodes, CSV=True)

    def get_main_dnn_action_and_value(self, observation: np.ndarray):
        return self.get_dnn_action_and_value(self._main_dnn, observation)

    def get_target_dnn_action_and_value(self, observation: np.ndarray):
        if self._double is True:
            return self.get_ddqn_action_and_value(self._main_dnn, self._target_dnn, observation)
        else:
            return self.get_dnn_action_and_value(self._target_dnn, observation)
        # return self.get_dnn_action_and_value(self._target_dnn, observation)

    def get_ddqn_action_and_value(self, main_dnn: tf.keras.Model, target_dnn: tf.keras.Model, observation: np.ndarray):
        # get argmax from target network, use main network to get q-value
        single = False
        if observation.ndim == 3:
            observation = observation[np.newaxis, ...]
            single = True
        main_action_value = main_dnn.predict(observation)
        target_action_value = target_dnn.predict(observation)
        if self._distribution is True:
            if single:
                z_space = np.repeat(np.expand_dims(self._z, axis=0), self._num_action, axis=0)
                main_action_value = np.sum(main_action_value[0] * z_space, axis=1)
                target_action_value = np.sum(target_action_value[0] * z_space, axis=1)
                best_action = np.argmax(main_action_value, axis=0)
                best_action_value = target_action_value[best_action]
                print('target single')
                return best_action, best_action_value, main_action_value
            else:
                return main_action_value, target_action_value

        else:
            best_action = np.argmax(main_action_value, axis=1)
            best_action_value = []
            for i in range(len(best_action)):
                best_action_value.append(target_action_value[i][best_action[i]])
            if single:
                best_action = best_action[0]
                best_action_value = best_action_value[0]
                main_action_value = main_action_value[0]
            return best_action, best_action_value, main_action_value

    def get_dnn_action_and_value(self, dnn: tf.keras.Model, observation: np.ndarray):
        single = False
        if observation.ndim == 3:
            observation = observation[np.newaxis, ...]
            single = True
        action_value = dnn.predict(observation)
        if self._distribution is True:
            if single:
                z_space = np.repeat(np.expand_dims(self._z, axis=0), self._num_action, axis=0)
                action_value = np.sum(action_value[0] * z_space, axis=1)
                best_action = np.argmax(action_value, axis=0)
                best_value = np.amax(action_value, axis=0)
                return best_action, best_value, action_value
            else:
                return action_value
        else:
            best_action = np.argmax(action_value, axis=1)
            best_value = np.amax(action_value, axis=1)
            if single:
                best_action = best_action[0]
                best_value = best_value[0]
                action_value = action_value[0]
            return best_action, best_value, action_value

    def get_random_action(self, sensing_prob: float):
        tx_data_prob = (1 - sensing_prob) / (self._num_action - 1)
        distribution = tx_data_prob * np.ones(self._num_action)
        distribution[0] = sensing_prob
        return int(np.random.choice(np.arange(self._num_action), p=distribution))

    def get_next_action(self, observation: np.ndarray, random_prob: float):
        if np.random.rand() < random_prob:
            action = self.get_random_action(self._random_sensing_prob)
        else:
            action, _, _ = self.get_main_dnn_action_and_value(observation)
        return int(action)

    def accumulate_replay_memory(self, replay_memory_size: int, random_prob: float, progress_report: bool):
        self._replay_memory.clear()
        ind = 0
        while ind < replay_memory_size:
            if progress_report:
                print(f"Replay memory sample: {ind}/{replay_memory_size}\r", end='')
            prev_observation_history = self._observation_history
            if self._noisy is True:
                action_index = self.get_next_action(prev_observation_history, 0)
            else:
                action_index = self.get_next_action(prev_observation_history, random_prob)
            action_dict = self.convert_action_index_to_dict(action_index)
            observation_dict = self.step(action_dict)
            self._latest_observation_dict = observation_dict
            reward = self.get_reward(action_dict, observation_dict)
            self.update_observation_history(action_dict, observation_dict)
            current_observation_history = self._observation_history
            discount_factor = self._sensing_discount_factor
            if action_dict['type'] == 'tx_data_packet':
                num_unit_packet = action_dict['num_unit_packet']
                discount_factor = discount_factor ** (num_unit_packet * self._sensing_unit_packet_length_ratio)
            self._n_step_buffer.append((prev_observation_history, action_index, reward, current_observation_history, discount_factor))

            if len(self._n_step_buffer) == self._n_step:
                prev_observation_history, action_index, reward, current_observation_history, discount_factor = self.calc_multistep_return(self._n_step_buffer)
                experience = (prev_observation_history, action_index, reward, current_observation_history, discount_factor)
                self._replay_memory.append(experience)
                ind += 1
        if progress_report:
            print()

    def accumulate_prioritized_replay_memory(self, replay_memory_size: int, random_prob: float, progress_report: bool):
        ind = 0
        while ind < replay_memory_size:
            if progress_report:
                print(f"Replay memory sample: {ind}/{replay_memory_size}\r", end='')
            prev_observation_history = self._observation_history
            if self._noisy is True:
                action_index = self.get_next_action(prev_observation_history, 0)
            else:
                action_index = self.get_next_action(prev_observation_history, random_prob)
            action_dict = self.convert_action_index_to_dict(action_index)
            observation_dict = self.step(action_dict)
            self._latest_observation_dict = observation_dict
            reward = self.get_reward(action_dict, observation_dict)
            self.update_observation_history(action_dict, observation_dict)
            current_observation_history = self._observation_history

            _, old_val, _ = self.get_main_dnn_action_and_value(prev_observation_history)
            _, target_val, _ = self.get_target_dnn_action_and_value(current_observation_history)

            discount_factor = self._sensing_discount_factor
            if action_dict['type'] == 'tx_data_packet':
                num_unit_packet = action_dict['num_unit_packet']
                discount_factor = discount_factor ** (num_unit_packet * self._sensing_unit_packet_length_ratio)
            self._n_step_buffer.append(
                (prev_observation_history, action_index, reward, current_observation_history, discount_factor))

            if len(self._n_step_buffer) == self._n_step:
                prev_observation_history, action_index, reward, current_observation_history, discount_factor = self.calc_multistep_return(self._n_step_buffer)
                if self._distribution is True:
                    if prev_observation_history.ndim == 3:
                        prev_observation_history = prev_observation_history[np.newaxis, ...]
                    old_val = self._main_dnn.predict(prev_observation_history)
                    old_val = np.squeeze(old_val)
                    old_val = old_val[action_index]

                    target_val = self.projection_distribution(current_observation_history, 1, reward, discount_factor)
                    error = abs(self._cce(old_val, target_val).numpy())
                    prev_observation_history = np.squeeze(prev_observation_history)
                else:
                    target_val = reward + discount_factor * target_val
                    error = abs(old_val - target_val)
                experience = (prev_observation_history, action_index, reward, current_observation_history, discount_factor)
                self._replay_memory.add(error, experience)
                ind += 1

    def calc_multistep_return(self, n_step_buffer):
        # buffer shape : prev_observation_history, action_index, reward, current_observation_history, discount_factor
        Return = 0
        discount_factor = 1
        for idx in range(self._n_step):
            Return += discount_factor * n_step_buffer[idx][2]  # n_step_buffer[idx][2] is reward
            discount_factor *= n_step_buffer[idx][4]  # n_step_buffer[idx][4] is discount factor
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], discount_factor

    def test_run(self, length: int):
        tx_success = 0
        tx_failure = 0
        sensing = 0
        reward = 0
        time = 0
        for ind in range(length):
            print(f"Test run: {ind}/{length}\r", end='')
            action_index, _, _ = self.get_main_dnn_action_and_value(self._observation_history)
            action_dict = self.convert_action_index_to_dict(int(action_index))
            observation_dict = self.step(action_dict)
            self.update_observation_history(action_dict, observation_dict)
            observation_type = observation_dict['type']
            if observation_type == 'sensing':
                sensing += 1
                time += 1
            elif observation_type == 'tx_data_packet':
                tx_freq_channel_list = action_dict['freq_channel_list']
                success_freq_channel_list = observation_dict['success_freq_channel_list']
                failure_freq_channel_list = list(set(tx_freq_channel_list) - set(success_freq_channel_list))
                num_unit_packet = action_dict['num_unit_packet']
                tx_time = self._sensing_unit_packet_length_ratio * num_unit_packet
                time += tx_time
                tx_success += len(success_freq_channel_list) * tx_time
                tx_failure += len(failure_freq_channel_list) * tx_time
            reward += self.get_reward(action_dict, observation_dict)
        reward /= time
        tx_success /= (time * self._num_freq_channel)
        tx_failure /= (time * self._num_freq_channel)
        sensing /= time
        self._result.append([reward, tx_success, tx_failure, sensing])
        print(f"\nReward: {reward}, Sensing: {sensing}, Tx Success: {tx_success}, Tx Failure: {tx_failure}")

    def mini_batch_replay(self, batch_size: int):
        samples = self._replay_memory
        random.shuffle(samples)
        for i in range(int(len(samples) / batch_size)):
            batch_samples = samples[i * batch_size:(i + 1) * batch_size]
            observation = np.stack([x[0] for x in batch_samples], axis=0)
            rewards = np.stack([x[2] for x in batch_samples], axis=0)
            next_observation = np.stack([x[3] for x in batch_samples], axis=0)
            discount_factor = np.stack([x[4] for x in batch_samples], axis=0)

            if self._distribution is True:

                target_action_distribution_reward = self.get_main_dnn_action_and_value(observation)
                target_dist_reward = np.zeros(np.shape(target_action_distribution_reward))
                proj_distribution = self.projection_distribution(next_observation, batch_size, rewards,
                                                                 discount_factor)
                pr = pd.DataFrame(proj_distribution)
                pr.to_csv('proj_distribution.csv', index=False, encoding='cp949')

                for ind, sample in enumerate(batch_samples):
                    action = sample[1]
                    target_dist_reward[ind][action] = proj_distribution[ind]
                target_action_reward = target_dist_reward

            else:
                _, _, target_action_reward = self.get_main_dnn_action_and_value(observation)
                _, future_reward, _ = self.get_target_dnn_action_and_value(next_observation)
                for ind, sample in enumerate(batch_samples):
                    action = sample[1]
                    immediate_reward = sample[2]
                    target_action_reward[ind, action] = immediate_reward + discount_factor[ind] * future_reward[ind]

            self._main_dnn.fit(observation, target_action_reward, callbacks=[self._tensorboard_callback])

    def prioritized_batch_replay(self, batch_size: int):
        length = int(self._replay_memory.capacity / batch_size)
        for i in range(length):
            mini_batch, idxs, is_weights = self._replay_memory.sample(batch_size)
            observation = np.stack([x[0] for x in mini_batch], axis=0)
            rewards = np.stack([x[2] for x in mini_batch], axis=0)
            next_observation = np.stack([x[3] for x in mini_batch], axis=0)
            discount_factor = np.stack([x[4] for x in mini_batch], axis=0)
            errors = []
            if self._distribution is True:
                target_action_distribution_reward = self.get_main_dnn_action_and_value(observation)
                target_dist_reward = np.zeros(np.shape(target_action_distribution_reward))
                proj_distribution = self.projection_distribution(next_observation, batch_size, rewards,
                                                                 discount_factor)
                pr = pd.DataFrame(proj_distribution)
                pr.to_csv('proj_distribution.csv', index=False, encoding='cp949')

                for ind, sample in enumerate(mini_batch):
                    action = sample[1]
                    target_dist_reward[ind][action] = proj_distribution[ind]
                    errors.append(abs(self._cce(target_action_distribution_reward[ind][action],
                                                target_dist_reward[ind][action]).numpy()))
                target_action_reward = target_dist_reward
            else:
                _, current_reward, target_action_reward = self.get_main_dnn_action_and_value(observation)
                _, future_reward, _ = self.get_target_dnn_action_and_value(next_observation)
                for ind, sample in enumerate(mini_batch):
                    action = sample[1]
                    immediate_reward = sample[2]
                    target_action_reward[ind, action] = immediate_reward + discount_factor[ind] * future_reward[ind]
                    errors.append(abs(current_reward[ind] - target_action_reward[ind, action]))
            for j in range(batch_size):
                idx = idxs[j]
                self._replay_memory.update(idx, errors[j])
            self._main_dnn.fit(observation, target_action_reward, sample_weight=is_weights,
                               callbacks=[self._tensorboard_callback])
            print(f"batch_progress: {i}/{length}\r", end='')

    def projection_distribution(self, next_observation, batch_size, rewards, discount_factor):
        # if batch_size == 1:
        #     next_observation = np.stack([next_observation for _ in range(2)], axis=0)
        #     rewards = np.stack([rewards for i in range(2)], axis=0)
        #     discount_factor = [discount_factor, discount_factor]
        #     batch_size = 2

        next_mn_dist, next_tg_dist = self.get_target_dnn_action_and_value(next_observation)

        if self._double:
            dist = next_mn_dist
        else:
            dist = next_tg_dist
        next_action = dist * self._z
        next_action = next_action.sum(2).argmax(1)
        next_action = np.expand_dims(np.expand_dims(next_action, axis=1), axis=1)
        next_action = np.broadcast_to(next_action, (dist.shape[0], 1, dist.shape[2]))
        next_dist = np.take_along_axis(next_tg_dist, next_action, axis=1).squeeze(1)


        rewards = np.expand_dims(rewards, axis=1).repeat(np.shape(next_dist)[1], axis=1)
        discount_factor = np.expand_dims(discount_factor, axis=1).repeat(np.shape(next_dist)[1], axis=1)
        supports = np.expand_dims(self._z, axis=0).repeat(np.shape(next_dist)[0], axis=0)

        Tz = rewards + discount_factor * supports
        Tz = Tz.clip(min=self._v_min, max=self._v_max)
        b = (Tz - self._v_min) / self._dz
        i = np.floor(b).astype('int64')
        u = np.ceil(b).astype('int64')

        offset = np.expand_dims(np.linspace(0, (batch_size - 1) * self._num_support, batch_size).astype('int64'),
                                axis=1).repeat(self._num_support, axis=1)

        proj_dist = np.zeros(np.shape(next_dist)).reshape(-1)

        ml = np.bincount((i + offset).reshape(-1), (next_dist * (u - b)).reshape(-1), np.size(proj_dist))
        mu = np.bincount((u + offset).reshape(-1), (next_dist * (b - i)).reshape(-1), np.size(proj_dist))
        proj_dist += ml
        proj_dist += mu

        proj_dist = np.reshape(proj_dist, np.shape(next_dist))
        # if batch_size == 2:
        #     proj_dist = proj_dist[0]
        return proj_dist

    def convert_action_index_to_dict(self, action_index: int) -> Dict:
        """ Convert action index to dictionary form
        Args:
            action_index: index of action (0: sensing, 1 to (2^num_freq_channel-1)*max_num_unit_packet: tx_data_packet)
        Returns:
            action in dictionary form
                'type': 'sensing' or 'tx_data_packet',
                'freq_channel_list': list of frequency channels for data transmission
                'num_unit_packet': number of unit packets
        """
        if action_index == 0:
            action_dict = {'type': 'sensing'}
        else:
            num_unit_packet = (action_index - 1) // self._num_freq_channel_combination + 1
            freq_channel_combination_index = (action_index - 1) % self._num_freq_channel_combination
            freq_channel_list = self._freq_channel_combination[freq_channel_combination_index]
            action_dict = {'type': 'tx_data_packet', 'freq_channel_list': freq_channel_list,
                           'num_unit_packet': num_unit_packet}
        return action_dict

    def update_observation_history(self, action: Dict, observation: Dict):
        observation_type = observation['type']
        new_observation = np.zeros((self._num_freq_channel, 2))
        new_observation_length = 1
        if observation_type == 'sensing':
            sensed_power = observation['sensed_power']
            occupied_channel_list = [int(freq_channel) for freq_channel in sensed_power
                                     if sensed_power[freq_channel] > self._cca_thresh]
            new_observation[occupied_channel_list, 0] = 1
            new_observation_length = 1
        elif observation_type == 'tx_data_packet':
            tx_freq_channel_list = action['freq_channel_list']
            success_freq_channel_list = observation['success_freq_channel_list']
            failure_freq_channel_list = list(set(tx_freq_channel_list) - set(success_freq_channel_list))
            num_unit_packet = action['num_unit_packet']
            new_observation[failure_freq_channel_list, 1] = 1
            new_observation_length = num_unit_packet * self._sensing_unit_packet_length_ratio
        new_observation = np.broadcast_to(new_observation, (new_observation_length, self._num_freq_channel, 2))
        self._observation_history = np.concatenate((new_observation, self._observation_history),
                                                   axis=0)[:self._observation_history_length, ...]

    def get_reward(self, action: Dict, observation: Dict):
        observation_type = observation['type']
        reward = 0
        if observation_type == 'sensing':
            reward = 0
        elif observation_type == 'tx_data_packet':
            num_unit_packet = action['num_unit_packet']
            num_tx_packet = len(action['freq_channel_list'])
            num_success_packet = len(observation['success_freq_channel_list'])
            num_failure_packet = num_tx_packet - num_success_packet
            reward = (num_success_packet * self._unit_packet_success_reward +
                      num_failure_packet * self._unit_packet_failure_reward) * num_unit_packet
        return reward

    def model_save(self, scenario: int, modelNumber: int, episode: int, CSV: bool = False):
        path = 'savedModel/scenario_%d/model_%d/' % (scenario, modelNumber)
        tf.saved_model.save(self._main_dnn, path + "episode_%d" % episode)
        if CSV:
            path += 'result_%d_%d.csv' % (scenario, modelNumber)
            saveCSV(self._result, path)

    def tensorboard_save(self):
        log_dir = "logs/fit/{scenario}/{modelNumber}/".format(scenario=self._scenario, modelNumber=self._modelNumber) \
                  + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return TensorBoard(log_dir=log_dir)
