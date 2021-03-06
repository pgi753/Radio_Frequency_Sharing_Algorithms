from player.wifi_dqn_player import DqnPlayer
import tensorflow as tf


tf.config.set_visible_devices([], 'GPU')

dnn_layers_list = [
    {'filters': 32,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'max_pool_size': (2, 1)},
    {'filters': 64,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'max_pool_size': (2, 1)},
    {'filters': 64,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'max_pool_size': (2, 1)},
    {'filters': 64,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'max_pool_size': (1, 1)}
]
# DQN_version: None(normal), 'double', 'dueling', 'prioritized'
identifier = 'p1'
player = DqnPlayer(identifier=identifier, max_num_unit_packet=3, observation_history_length=256,
                   sensing_unit_packet_length_ratio=24, unit_packet_success_reward=10,
                   unit_packet_failure_reward=-40, dnn_layers_list=dnn_layers_list, random_sensing_prob=0.5,
                   sensing_discount_factor=0.992, dnn_learning_rate=0.001, scenario=1500, modelNumber=3, dueling=True,
                   double=True, PER=False, noisy=False, distribution=True, v_max=250, v_min=-500, num_support=60, n_step=1)
player.connect_to_server('127.0.0.1', 8000)
player.train_dnn(num_episodes=100, replay_memory_size=1000, mini_batch_size=1000, initial_epsilon=1,
                 epsilon_decay=0.95, min_epsilon=0.1, dnn_epochs=1, progress_report=True, test_run_length=1000)
