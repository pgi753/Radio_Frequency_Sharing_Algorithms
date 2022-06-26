from player.player import Player
import numpy as np


class WiFiCSMAPlayer(Player):
    def __init__(self, identifier, contention_window_size, num_unit_packet):
        super(WiFiCSMAPlayer, self).__init__(identifier)
        self._contention_window_size = contention_window_size
        self._num_unit_packet = num_unit_packet
        self._freq_channel_list = []
        self._primary_channel = 0
        self._back_off: int = np.random.randint(1, self._contention_window_size)
        self._cca_thresh = -70

    def run(self, execution_number):
        action = {'type': 'sensing'}
        self._freq_channel_list = self.operator_info['freq channel list']
        self._primary_channel = self._freq_channel_list[0]
        for it in range(execution_number):
            observation = self.step(action)
            observation_type = observation['type']
            if observation_type == 'sensing':
                sensed_power = observation['sensed_power']
                sensed_power = {int(freq_channel): sensed_power[freq_channel] for freq_channel in sensed_power}
                primary_sensed_power = sensed_power[self._primary_channel]
                if primary_sensed_power > self._cca_thresh:
                    action = {'type': 'sensing'}
                else:
                    self._back_off -= 1
                    if self._back_off <= 0:
                        freq_channel_list = []
                        for ch in sensed_power:
                            if sensed_power[ch] <= self._cca_thresh:
                                freq_channel_list.append(ch)
                        action = {'type': 'tx_data_packet', 'freq_channel_list': freq_channel_list,
                                  'num_unit_packet': self._num_unit_packet}
                    else:
                        action = {'type': 'sensing'}
            elif observation_type == 'tx_data_packet':
                self._back_off = np.random.randint(1, self._contention_window_size)
                action = {'type': 'sensing'}
