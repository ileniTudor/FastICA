def change_to_one_channel(signal):
    if len(signal.shape) == 2:
            one_chanel_signal = (signal[:, 0] + signal[:, 1]) / 2
            # one_chanel_signal = np.asarray(one_chanel_signal,dtype=np.int16)
            return one_chanel_signal