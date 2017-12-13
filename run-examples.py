import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from data_utils.IO_util import get_audio_data
from data_utils.generate_data import generate_toy_signals
from decomposition.fastICA import ica
from data_utils.create_mixtures import mix_sources
from evaluation.calculate_person_correlation import calculate_pearson_correlation

sample_dir = os.path.abspath('audio_samples/s1/')

def run_for_toy_data():
    # generate random signal
    s1, s2, s3 = generate_toy_signals()

    X = np.c_[s1, s2, s3].T

    S = ica(X, max_iter=4000)
    plt.figure()
    for s in S:
        plt.plot(s)
    plot_mixture_sources_predictions(X,[s1,s2,s3],S)
    plt.show()


def plot_mixture_sources_predictions(X, original_sources, S):
    plt.figure()

    plt.subplot(3, 1, 1)
    for x in X:
        plt.plot(x)
    plt.title("mixtures")

    plt.subplot(3, 1, 2)
    for s in original_sources:
        plt.plot(s)
    plt.title("real sources")

    plt.subplot(3,1,3)
    for s in S:
        plt.plot(s)
    plt.title("predicted sources")
    plt.show()


def run_for_2observations(wrtie_sources_to_disk:bool = False):
    sampling_rate, mix1 = get_audio_data(os.path.join(sample_dir, "mix1.wav"))
    sampling_rate, mix2 = get_audio_data(os.path.join(sample_dir, "mix2.wav"))
    sampling_rate, real_source_1 = get_audio_data(os.path.join(sample_dir, "source1.wav"))
    sampling_rate, real_source_2 = get_audio_data(os.path.join(sample_dir, "source2.wav"))
    actual = mix_sources([real_source_1,real_source_2],False)
    X = mix_sources([mix1, mix2])
    S = ica(X, max_iter=1000)
    person_coeff = calculate_pearson_correlation(S,actual,True)
    print("Pearson correlation coefficient between the predicted sources and actual sources is",person_coeff)
    if wrtie_sources_to_disk:
        for i,s in enumerate(S):
            wavfile.write(os.path.join(sample_dir, 'predicted_source' + str(i) + '.wav'), sampling_rate, s)

    plot_mixture_sources_predictions(X,[real_source_1,real_source_2],S)


if __name__ == "__main__":
    # run_for_toy_data()
    run_for_2observations()
