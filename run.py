import os

import numpy as np
from scipy.io import wavfile

from data_utils.IO_util import get_audio_data
from data_utils.generate_data import generate_toy_signals
from decomposition.fastICA import ica
from data_utils.create_mixtures import mix_sources
from evaluation.calculate_person_correlation import calculate_pearson_correlation
from evaluation.plotter import plot_mixture_sources_predictions

sample_dir = os.path.abspath('audio_samples/s1/')


def run_for_toy_data():
    # generate random signal
    s1, s2, s3 = generate_toy_signals()

    X = np.c_[s1, s2, s3].T

    S = ica(X, max_iter=4000)

    plot_mixture_sources_predictions(X, [s1, s2, s3], S)


def run_for_2observations(wrtie_sources_to_disk: bool = False):
    sampling_rate, mix1 = get_audio_data(os.path.join(sample_dir, "mix1.wav"))
    sampling_rate, mix2 = get_audio_data(os.path.join(sample_dir, "mix2.wav"))
    sampling_rate, real_source_1 = get_audio_data(os.path.join(sample_dir, "source1.wav"))
    sampling_rate, real_source_2 = get_audio_data(os.path.join(sample_dir, "source2.wav"))
    actual = mix_sources([real_source_1, real_source_2], False)
    X = mix_sources([mix1, mix2])
    S = ica(X, max_iter=1000)
    person_coeff = calculate_pearson_correlation(S, actual, False)
    print("Pearson correlation coefficient between the predicted sources and actual sources is", person_coeff)
    if wrtie_sources_to_disk:
        for i, s in enumerate(S):
            wavfile.write(os.path.join(sample_dir, 'predicted_source' + str(i) + '.wav'), sampling_rate, s)

    plot_mixture_sources_predictions(X, [real_source_1, real_source_2], S)


def run_ica(signal1, signal2):
    actual = mix_sources([signal1, signal2], apply_noise=False)
    X = mix_sources([signal1, signal2], apply_noise=True)
    S = ica(X,tolerance=1e-4, max_iter=1000,verbose=False)
    return calculate_pearson_correlation(S, actual, False)


def run_for_test_dataset():
    data_set_dir = "./audio_samples/test_dataset/"
    samples_number = len(list(os.walk(data_set_dir))) - 1
    pearson_coeff_list = np.zeros(shape=samples_number)

    for i, dir in enumerate(os.listdir(data_set_dir)):
        files = os.listdir(data_set_dir + dir)
        if len(files) == 2:
            samp_rate, signal1 = get_audio_data(file_name=data_set_dir + dir + "/" + files[0])
            samp_rate, signal2 = get_audio_data(file_name=data_set_dir + dir + "/" + files[1])
            min_length = min(len(signal1), len(signal2))
            if len(signal1) != min_length:
                signal1 = np.resize(signal1, min_length)
            else:
                signal2 = np.resize(signal2, min_length)
            mean_pers_coeff = run_ica(signal1, signal2)
            pearson_coeff_list[i] = mean_pers_coeff
        else:
            continue

    mean_pearson_ICA = np.mean(pearson_coeff_list)
    min_ICA = np.min(pearson_coeff_list)
    max_ICA = np.max(pearson_coeff_list)
    print("mean_pearson_ICA", mean_pearson_ICA)
    print("min ica", min_ICA)
    print("max ica", max_ICA)


if __name__ == "__main__":
    # run_for_toy_data()
    run_for_2observations(False)
    # run_for_test_dataset()
