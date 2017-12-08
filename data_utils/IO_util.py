from scipy.io import wavfile


def get_audio_data(file_name:str):
    samplingRate, signal = wavfile.read(file_name)
    # print("readed sampling Rate",samplingRate)
    return samplingRate, signal

def write_audio_data(file_name,rate,data):
    wavfile.write(file_name,rate,data)