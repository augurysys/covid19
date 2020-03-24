# imports:

import pyaudio
import wave


# default parameters:

CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "record.wav"


# constant parameters:

FORMAT = pyaudio.paInt16
CHUNK = 1024


# functions:

def run(f_s=RATE, duration=RECORD_SECONDS, channels=CHANNELS, filename=WAVE_OUTPUT_FILENAME):
    
    # audio recorder:
    audio = pyaudio.PyAudio()

    # start recording:
    stream = audio.open(format=FORMAT, channels=channels, rate=f_s, input=True, frames_per_buffer=CHUNK)
    frames = []

    # recording:
    print("recording...")
    for i in range(0, int(f_s / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    # stop recording:
    print("finished recording")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # saving:
    print("saving...")
    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(channels)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(f_s)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    print("finished saving")
    
    return