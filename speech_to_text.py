import pyaudio
import wave
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
import os
import io

def record_and_transcribe():
    # Thông số ghi âm
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000
    CHUNK = 1024
    RECORD_SECONDS = 15
    WAVE_OUTPUT_FILENAME = "output.wav"
    GOOGLE_CLOUD_KEY_PATH = r'GoogleCloudKey_MyServiceAcct.json'
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording.")
    # Dừng ghi âm
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Recording saved as {WAVE_OUTPUT_FILENAME}")
    client = speech.SpeechClient.from_service_account_file(GOOGLE_CLOUD_KEY_PATH)
    with io.open(WAVE_OUTPUT_FILENAME, 'rb') as f:
        audio_data = f.read()
    audio_file = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        sample_rate_hertz=RATE,
        enable_automatic_punctuation=True,
        language_code='vi-VN'
    )
    response = client.recognize(
        config=config,
        audio=audio_file
    )
    transcripts = []
    for result in response.results:
        transcripts.append(result.alternatives[0].transcript)

    # Xóa file ghi âm
   # os.remove(WAVE_OUTPUT_FILENAME)
    print(f"Recording file {WAVE_OUTPUT_FILENAME} deleted.")
    print(transcripts)
    return transcripts