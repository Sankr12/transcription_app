import whisper
from IPython.display import Audio
import gradio as gr
import time

model = whisper.load_model("base")
model.device

Audio("Sports.wav")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("Sports.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

#detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected Language: {max(probs, key=probs.get)}")

#decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

#print the recognized text
print(result.text)

def transcribe(audio):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    #detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected Language: {max(probs, key=probs.get)}")

    #decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    #print the recognized text
    return result.text

demo = gr.Interface(    
    title='AI-based Audio Transcription, Recognition and Translation Web App',
    fn = transcribe,
    inputs=[gr.Audio(sources=["microphone", "upload"], type="filepath"),
    # gr.Audio(sources=["upload"], type="filepath", label="Upload an audio file (Supported formats: mp3, wav)")
    ],
    outputs=["textbox"],
    live=True
)

demo.launch()