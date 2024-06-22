import os
import librosa
import pickle
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration



model_path = os.path.join('models')
imgFolder = os.path.join('static', 'assets')

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")


def Audio_Deepfake_Detection(sound_file):
    sample, sr = librosa.load(sound_file, sr = 16000)
    input_feature = processor(sample, sampling_rate=sr, return_tensors="pt").input_features
    images = model(input_feature).last_hidden_state
    automl = pickle.load(open('models/model.pkl','rb'))
    automl.eval()
    images = images[-1, 0:100 , :]
    print(images.shape)
    images = images.unsqueeze(0)
    images = images.unsqueeze(0)
    print(images.shape)
    images = images.float()
    outputs=automl(images)
    _,prediction=torch.max(outputs.data,1)
    print(outputs.data)
    print(_,prediction)
    y_pred=prediction.float()
    if y_pred==1:
        label = os.path.join(imgFolder, 'jh.png')
        label1='Real'
        
    elif y_pred==0:
        label1='Fake'
        label = os.path.join(imgFolder, '123.png')
    return label , label1 