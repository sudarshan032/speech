import librosa
import pandas as pd
import torch
from transformers import WhisperProcessor
from typing import Optional
from transformers.models.whisper.modeling_whisper import WhisperModel, WhisperEncoder
from transformers.models.whisper.configuration_whisper import WhisperConfig
import torch.nn as nn
import os
import numpy as np
from tensorflow.keras.models import load_model
import soundfile as sf

class WhisperWordClassifier(WhisperModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)
        self.post_init()

    def forward(
            self,
            input_features: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
        ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_outputs = self.encoder(
            input_features,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return encoder_outputs

class BiGRUAudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_units, num_layers):
        super(BiGRUAudioClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.bigru = nn.GRU(input_size=input_size, hidden_size=hidden_units, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.255)
        self.fc = nn.Linear(hidden_units * 2, num_classes)

    def forward(self, x):
        output, _ = self.bigru(x)
        output = self.dropout(output)
        last_hidden_state = torch.cat((output[:, -1, :self.hidden_units], output[:, 0, self.hidden_units:]), dim=1)
        output = self.fc(last_hidden_state)
        return output

# Load the necessary models
processor_small = WhisperProcessor.from_pretrained("openai/whisper-small")
model_small = WhisperWordClassifier.from_pretrained("openai/whisper-small")
model_cnn = load_model('models/model_noise_white_40.h5')

# Directory and mapping initialization
directory = r"test"
classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
class_to_idx = {i: c for i, c in enumerate(classes)}

df = pd.read_excel("mapping.xlsx")
mapping = df.set_index('FILE NAME').T.to_dict('list')

class_labels = ["very low", "low", "medium", "high"]

def preprocess_audio(file_path, target_shape=(20, 2000)):
    sample, sr = librosa.load(file_path, sr=16000)
    input_feature = librosa.feature.mfcc(y=sample, sr=sr, n_mfcc=20)
    if input_feature.shape[1] > target_shape[1]:
        input_feature = input_feature[:, :target_shape[1]]
    else:
        padding = target_shape[1] - input_feature.shape[1]
        input_feature = np.pad(input_feature, ((0, 0), (0, padding)), mode='constant')
    return np.expand_dims(input_feature, axis=-1)

def predict_label(model, audio_file):
    processed_audio = preprocess_audio(audio_file)
    processed_audio = np.expand_dims(processed_audio, axis=0)  # Add batch dimension
    prediction = model.predict(processed_audio)
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_label_index]
    return predicted_label

def dysarthric_asr(sound_file):
    try:
        # Save the uploaded file temporarily
        temp_filename = 'temp_audio.wav'
        sound_file.save(temp_filename)
        
        # CNN Model Prediction
        cnn_label = predict_label(model_cnn, temp_filename)
        
        # Whisper + BiGRU Model Prediction
        sample, sr = librosa.load(temp_filename, sr=16000)
        input_feature = processor_small(sample, sampling_rate=sr, return_tensors="pt").input_features
        images = model_small(input_feature).last_hidden_state
        
        saved_model = BiGRUAudioClassifier(768, 155, 256, 2)
        saved_model.load_state_dict(torch.load('models/dys_model', map_location=torch.device('cpu')))
        saved_model.eval()
        
        images = images[-1, 0:500, :]
        images = images.unsqueeze(0)
        images = images.float()
        
        outputs = saved_model(images)
        _, prediction = torch.max(outputs.data, 1)
        whisper_bigrucnn_label = mapping[class_to_idx[prediction.numpy()[0]]][0]
        
        os.remove(temp_filename)  # Clean up the temporary file
        
        return whisper_bigrucnn_label, cnn_label
    except sf.LibsndfileError as e:
        return str(e), None