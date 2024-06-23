import librosa
import pandas as pd
import torch
from transformers import WhisperProcessor
from typing import Optional
from transformers.models.whisper.modeling_whisper import WhisperModel, WhisperEncoder
from transformers.models.whisper.configuration_whisper import WhisperConfig
import torch.nn as nn
import os


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
    
    def __init__(self,input_size, num_classes, hidden_units, num_layers):
        super(BiGRUAudioClassifier, self).__init__()
        self.input_size = input_size 
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.bigru = nn.GRU(input_size=input_size, hidden_size=hidden_units, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.255)
        # self.fc = nn.Linear(hidden_units, num_classes)
        self.fc = nn.Linear(hidden_units * 2, num_classes)

    def forward(self, x):
        # x: (batch_size, sequence_length, num_features)

        # Pass the input through the bi-GRU layers
        output, _ = self.bigru(x)
        output = self.dropout(output)
        # Extract the last hidden state (concatenate forward and backward hidden states)
        last_hidden_state = torch.cat((output[:, -1, :self.hidden_units], output[:, 0, self.hidden_units:]), dim=1)
        # Apply the fully connected layer for classification
        output = self.fc(last_hidden_state)

        return output
    

processor_small = WhisperProcessor.from_pretrained("openai/whisper-small")
model_small = WhisperWordClassifier.from_pretrained("openai/whisper-small")


def dysarthric_asr(sound_file):
    directory = r"test"
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    class_to_idx = {i: c for i, c in enumerate(classes)}
    # MAPPING
    df = pd.read_excel("mapping.xlsx")
    mapping = df.set_index('FILE NAME').T.to_dict('list')

    sample, sr = librosa.load(sound_file, sr = 16000)
    input_feature = processor_small(sample, sampling_rate=sr, return_tensors="pt").input_features
    images = model_small(input_feature).last_hidden_state
    # saved_model='bigru_whispersmall_original.pkl'
    # with open(saved_model, 'rb') as file:
    #     automl=CPU_unpickler(file).load()
    saved_model = BiGRUAudioClassifier(768, 155, 256, 2)
    saved_model.load_state_dict(torch.load('models/dys_model',map_location=torch.device('cpu')))
    saved_model.eval()
    # automl = pickle.load(open('bigru_whispersmall_original.pkl','rb'))
    # print(images.shape)
    # images = images.unsqueeze(1)
    images = images[-1, 0:500 , :]
    images = images.unsqueeze(0)
    images = images.float()
    print(images.shape)
    outputs=saved_model(images)
    _,prediction=torch.max(outputs.data,1)
    # print(prediction.numpy()[0])
    # print(outputs.data)
    label=mapping[class_to_idx[prediction.numpy()[0]]][0]
    return label 