import os
import io
import pickle
from spafe.features.lfcc import lfcc
from keras.models import load_model
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.modeling_whisper import WhisperModel, WhisperEncoder
from transformers.models.whisper.configuration_whisper import WhisperConfig
from typing import Optional, Tuple, Union
import librosa 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import pandas as pd

model_path = os.path.join('models')
imgFolder = os.path.join('static', 'assets')

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model.config.forced_decoder_ids = None

# model_small = WhisperModel.from_pretrained("openai/whisper-small")
processor_small = WhisperProcessor.from_pretrained("openai/whisper-small")

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
model = WhisperWordClassifier.from_pretrained("openai/whisper-base")
model_small = WhisperWordClassifier.from_pretrained("openai/whisper-small")



# Define the Basic Block for ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define the ResNet architecture
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Function to create ResNet-50 model
def resnet50(num_classes=2):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

# Example usage:
# model = resnet50(num_classes=1000) # Change num_classes to match your specific classification task.
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


# define gru model class
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


def fun1(wave_file):
    data, sr = librosa.load(wave_file)
    
    def cqtspec(audio,sr,min_freq=30,octave_resolution=14):
        max_frequency= sr/2
        num_freq = round(octave_resolution * np.log2(max_frequency/min_freq))
        # step_length = int(pow(2, int(np.ceil(np.log2(0.04 * sr)))) / 2)
        cqt_spectrogram = np.abs(librosa.cqt(data,sr=sr,fmin=min_freq,bins_per_octave=octave_resolution,n_bins=num_freq))
        return cqt_spectrogram

    def cqhc(audio,sr,min_freq=30,octave_resolution=14,num_coeff=20):
        cqt_spectrogram=np.power(cqtspec(audio,sr,min_freq,octave_resolution),2)
        num_freq=np.shape(cqt_spectrogram)[0] 
        ftcqt_spectrogram=np.fft.fft(cqt_spectrogram,2*num_freq-1,axis=0)
        absftcqt_spectrogram=abs(ftcqt_spectrogram)
        # spectral_component=np.real(np.fft.ifft(absftcqt_spectrogram,axis=0)[0:num_freq,:])
        pitch_component=np.real(np.fft.ifft(ftcqt_spectrogram/(absftcqt_spectrogram+1e-14),axis=0)[0:num_freq,:])
        coeff_indices=np.round(octave_resolution*np.log2(np.arange(1,num_coeff+1))).astype(int)
        audio_cqhc=pitch_component[coeff_indices,:]
        return audio_cqhc
    
    #Input shape - (20,290,1)
    feat = cqhc(data,sr,min_freq=30,octave_resolution=14,num_coeff=20)
    
    #Pad = 290
    shape = 290 - feat.shape[1]
    feat_pad = np.pad(feat, ((0,0), (0,shape)), 'constant')
    feat_pad = feat_pad.reshape(1, 20, 290, 1)
    
    #Load model
    
    model = load_model('models/emotion_h5_file.h5')
    ans = model(feat_pad)
    print("loaded")
    output = np.argmax(ans)
    if output == 0:
        labels = os.path.join(imgFolder, 'angry.gif')
        labels1='Angry'
    elif output == 1:
        labels = os.path.join(imgFolder, 'happy.gif')
        labels1='Happy'
    elif output == 2:
        labels = os.path.join(imgFolder, 'neutral.gif')
        labels1='Neutral'
    elif output == 3:
        labels = os.path.join(imgFolder, 'sad.gif')
        labels1='Sad'
    elif output == 4:
        labels = os.path.join(imgFolder, 'surprise.gif')
        labels1='Surprise'
    print(labels)
    return labels, labels1

def fun2(sound_file):

    audio,sr=librosa.load(sound_file)
    mfccs=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=13)
    mfccs_processed=np.mean(mfccs.T,axis=0)
    test_extracted = mfccs_processed
    test_extracted=test_extracted.reshape((1,-1))
    classifier_mod = pickle.load(open('models/knnpickle_file','rb'))
    y_pred=classifier_mod.predict(test_extracted)
    if y_pred==0:
        label = os.path.join(imgFolder, 'normal.png')
        label1='Normal cry'
        
    elif y_pred==1:
        label1='Pathology cry'
        label = os.path.join(imgFolder, 'pathological.png')
    return label , label1



def fun3(sound_file):
    directory = r"test"
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    class_to_idx = {i: c for i, c in enumerate(classes)}
    # MAPPING
    df = pd.read_excel("mapping.xlsx")
    mapping = df.set_index('FILE NAME').T.to_dict('list')

    sample, sr = librosa.load(sound_file, sr = 16000)
    input_feature = processor_small(sample, sampling_rate=sr, return_tensors="pt").input_features
    print("fgh")
    images = model_small(input_feature).last_hidden_state
    print("fgh")
    # saved_model='bigru_whispersmall_original.pkl'
    # with open(saved_model, 'rb') as file:
    #     automl=CPU_unpickler(file).load()
    saved_model = BiGRUAudioClassifier(768, 155, 256, 2)
    saved_model.load_state_dict(torch.load('models/dys_model',map_location=torch.device('cpu')))
    saved_model.eval()
    # automl = pickle.load(open('bigru_whispersmall_original.pkl','rb'))
    print(images.shape)
    # images = images.unsqueeze(1)
    images = images[-1, 0:500 , :]
    images = images.unsqueeze(0)
    images = images.float()
    print(images.shape)
    outputs=saved_model(images)
    _,prediction=torch.max(outputs.data,1)
    print(prediction.numpy()[0])
    # print(outputs.data)
    label=mapping[class_to_idx[prediction.numpy()[0]]][0]
    return label, label 

def fun4(sound_file):
    def lfcc_mine(sr,audio):
        lfccs=lfcc(audio,fs=sr,num_ceps=20,nfft=512,win_len=0.025,win_hop=0.010)
        return lfccs
    audio,sr=librosa.load(sound_file)
    feat=lfcc_mine(sr,audio)
    feat=feat.T
    train_extracted_1 = np.zeros((1, 20, 40))
    for i in range(len(feat)):
        shape = 40 - feat.shape[1]
        if shape < 0:
                shape = 0
        feat_pad = np.pad(feat, ((0,0), (0,shape)), 'constant')
        train_extracted_1[:, :, :] = feat_pad[:, :40]
    feat_pad = train_extracted_1.reshape(1, 20, 40, 1)
    print("Helnk sm ms< mc")
    model = load_model('models/VLD_LFCC.h5', compile=False)
    ans = model.predict(train_extracted_1)
    print("Helnk sm ms< mc")
    ans = model(feat_pad)
    print(ans[0][0])
    
    ans=ans[0][0]
    y_pred = tf.cast((ans > 0.5), tf.uint8)
    if y_pred==0:
        label1='Genuine Voice'
        label = os.path.join(imgFolder, 'v1.png')
        
    elif y_pred==1:
        label1='Spoofed Voice'
        label = os.path.join(imgFolder, 'ads.png')
    return label , label1 


def fun5(sound_file):
    sample, sr = librosa.load(sound_file, sr = 16000)
    input_feature = processor(sample, sampling_rate=sr, return_tensors="pt").input_features
    # print("fgh")
    images = model(input_feature).last_hidden_state
    # print("fgh")
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