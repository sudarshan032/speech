import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from spafe.features.lfcc import lfcc
import numpy as np
import os

model_path = os.path.join('models')
imgFolder = os.path.join('static', 'assets')


def Voice_Liveness_Detection(sound_file):
    def lfcc_mine(sr, audio):
        lfccs = lfcc(audio, fs=sr, num_ceps=20)
        return lfccs

    audio, sr = librosa.load(sound_file, sr=None)
    feat = lfcc_mine(sr, audio)
    feat = feat.T
    train_extracted_1 = np.zeros((1, 20, 40))
    for i in range(len(feat)):
        shape = 40 - feat.shape[1]
        if shape < 0:
            shape = 0
        feat_pad = np.pad(feat, ((0, 0), (0, shape)), 'constant')
        train_extracted_1[:, :, :] = feat_pad[:, :40]
    feat_pad = train_extracted_1.reshape(1, 20, 40, 1)

    model = load_model(os.path.join(model_path, 'VLD_LFCC.h5'), compile=False)
    ans = model.predict(feat_pad)
    
    ans = ans[0][0]
    y_pred = tf.cast((ans > 0.5), tf.uint8)
    if y_pred == 0:
        label1 = 'Genuine Voice'
        label = os.path.join(imgFolder, 'v1.png')
    elif y_pred == 1:
        label1 = 'Spoofed Voice'
        label = os.path.join(imgFolder, 'ads.png')
    return label, label1
