import librosa
import os
import pickle
import numpy as np

model_path = os.path.join('models')
imgFolder = os.path.join('static', 'assets')

def INFANT_CRY_CLASSIFICATION(sound_file):

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