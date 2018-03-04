from glob import glob
import os
import librosa
import re
import pandas as pd
import numpy as np

def get_elan_annotations(paths):
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep='\t', header=None)
        df = df[[3, 5, 7, 8]]
        df = df.rename(columns={3: 'start', 5: 'finish', 7: 'length', 8: 'label'})
        dfs.append(df)
    return dfs

def get_praat_annotations(paths):
    dfs = []
    for path in paths:
        with open(path) as f:
            annotation = [l.strip() for l in f.readlines()]
            
        indxs = [i for i, l in enumerate(annotation) if l == '"IntervalTier"']
        annotation = annotation[indxs[0] + 5:indxs[1] if len(indxs) > 1\
                                else len(annotation)]
        annotation_dicts = []

        for s, e, l in zip(annotation[0::3], annotation[1::3], annotation[2::3]):
            annotation_dicts.append({
                'start': float(s),
                'finish': float(e),
                'label': l.replace('"', ''),
                'length': float(e) - float(s)
            })
        df = pd.DataFrame(annotation_dicts)
        dfs.append(df)
    return dfs

def get_audios(paths):    
    res = []
    for path in paths:
        audio, sr = librosa.load(path)
        res.append(audio)
    return res, sr

def get_label_data(annotation, audio, label, sr):
    start_samples_indxs = librosa.time_to_samples(annotation[annotation.label ==\
                                                              label].start.values, sr)
    finish_samples_indxs = librosa.time_to_samples(annotation[annotation.label ==\
                                                               label].finish.values, sr)
    
    data = []
    for s, f in zip(start_samples_indxs, finish_samples_indxs):
        data.append(audio[np.arange(s, f)])
    return data

def get_unlabel_data(annotation, audio, labels, sr):
    start_samples_indxs = librosa.time_to_samples(\
                          annotation[annotation.label.isin(labels)].start.values, sr)
    finish_samples_indxs = librosa.time_to_samples(\
                           annotation[annotation.label.isin(labels)].finish.values, sr)
    finish_samples_indxs = np.hstack([[0], finish_samples_indxs])
    start_samples_indxs = np.hstack([start_samples_indxs, [len(start_samples_indxs) - 1]])
    
    data = []
    for s, f in zip(finish_samples_indxs, start_samples_indxs):
        data.append(audio[np.arange(s, f)])
    return data

def find_annotation_paths(ext='.txt'):
    return sorted(glob('../annotations/*' + ext))

def find_audio_from_annotations_paths(ann_ps):
    audio_paths = []
    for p in ann_ps:
        path = re.sub('annotations', 'audio', re.split('\.\w', p)[0])
        path = glob(path + '.*[m4a|3gpp]')[0]
        audio_paths.append(path)
    return audio_paths