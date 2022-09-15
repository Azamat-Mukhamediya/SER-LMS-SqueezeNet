import os
import librosa
import pandas as pd


def get_audio_emodb(folder_path):
    classes = ['angry', 'anxiety', 'boredom',
               'disgust', 'happy', 'neutral', 'sadness']
    audio_list = []

    for filename in os.listdir(folder_path):
        audio, sample_rate = librosa.load(
            folder_path + '/' + filename, sr=16000)

        if filename[-6:-5] == 'W':
            emotion = 'angry'
        elif filename[-6:-5] == 'A':
            emotion = 'anxiety'
        elif filename[-6:-5] == 'L':
            emotion = 'boredom'
        elif filename[-6:-5] == 'E':
            emotion = 'disgust'
        elif filename[-6:-5] == 'F':
            emotion = 'happy'
        elif filename[-6:-5] == 'N':
            emotion = 'neutral'
        elif filename[-6:-5] == 'T':
            emotion = 'sadness'

        label_id = classes.index(emotion)
        audio_with_label = [audio, label_id]
        audio_list.append(audio_with_label)

    output = audio_list
    return output


def get_audio_iemocap(folder_path):

    classes = ['neu', 'hap', 'sad', 'ang']
    audio_list = []

    file = pd.read_csv('iemocap_full_dataset.csv', usecols=[
                       'method', 'emotion', 'path'])

    for ind, data in file.iterrows():
        if (data[0] == 'impro'):
            if(data[1] in classes or data[1] == 'exc'):

                filepath = folder_path + '/' + data[2]
                audio, sample_rate = librosa.load(filepath, sr=16000)

                if(data[1] == 'exc'):
                    label_id = classes.index('hap')
                else:
                    label_id = classes.index(data[1])

                audio_with_label = [audio, label_id]
                audio_list.append(audio_with_label)

    output = audio_list

    return output


def get_audio_savee(folder_path):
    classes = ['neutral', 'happy', 'sad',
               'angry', 'fear', 'disgust', 'surprise']
    audio_list = []

    for filename in os.listdir(folder_path):
        audio, sample_rate = librosa.load(
            folder_path + '/' + filename, sr=16000)

        if filename[-8:-6] == '_a':
            emotion = 'angry'

        elif filename[-8:-6] == '_d':
            emotion = 'disgust'

        elif filename[-8:-6] == '_f':
            emotion = 'fear'

        elif filename[-8:-6] == '_h':
            emotion = 'happy'

        elif filename[-8:-6] == '_n':
            emotion = 'neutral'

        elif filename[-8:-6] == 'sa':
            emotion = 'sad'

        elif filename[-8:-6] == 'su':
            emotion = 'surprise'

        label_id = classes.index(emotion)

        audio_with_label = [audio, label_id]
        audio_list.append(audio_with_label)

    output = audio_list
    return output


def get_data(db, folder_path):
    if db == 'emodb':
        audios = get_audio_emodb(folder_path)
        num_classes = 7  # number of classes
    elif db == 'iemocap':
        audios = get_audio_iemocap(folder_path)
        num_classes = 4  # number of classes
    elif db == 'savee':
        audios = get_audio_savee(folder_path)
        num_classes = 7  # number of classes

    return audios, num_classes
