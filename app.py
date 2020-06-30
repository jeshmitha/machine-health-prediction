from flask import Flask, request
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import librosa
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def handle_request():
    # return "Flask Server & Android are Working Successfully"

    audio_file = request.files["file"]

    # print(audio_file)
    audio_file.save("audio_posted")
    # return "audio_file"

    # @app.route('/predict',methods=["POST"])
    # def predict():

    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 14):
        header += f' mfcc{i}'
    header = header.split()
    df = pd.DataFrame(columns=header)

    # get audio file and save it
    # audio_file = request.files["file"]
    file_name = "audio_posted"  # str(random.randint(0,100000))
    # audio_file.save(file_name)

    myaudio = AudioSegment.from_file(file_name, "wav")
    chunk_length_ms = 4000  # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms)

    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        # print("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")

        # extract features
        y, sr = librosa.load(chunk_name, mono=True)
        # print(librosa.display.waveplot(y=y, sr=sr))
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # filename=filename.replace(" ",".")
        to_append = f'{chunk_name} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'

        to_append_series = pd.Series(to_append.split(), index=df.columns)
        df = df.append(to_append_series, ignore_index=True)

        # remove the audio chunk file
        os.remove(chunk_name)

    # typecast
    df[['chroma_stft']] = df[['chroma_stft']].astype(float)
    df[['rmse']] = df[['rmse']].astype(float)
    df[['spectral_centroid']] = df[['spectral_centroid']].astype(float)
    df[['spectral_bandwidth']] = df[['spectral_bandwidth']].astype(float)
    df[['rolloff']] = df[['rolloff']].astype(float)
    df[['zero_crossing_rate']] = df[['zero_crossing_rate']].astype(float)
    df[['mfcc1']] = df[['mfcc1']].astype(float)
    df[['mfcc2']] = df[['mfcc2']].astype(float)
    df[['mfcc3']] = df[['mfcc3']].astype(float)
    df[['mfcc4']] = df[['mfcc4']].astype(float)
    df[['mfcc5']] = df[['mfcc5']].astype(float)
    df[['mfcc6']] = df[['mfcc6']].astype(float)
    df[['mfcc7']] = df[['mfcc7']].astype(float)
    df[['mfcc8']] = df[['mfcc8']].astype(float)
    df[['mfcc9']] = df[['mfcc9']].astype(float)
    df[['mfcc10']] = df[['mfcc10']].astype(float)
    df[['mfcc11']] = df[['mfcc11']].astype(float)
    df[['mfcc12']] = df[['mfcc12']].astype(float)
    df[['mfcc13']] = df[['mfcc13']].astype(float)

    # make predictions
    X = df[['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1',
            'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
            'mfcc13']]

    trained_model = pickle.load(open('xgboost_trained_model.pkl', 'rb'))
    Y_pred = trained_model.predict(X)

    # send back the predictions in json format
    output = ""
    n = len(Y_pred)
    for i in range(n):
        output = output + str(Y_pred[i])

    # print(output)
    return output

#
# app.run(host="0.0.0.0", port=5000)
