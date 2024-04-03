import os
import noisereduce
import soundfile
import librosa
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler

VOLUME_THRESHOLD = -20
DROP_AUDIO_THRESHOLD = -10
DATA_SUM_THRESHOLD  = -3
NOISE_REDUCETION_ROUND = 4
DATA_STREET = './data/'

DICT_NUM_KEY = "number"
DICT_NAME_KEY = "fileName"
DICT_MFCC_KEY = "mfcc"
DICT_ZCR_KEY = "zcr"
DICT_MELSPECT_KEY = "melspectogram"
DICT_CHROMACENE_KEY = "chromaCene"
DICT_CHROMACQT_KEY = "chromaCqt"
DICT_CHROMASTFT_KEY = "chromaStft"
DICT_CHROMAVQT_KEY = "chromaVqt"
DICT_SPECTRALROLLOFF_KEY = "spectralRollOff"

DataFileNames = os.listdir(DATA_STREET)

def noiseKamKon(fileName):
    for i in range (0, NOISE_REDUCETION_ROUND):
        data, sampleRate = soundfile.read(fileName)
        denoisedData = noisereduce.reduce_noise(y=data, sr=sampleRate)
        soundfile.write(file=fileName, data=denoisedData, samplerate=sampleRate)

def noiseDetector(fileName):
    data, _ = librosa.load(fileName)
    sum = 0
    for d in data:
        sum += d
    return sum

def noiseFilterKon(dataFileNames):
    eliminatedFiles = []
    for fileName in dataFileNames:
        dataSum = noiseDetector(DATA_STREET + fileName)
        if dataSum < DROP_AUDIO_THRESHOLD:
            eliminatedFiles.append(fileName)
            print(fileName, " dropped")

        elif dataSum < DATA_SUM_THRESHOLD:
            noiseKamKon(DATA_STREET + fileName)
            print(fileName, " reduced")

    return eliminatedFiles

def eliminateFiles(eliminationList, dataFileNames):
    for el in eliminationList:
        dataFileNames.remove(el)

    return dataFileNames

def volumZiadKon(fileName, currentVolume):
    audio = AudioSegment.from_file(fileName)
    modifiedAudio = audio + (VOLUME_THRESHOLD - currentVolume)
    modifiedAudio.export(fileName, format="wav")

def getVolumValue(fileName):
    audio = AudioSegment.from_file(fileName)
    volume = audio.dBFS
    return volume

def standardizeVolumes(dataFileNames):
    for fileName in dataFileNames:
        fileName = DATA_STREET + fileName
        volume = getVolumValue(fileName)
        if volume < VOLUME_THRESHOLD:
            volumZiadKon(fileName, volume)

def extractMFCCs(fileDictionary):
    for item in fileDictionary:
        file = DATA_STREET + item.get(DICT_NAME_KEY)
        audioData, sampleRate = librosa.load(file)
        mfccs = librosa.feature.mfcc(y=audioData, sr=sampleRate, n_mfcc=13)
        item[DICT_MFCC_KEY] = mfccs

    return fileDictionary

def plotMFCCs(fileDictionary):
    for item in fileDictionary:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(item.get(DICT_MFCC_KEY), x_axis='time', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC')
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficient')
        plt.show()

def extractZeroCrossingRates(fileDictionary):
    for item in fileDictionary:
        fileName = DATA_STREET + item.get(DICT_NAME_KEY)
        audioData, _ = librosa.load(fileName)
        zcr = librosa.feature.zero_crossing_rate(y=audioData)
        item[DICT_ZCR_KEY] = zcr

    return fileDictionary

def extractMelSpectrogram(fileDictionary):
    for item in fileDictionary:
        fileName = DATA_STREET + item.get(DICT_NAME_KEY)
        audioData, sampleRate = librosa.load(fileName)
        melSpectrogram = librosa.feature.melspectrogram(y=audioData, sr=sampleRate)
        item[DICT_MELSPECT_KEY] = melSpectrogram

    return fileDictionary

def extractSpectralRollOff(fileDictionary):
    for item in fileDictionary:
        fileName = DATA_STREET + item.get(DICT_NAME_KEY)
        audioData, sampleRate = librosa.load(fileName)
        chromaVqt = librosa.feature.spectral_rolloff(y=audioData, sr=sampleRate)
        item[DICT_SPECTRALROLLOFF_KEY] = chromaVqt
    
    return fileDictionary

def extractChromaCene(fileDictionary):
    for item in fileDictionary:
        fileName = DATA_STREET + item.get(DICT_NAME_KEY)
        audioData, sampleRate = librosa.load(fileName)
        chromaCene = librosa.feature.chroma_cens(y=audioData, sr=sampleRate)
        # chromaCeneNorm = librosa.util.normalize(chromaCene, axis=0)
        # chromaCeneNormRow = librosa.util.normalize(chromaCeneNorm, axis=1)
        item[DICT_CHROMACENE_KEY] = chromaCene

    return fileDictionary

def extractChromaCqt(fileDictionary):
    for item in fileDictionary:
        fileName = DATA_STREET + item.get(DICT_NAME_KEY)
        audioData, sampleRate = librosa.load(fileName)
        chromaCqt = librosa.feature.chroma_cqt(y=audioData, sr=sampleRate)
        chromaCqtNorm = librosa.util.normalize(chromaCqt, axis=0)
        chromaCqtNormRow = librosa.util.normalize(chromaCqt, axis=1)
        item[DICT_CHROMACQT_KEY] = chromaCqtNormRow

    return fileDictionary

def extractChromaStft(fileDictionary):
    for item in fileDictionary:
        fileName = DATA_STREET + item.get(DICT_NAME_KEY)
        audioData, sampleRate = librosa.load(fileName)
        chromaStft = librosa.feature.chroma_stft(y=audioData, sr=sampleRate)
        chromaStftNorm = librosa.util.normalize(chromaStft, axis=0)
        chromaStftNormRow = librosa.util.normalize(chromaStft, axis=1)
        item[DICT_CHROMASTFT_KEY] = chromaStftNormRow

    return fileDictionary

def extractChromaVqt(fileDictionary):
    for item in fileDictionary:
        fileName = DATA_STREET + item.get(DICT_NAME_KEY)
        audioData, sampleRate = librosa.load(fileName)
        chromaVqt = librosa.feature.chroma_vqt(y=audioData, sr=sampleRate, intervals="equal")
        chromaVqtNorm = librosa.util.normalize(chromaVqt, axis=0)
        chromaVqtNormRow = librosa.util.normalize(chromaVqt, axis=1)
        item[DICT_CHROMAVQT_KEY] = chromaVqtNormRow

    return fileDictionary

def clusterAudioFiles(dataFileNames):
    dictList = []
    for i in range(10):
        for file in dataFileNames:
            if file.startswith(str(i) + "_"):
                fileNumberGroupDictionary = {}
                fileNumberGroupDictionary[DICT_NUM_KEY] = str(i)
                fileNumberGroupDictionary[DICT_NAME_KEY] = file
                dictList.append(fileNumberGroupDictionary)

    return dictList

def normalizeFeatures(fileDictionaries, dictionaryKey):
    for f in fileDictionaries:
        params = f[dictionaryKey]
        scaler = MinMaxScaler()
        normalizedParams = scaler.fit_transform(params)
        f[dictionaryKey] = normalizedParams

    return fileDictionaries

def normalizeChromaFeatures(fileDictionary, featureTag):
    for f in fileDictionary:
        for e in f[featureTag]:
            for i in e:
                max = numpy.max(e)
                min = numpy.min(e)
                if i < min:
                    min = i
                if i > max:
                    max = i
            
            for j in e:
                j = (j - min) / (max - min)
                j = float(j)
                print(j)
            print("doneeeeeeeeeeeeeeeee")
    # for f in fileDictionary:
    #     for i in range (0, len(f[featureTag])):
    #         f[featureTag][i] = (max - f[featureTag][i]) / max - min
    #         print(f[featureTag][i])
    
    return fileDictionary

standardizeVolumes(DataFileNames)
eliminationList = noiseFilterKon(DataFileNames)
DataFileNames = eliminateFiles(eliminationList, DataFileNames)
fileDictionary = {}
fileDictionary = clusterAudioFiles(DataFileNames)
fileDictionary = extractMFCCs(fileDictionary)
fileDictionary = extractZeroCrossingRates(fileDictionary)
fileDictionary = extractMelSpectrogram(fileDictionary)
fileDictionary = extractChromaCene(fileDictionary)
fileDictionary = extractChromaCqt(fileDictionary)
fileDictionary = extractChromaStft(fileDictionary)
fileDictionary = extractChromaVqt(fileDictionary)
fileDictionary = extractSpectralRollOff(fileDictionary)
# plotMFCCs(fileDictionary)
fileDictionary = normalizeFeatures(fileDictionary, DICT_MFCC_KEY)
fileDictionary = normalizeFeatures(fileDictionary, DICT_ZCR_KEY)
fileDictionary = normalizeFeatures(fileDictionary, DICT_MELSPECT_KEY)
fileDictionary = normalizeFeatures(fileDictionary, DICT_SPECTRALROLLOFF_KEY)

fileDictionary = normalizeFeatures(fileDictionary, DICT_CHROMACENE_KEY)
fileDictionary = normalizeFeatures(fileDictionary, DICT_CHROMACQT_KEY)
fileDictionary = normalizeFeatures(fileDictionary, DICT_CHROMASTFT_KEY)
fileDictionary = normalizeFeatures(fileDictionary, DICT_CHROMAVQT_KEY)
#print(fileDictionary[1])

# hiddenStatesNum = 10
# model = hmm.GaussianHMM(n_components=hiddenStatesNum, covariance_type= 'diag')
# model.fit(fileDictionary)
# decodedStates = model.predict("./data2/7_yweweler_34.wav")
# print(decodedStates.reshape(-1, 1))