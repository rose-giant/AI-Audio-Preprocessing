import os
import noisereduce
import soundfile
import librosa
from pydub import AudioSegment

VOLUME_THRESHOLD = -20
DROP_AUDIO_THRESHOLD = -10
DATA_SUM_THRESHOLD  = -3
NOISE_REDUCETION_ROUND = 4
DATA_STREET = './recordings/'

DICT_NUM_KEY = "number"
DICT_NAME_KEY = "fileName"
DICT_MFCC_KEY = "mfcc"
DICT_ZCR_KEY = "zcr"
DICT_MELSPECT_KEY = "melspectogram"
DICT_CHROMACENE_KEY = "chromaCene"
DICT_CHROMACQT_KEY = "chromaCqt"
DICT_CHROMASTFT_KEY = "chromaStft"

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

    #print(fileName, volume)
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
        mfccs = librosa.feature.mfcc(y=audioData, sr=sampleRate)
        item[DICT_MFCC_KEY] = mfccs

    return fileDictionary

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

def extractChromaCene(fileDictionary):
    for item in fileDictionary:
        fileName = DATA_STREET + item.get(DICT_NAME_KEY)
        audioData, sampleRate = librosa.load(fileName)
        chromaCene = librosa.feature.chroma_cens(y=audioData, sr=sampleRate)
        item[DICT_CHROMACENE_KEY] = chromaCene

    return fileDictionary

def extractChromaCqt(fileDictionary):
    for item in fileDictionary:
        fileName = DATA_STREET + item.get(DICT_NAME_KEY)
        audioData, sampleRate = librosa.load(fileName)
        chromaCqt = librosa.feature.chroma_cqt(y=audioData, sr=sampleRate)
        item[DICT_CHROMACQT_KEY] = chromaCqt

    return fileDictionary

def extractChromaStft(fileDictionary):
    for item in fileDictionary:
        fileName = DATA_STREET + item.get(DICT_NAME_KEY)
        audioData, sampleRate = librosa.load(fileName)
        chromaStft = librosa.feature.chroma_stft(y=audioData, sr=sampleRate)
        item[DICT_CHROMASTFT_KEY] = chromaStft

    return fileDictionary

def extractChromaVqt(datafileNames):
    for fileName in datafileNames:
        fileName = DATA_STREET + fileName
        audioData, sampleRate = librosa.load(fileName)
        chromaVqt = librosa.feature.chroma_vqt(y=audioData, sr=sampleRate, intervals="")
        print(fileName, ": ",chromaVqt)

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

print(len(DataFileNames))
standardizeVolumes(DataFileNames)
eliminationList = noiseFilterKon(DataFileNames)
DataFileNames = eliminateFiles(eliminationList, DataFileNames)
print(len(DataFileNames))
fileDictionary = clusterAudioFiles(DataFileNames)
fileDictionary = extractMFCCs(fileDictionary)
fileDictionary = extractZeroCrossingRates(fileDictionary)
fileDictionary = extractMelSpectrogram(fileDictionary)
fileDictionary = extractChromaCene(fileDictionary)
# fileDictionary = extractChromaCqt(fileDictionary)
# fileDictionary = extractChromaStft(fileDictionary)
print(fileDictionary.get(DICT_CHROMACENE_KEY))