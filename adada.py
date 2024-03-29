import os
import noisereduce
import soundfile
import librosa
from pydub import AudioSegment

VOLUME_THRESHOLD = -20
DROP_AUDIO_THRESHOLD = -10
DATA_SUM_THRESHOLD  = -3
NOISE_REDUCETION_ROUND = 4
DATA_STREET = './data/'

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

    #print("data sum of ", fileName, " is ", sum)
    return sum

def noiseFilterKon(dataFileNames):
    eliminatedFiles = []
    for fileName in dataFileNames:
        dataSum = noiseDetector(DATA_STREET + fileName)
        if dataSum < DROP_AUDIO_THRESHOLD:
            eliminatedFiles.append(fileName)
            print(fileName, " was dropped")

        elif dataSum < DATA_SUM_THRESHOLD:
            noiseKamKon(DATA_STREET + fileName)
            print(fileName, " was denoised")

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

    print(fileName, volume)
    return volume

def standardizeVolumes(dataFileNames):
    for fileName in dataFileNames:
        fileName = DATA_STREET + fileName
        volume = getVolumValue(fileName)
        if volume < VOLUME_THRESHOLD:
            volumZiadKon(fileName, volume)

print(len(DataFileNames))
eliminationList = noiseFilterKon(DataFileNames)
DataFileNames = eliminateFiles(eliminationList, DataFileNames)
print(len(DataFileNames))

standardizeVolumes(DataFileNames)