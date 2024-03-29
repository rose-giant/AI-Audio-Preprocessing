import os
import noisereduce
import soundfile
import librosa

DROP_AUDIO_THRESHOLD = -10
DATA_SUM_THRESHOLD  = -3
NOISE_REDUCETION_ROUND = 4
DATA_STREET = './recordings/'

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

#print(len(DataFileNames))
eliminationList = noiseFilterKon(DataFileNames)
#print(eliminationList)
eliminateFiles(eliminationList, DataFileNames)
#print(len(DataFileNames))