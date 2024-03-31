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

def extractMFCCs(datafileNames):
    for fileName in datafileNames:
        fileName = DATA_STREET + fileName
        audioData, sampleRate = librosa.load(fileName)
        mfccs = librosa.feature.mfcc(y=audioData, sr=sampleRate)
        print(fileName, ": ",mfccs)

def extractZeroCrossingRates(datafileNames):
    for fileName in datafileNames:
        fileName = DATA_STREET + fileName
        audioData, _ = librosa.load(fileName)
        zcr = librosa.feature.zero_crossing_rate(y=audioData)
        print(fileName, ": ",zcr)

def extractMelSpectrogram(datafileNames):
    for fileName in datafileNames:
        fileName = DATA_STREET + fileName
        audioData, sampleRate = librosa.load(fileName)
        melSpectrogram = librosa.feature.melspectrogram(y=audioData, sr=sampleRate)
        print(fileName, ": ",melSpectrogram)

def extractChromaCene(datafileNames):
    for fileName in datafileNames:
        fileName = DATA_STREET + fileName
        audioData, sampleRate = librosa.load(fileName)
        chromaCene = librosa.feature.chroma_cens(y=audioData, sr=sampleRate)
        print(fileName, ": ",chromaCene)

def extractChromaCqt(datafileNames):
    for fileName in datafileNames:
        fileName = DATA_STREET + fileName
        audioData, sampleRate = librosa.load(fileName)
        chromaCqt = librosa.feature.chroma_cqt(y=audioData, sr=sampleRate)
        print(fileName, ": ",chromaCqt)

def extractChromaStft(datafileNames):
    for fileName in datafileNames:
        fileName = DATA_STREET + fileName
        audioData, sampleRate = librosa.load(fileName)
        chromaStft = librosa.feature.chroma_stft(y=audioData, sr=sampleRate)
        print(fileName, ": ",chromaStft)

def extractChromaVqt(datafileNames):
    for fileName in datafileNames:
        fileName = DATA_STREET + fileName
        audioData, sampleRate = librosa.load(fileName)
        chromaVqt = librosa.feature.chroma_vqt(y=audioData, sr=sampleRate, intervals="")
        print(fileName, ": ",chromaVqt)

def clusterAudioFiles(dataFileNames):
    fileNumberGroupDictionary = {}
    for i in range(10):
        label = str(i)
        fileNumberGroupDictionary[label] = []
        for file in dataFileNames:
            if file.startswith(label + "_"):
                fileNumberGroupDictionary[label].append(os.path.join(file))
    return fileNumberGroupDictionary


print(len(DataFileNames))
standardizeVolumes(DataFileNames)
eliminationList = noiseFilterKon(DataFileNames)
DataFileNames = eliminateFiles(eliminationList, DataFileNames)
print(len(DataFileNames))
fileNumberGroupDictionary = {}
fileNumberGroupDictionary = clusterAudioFiles(DataFileNames)
#extractMFCCs(DataFileNames)
#extractZeroCrossingRates(DataFileNames)
#extractMelSpectrogram(DataFileNames)
#extractChromaCene(DataFileNames)
#extractChromaCqt(DataFileNames)
#extractChromaStft(DataFileNames)