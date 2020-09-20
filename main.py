import preprocess as pps
from datetime import datetime
import os
import shutil

print("You will be asked for 9 different directories to store all of the information in. They must all be different. \n The information that needs directories is:")
print("Where the data to be preprocessed is currently stored")
print("Where to put the raw audio")
print("Where to put the raw MIDI")
print("Where to put the CSVMIDI")
print("Where to put the full time series")
print("Where to put the cut audio")
print("Where to put the cut and resampled audio")
print("Where to put the split audio")
print("Where to put the split \"MIDI\"")
beg = input("Part that is the same for all: ")
mainloc = beg + input("Directory of data: ")
if(not(os.path.isdir(mainloc))):
    os.mkdir(mainloc)
audioloc = beg + input("Directory audio files should go: ")
if(not(os.path.isdir(audioloc))):
    os.mkdir(audioloc)
midiloc = beg + input("Directory MIDI files should go: ")
if(not(os.path.isdir(midiloc))):
    os.mkdir(midiloc)
csvloc = beg + input("Directory CSV should go: ")
if(not(os.path.isdir(csvloc))):
    os.mkdir(csvloc)
timeseriesloc = beg + input("Directory timeseries should go: ")
if(not(os.path.isdir(timeseriesloc))):
    os.mkdir(timeseriesloc)
cutaudloc = beg + input("Directory the cut audio should go: ")
if(not(os.path.isdir(cutaudloc))):
    os.mkdir(cutaudloc)
cutsampaudloc = beg + input("Directory the cut and downsampled audio should go: ")
if(not(os.path.isdir(cutsampaudloc))):
    os.mkdir(cutsampaudloc)
splitaudloc = beg + input("Directory the split audio should go: ")
if(not(os.path.isdir(splitaudloc))):
    os.mkdir(splitaudloc)
splitmidloc = beg + input("Directory the split \"MIDI\" should go: ")
if(not(os.path.isdir(splitmidloc))):
    os.mkdir(splitmidloc)

print("Now you will be asked for the endings to use for each of the created data")
cutend = input("How to end the cut audio: ") + ".wav" #'cut.wav'
cutsampend = input("How to end the cut and resampled audio: ") + ".wav" #'cutandresample.wav'
timeseriesend = input("How to end the time series (will always be preceded by the size of the steps): ") + ".csv" #'chunk.csv' #will always be preceded by size
audsplitend = input("How to end the split audio (will always be preceded by the size, \"sec split\" and the number split): ") + ".wav" #'_.wav' #will always be preceded by size "secsplit" and number of split
tssplitend = input("How to end the split \"MIDI\" (will always be preceded by the size, \"sec split\" and the number split): ") + ".csv" #'_.csv' #will always be preceded by size "secsplit" and number of split

now = datetime.now()
start_time = now.strftime("%H:%M:%S")

#make sure we have .wav files not mp3
pps.batch_mp3_to_wav(mainloc)

#sort all files
pps.move_to_new_directory(mainloc, audioloc, '.wav')
pps.move_to_new_directory(mainloc, midiloc, '.midi')
pps.move_to_new_directory(mainloc, midiloc, '.mid')

#cut and downsample audio
pps.batch_cut_audio(audioloc, '.mid', midiloc, cutend)
pps.move_to_new_directory(audioloc, cutaudloc, cutend)
pps.batch_resample_audio(cutaudloc, 16000, cutend, cutsampend)
pps.move_to_new_directory(cutaudloc, cutsampaudloc, cutsampend)

#midi to time series
pps.batch_midi_to_csv(midiloc)
pps.move_to_new_directory(midiloc, csvloc, '.csv')
pps.batch_time_series(csvloc, 1024, timeseriesend)
pps.move_to_new_directory(csvloc, timeseriesloc, timeseriesend)

#split into smaller chunks
pps.batch_split_audio(cutsampaudloc, 120, cutsampend, audsplitend, 1024, 16000)
pps.move_to_new_directory(cutsampaudloc, splitaudloc, audsplitend)
pps.batch_split_time_series(timeseriesloc, 120, timeseriesend, tssplitend, 1024, 16000)
pps.move_to_new_directory(timeseriesloc, splitmidloc, tssplitend)

rem = input("Remove all but raw data and fully preprocessed data? y/n ")
if (rem == "y"):
    shutil.rmtree(csvloc)
    shutil.rmtree(timeseriesloc)
    shutil.rmtree(cutaudloc)
    shutil.rmtree(cutsampaudloc)
    direc = os.listdir(mainloc)
    if (direc == 0):
        os.rmdir(mainloc)


now = datetime.now()
end_time = now.strftime("%H:%M:%S")

print("Start time: ", start_time)
print("End time: ", end_time)
