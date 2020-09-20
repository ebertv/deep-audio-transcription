import preprocess as pps

from datetime import datetime

mainloc = r'/mnt/c/users/victo/documents/research/datasets/saarland/rawdata'
audioloc = r'/mnt/c/users/victo/documents/research/datasets/saarland/audio'
midiloc = r'/mnt/c/users/victo/documents/research/datasets/saarland/midi'
csvloc = r'/mnt/c/users/victo/documents/research/datasets/saarland/csv'
timeseriesloc = r'/mnt/c/users/victo/documents/research/datasets/saarland/time_series'
cutaudloc = r'/mnt/c/users/victo/documents/research/datasets/saarland/cutaudio'
cutsampaudloc = r'/mnt/c/users/victo/documents/research/datasets/saarland/cutdsampaud'
splitaudloc = r'/mnt/c/users/victo/documents/research/datasets/saarland/split/audio'
splitmidloc = r'/mnt/c/users/victo/documents/research/datasets/saarland/split/time_series'

cutend = 'cut.wav'
cutsampend = 'cutandresample.wav'
timeseriesend = 'chunk.csv' #will always be preceded by size
audsplitend = '_.wav' #will always be preceded by size "secsplit" and number of split
tssplitend = '_.csv' #will always be preceded by size "secsplit" and number of split

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

now = datetime.now()
end_time = now.strftime("%H:%M:%S")

print("Start time: ", start_time)
print("End time: ", end_time)
