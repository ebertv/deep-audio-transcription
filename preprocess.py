import os
import py_midicsv
from mido import MidiFile, second2tick
import sox
import shutil
import math
import csv
import pandas as pd
import numpy as np
from pydub import AudioSegment
import ffmpeg

#convert a midi file to a csv file
def midi_to_csv(infile):
    csv_string = py_midicsv.midi_to_csv(infile)
    if infile.endswith(".midi"):
        newfile = infile.replace('.midi', '.csv')
    if infile.endswith(".mid"):
        newfile = infile.replace('.mid', '.csv')
    else:
        print("Not a MIDI file")
        return
    f = open(newfile, 'w+')
    for i in range (len(csv_string)):
        f.write(csv_string[i])

#convert mp3 to wav
def mp3_to_wav(infile):
    newfile = infile.replace('mp3', 'wav')
    outfile = AudioSegment.from_mp3(infile)
    outfile.export(newfile, format="wav")

#cut an audio file to length of a midi file
def cut_audio(audio_file, midi_file, new_ending):
    mid = MidiFile(midi_file)
    newfile = audio_file
    newfile = newfile.replace('.wav', '_' + new_ending)
    tfm = sox.Transformer()
    tfm.trim(0,mid.length)
    tfm.build(audio_file, newfile)

#downsample an audio file (must be done after cutting)
def resample_audio(audio_path, sampling_rate, ending, new_ending):
    tfn = sox.Transformer()
    tfn.rate(sampling_rate)
    remix_dictionary = {1:[1,2]}
    tfn.remix(remix_dictionary)
    tfn.build(audio_path, audio_path.replace(ending, new_ending))

#go from midicsv to time series
def time_series(input_file, output_file, size):
    song = pd.read_csv(input_file, names = ["Track", "Tick", "Call", "Channel", "Note", "Velocity", "0"])
    chunk = [0.000]*132
    chunk[0] = "Time"
    chunk[129] = "64 Control_c"
    chunk[130] = "67 Control_c"
    chunk[131] = "66 Control_c"
    for i in range(0,128):
        chunk[i+1] = str(i)+" Note_on_c"
    with open(output_file, 'a', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(chunk)
    chunk = [0.000]*132
    num_rows = len(song.axes[0])-1
    row = 0
    ppq = song.loc[row, "Velocity"]
    step = (ppq*size)/8000
    for i in np.arange (0, song.loc[num_rows-1, "Tick"], step):
        nexti = i+step
        cur_tick = song.loc[row, "Tick"]
        chunk[0] = i/(2*ppq) 
        while i <= cur_tick and cur_tick <= nexti:
            cur_call = song.loc[row, "Call"]
            if(cur_call == " Note_on_c"):
                cur_note = song.loc[row, "Note"]
                cur_vel = song.loc[row, "Velocity"]
                chunk[int(cur_note)+1]=cur_vel/127
            if(cur_call == " Note_off_c"):
                cur_note = song.loc[row, "Note"]
                cur_vel = 0
                chunk[int(cur_note)+1]=cur_vel/127
            if(cur_call == " Control_c"):
                cur_note = song.loc[row, "Note"]
                cur_vel = song.loc[row, "Velocity"]
                place = 0
                if(cur_note == 64):
                    place = 129
                elif(cur_note == 67):
                    place = 130
                elif(cur_note == 66):
                    place = 131
                chunk[place]=cur_vel/127
            row+=1
            cur_tick = song.loc[row, "Tick"]
        with open(output_file, 'a', newline='') as result_file:
            wr = csv.writer(result_file, dialect='excel')
            wr.writerow(chunk)

#snap time to nearest time series length in ms
def time_to_size(time, size, sampling_rate):
    z = size/sampling_rate
    a = time % z
    b = time - a
    c = b + 0.064
    if (time-b > c-time):
        time = c
    else:
        time = b
    return time

#split the audio file in to segments of lensplit seconds
def split_audio(in_file, lensplit, size, sampling_rate, new_ending):
    lensplit = time_to_size(lensplit, size, sampling_rate)
    lensplitnop = str(lensplit).replace('.', 'point')
    song = AudioSegment.from_file(in_file)
    song_len = song.duration_seconds*16000
    i = 0
    nexti = 16000*lensplit
    j = 1
    in_copy = in_file
    while(nexti < song_len):
        out_end = '_'+lensplitnop+'secsplit'+str(j)
        out_file = in_copy.replace('.wav', out_end + new_ending)
        out = ffmpeg.input(in_file)
        out = ffmpeg.filter_(out, 'atrim', start_pts = i, end_pts = nexti)
        out = ffmpeg.output(out, out_file)
        out.run() 
        i = nexti
        nexti += 16000*lensplit
        j += 1
    out_end = '_'+lensplitnop+'secsplit'+str(j)
    out_file = in_copy.replace('.wav', out_end + new_ending)
    out = ffmpeg.input(in_file)
    out = ffmpeg.filter_(out, 'atrim', start_pts = i)
    out = ffmpeg.output(out, out_file)
    out.run()

#split csv file in to chunks of lensplit seconds
def split_time_series(in_file, lensplit, size, sampling_rate, new_ending):
    lensplit = time_to_size(lensplit, size, sampling_rate)
    lensplitnop = str(lensplit).replace('.', 'point')
    x = int(lensplit/0.064)
    i = 1
    nexti = i+x
    j = 1
    in_copy = in_file
    data = []
    with open(in_file, 'r') as f:
        read_file = csv.reader(f)
        for row in read_file:
            data.append(row)
    while(nexti < len(data)):
        out_end = '_'+lensplitnop+'secsplit'+str(j)
        out_file = in_copy.replace('.csv', out_end + new_ending)
        copy_csv(i, nexti, data, out_file)
        i = nexti
        nexti += x
        j += 1
    out_end = '_'+lensplitnop+'secsplit'+str(j)
    out_file = in_copy.replace('.csv', out_end + new_ending) 
    copy_csv(i, nexti, data, out_file)

#helper code
def copy_csv(first_line_to_include, first_line_to_exclude, data, out_file):
    if(first_line_to_exclude < first_line_to_include):
        print('Please make exclude greater than include')
        return
    with open(out_file, 'w+') as f:
        out = csv.writer(f)
        out.writerow(data[0])
        for i in range(first_line_to_include, first_line_to_exclude):
            if(i < len(data)):
                out.writerow(data[i])
            else:
                break


#batch codes

#batch mp3_to_wav
def batch_mp3_to_wav(directory):
    count = 0
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith("mp3"):
                count += 1
                mp3_to_wav(filepath)
                print("mp3 to wav: ", count) 

#batch midi_to_csv
def batch_midi_to_csv(directory):
    count = 0
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith("midi") or filepath.endswith(".mid"):
                count += 1
                midi_to_csv(filepath)
                print("MIDI to CSV: ", count)

#batch cut_audio (all audio files in one directory, all midi in one directory. If subdirectories, set up same way)
def batch_cut_audio(audio_directory, midi_ending, midi_directory, new_ending):
    count = 0
    for subdir, dirs, files in os.walk(audio_directory):
        for filename in files:
            filepath = subdir + os.sep + filename 
            if filepath.endswith(".wav"):
                midiversion = filepath
                midiversion = midiversion.replace('.wav', midi_ending)
                midiversion = midiversion.replace(audio_directory, midi_directory)
                cut_audio(filepath, midiversion, new_ending)
                count += 1
                print ("Cut: ", count)

#batch resample_audio (must be done after cut for now)
def batch_resample_audio(directory, sampling_rate, ending, new_ending):
    count = 0
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            filepath = subdir + os.sep + filename 
            if filepath.endswith("_cut.wav"):
                count += 1
                resample_audio(filepath, sampling_rate, ending, new_ending)
                print("Resample: ", count)

#batch time_series (must be in their own folder)
def batch_time_series(csv_directory, size, new_ending):
    count = 0
    for subdir, dirs, files in os.walk(csv_directory):
        for filename in files:
            filepath = subdir + os.sep + filename 
            if filepath.endswith(".csv"):
                count+=1
                ending = '_' + str(size) + new_ending
                newfile = filepath.replace(".csv", ending)
                time_series(filepath, newfile, size)
                print("Time series: ", count)

#batch split_audio
def batch_split_audio(directory, length, ending, new_ending, size, sampling_rate):
    count = 0
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(ending):
                count += 1
                split_audio(filepath, length, size, sampling_rate, new_ending)
                print("Split audio: ", count) 

#batch split_csv
def batch_split_time_series(directory, length, ending, new_ending, size, sampling_rate):
    count = 0
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(ending):
                count += 1
                split_time_series(filepath, length, size, sampling_rate, new_ending,)
                print("Split time_series: ", count) 

#move (not copy) all with ending to new directory
def move_to_new_directory(current_loc, new_loc, ending):
    count = 0
    for subdir, dirs, files in os.walk(current_loc):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(ending):
                count+=1
                shutil.move(filepath, new_loc)
                print("Move: ", count)