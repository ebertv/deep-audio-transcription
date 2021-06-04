import csv
import pandas as pd
import py_midicsv
import os

def time_to_midicsv(input_file, output_file, ppq):
    # "0, 0, Header, format, num_tracks, division"
        # format: 0, 1, 2 (maestro and saarland are 1)
        # num_tracks: (Maestro and saarland are 2, 1->meta, 2->song)
        # division is ppq (some maestro is 384, some is 480. Saarland is 480)
    line = [""]*6
    line[0] = "0"
    line[1] = "0"
    line[2] = "Header"
    line[3] = "1"
    line[4] = "2" 
    line[5] = str(ppq)
    with open(output_file, 'a', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(line)

    # "1, 0, start_track"
    line = [""]*3
    line[0] = "1"
    line[1] = "0"
    line[2] = "Start_track"
    with open(output_file, 'a', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(line)

    # "1, 0, title_t, title 
        # title = usually a lot of \00000111\1\1\1\1\ etc 
        # saarland has and maestro doesn't 
    line = [""]*4
    line[0] = "1"
    line[1] = "0"
    line[2] = "Title_t"
    line[3] = "title"
    with open(output_file, 'a', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(line)

    # "1, 0, tempo, number"
        # number = microseconds per quarternote
        # maestro and saarland are 500000 = 120 bpm
    line = [""]*4
    line[0] = "1"
    line[1] = "0"
    line[2] = "Tempo"
    line[3] = "500000"
    with open(output_file, 'a', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(line)

    # "1, 0, time_sig, num, denom, click, notesQ"
        # num = numerator (4 for maestro and saarland)
        # denom = abs power of 2 of denom (2 = quarter, 3 = eighth, etc...) (2 for maestro and saarland)
        # click = midi ticks per metronome click (24 for maestro and saarland)
        # notesQ = 32nd notes in quarter note (8 for maestro and saarland)
    line = [""]*7
    line[0] = "1"
    line[1] = "0"
    line[2] = "Time_signature"
    line[3] = "4"
    line[4] = "2" 
    line[5] = "24"
    line[6] = "8"
    with open(output_file, 'a', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(line)

    # "1, 1 (last time in ticks + 1), end_track"
    line = [""]*3
    line[0] = "1"
    line[1] = "1"
    line[2] = "End_track"
    with open(output_file, 'a', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(line)

    # "2, 0, start_track"
    line = [""]*3
    line[0] = "2"
    line[1] = "0"
    line[2] = "Start_track"
    with open(output_file, 'a', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(line)

    # "2, 0, title_t, title"
        # some have this some don't in maestro, all do in saarland
    line = [""]*4
    line[0] = "2"
    line[1] = "0"
    line[2] = "Title_t"
    line[3] = "title"
    with open(output_file, 'a', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(line)

    # "2, 0, program_c, channel, program_num"
        # channel = 0
        # program_num = 0 (for piano)
        # saarland doesn't have this
    line = [""]*5
    line[0] = "2"
    line[1] = "0"
    line[2] = "Program_c"
    line[3] = "0"
    line[4] = "0" 
    with open(output_file, 'a', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(line)

    # "2, time_in_ticks, call, 0, note, velocity"
        # call = note_on_c, note_off_c, control_c
        # maestro doesn't have note_off_c, and only has note_on_c with velocity 0
            # this is probably the best thing to do for everything
        # this is through the second to last line. This is all of the data in the time series
    songfile = open(input_file, 'r')
    songreader = csv.reader(songfile)
    song = []
    for row in songreader:
        song.append(row)
    line = [""]*6
    line[0] = "2"
    line[3] = "0"
    pastrow = ["0.0"]*132
    for r in range(1, len(song)):
        for c in range(1, 132):
            #print("song[", r, "][", c, "] = ", song[r][c])
            #print("c: ", c)
            #print("pastrow[c]: ", pastrow[c])
            if(pastrow[c] != int(float(song[r][c])*127)):
                if(song[r][c] != "0.0" or (song[r][c] == "0.0" and pastrow[c] != "0.0")): 
                    line[1] = int(float(song[r][0])*0.064*2*ppq)
                    lastnum = line[1]
                    #print(song[r][0])
                    #print(line[1])
                    if(c <= 128):
                        line[2] = "Note_on_c"
                        line[4] = c-1
                    elif(c > 128):
                        line[2] = "Control_c"
                        if(c == 129):
                            line[4] = 64
                        elif(c == 130):
                            line[4] = 67
                        elif(c == 131):
                            line[4] = 66
                    line[5] = int(float(song[r][c])*127)
                    pastrow[c] = line[5]
                    #print("line[5]: ", pastrow[c])
                    with open(output_file, 'a', newline='') as result_file:
                        wr = csv.writer(result_file, dialect='excel')
                        wr.writerow(line)
    
    # "2, last_time_in_ticks + 1, end_track"
    line = [""]*3
    line[0] = "2"
    line[1] = str(int(lastnum)+1) #change so last num is numseconds in ticks
    line[2] = "End_track"
    with open(output_file, 'a', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(line)

    # "0, 0, end_of_file"
    line = [""]*3
    line[0] = "0"
    line[1] = "0"
    line[2] = "End_of_file"
    with open(output_file, 'a', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(line)

def midicsv_to_midi(input_file, output_file):
    midi_object = py_midicsv.csv_to_midi(open(input_file))
    with open(output_file, "wb") as outfile:
        midi_writer = py_midicsv.FileWriter(outfile)
        midi_writer.write(midi_object)

def timeseries_to_midi(input_file, middle_file, output_file):
    time_to_midicsv(input_file, middle_file, 384)
    print("To_midi_csv done")
    midicsv_to_midi(middle_file, output_file)
    os.remove(middle_file)

#batch codes

timeseries_to_midi("../tests/velocity/midi_audio_diff_1not5.csv", "../tests/velocity/middle.csv", "../tests/velocity/midi_audio_diff_1not5.midi")