from dataset import AudioClassificationDataset
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses, Sequential
import matplotlib.pyplot as plt
import os
import neptune
import csv





if __name__ == "__main__":

    print("Loading dataset...")


    maestro = AudioClassificationDataset("/nfs/guille/eecs_research/soundbendor/transcription/maestro_splits/10_sec/data_expressive/MIDI_10_sec", "dataset")
    maestro.generate(1024)
    maestro.load(batch_size=16, output_size=131)


    print("Dataset loaded!")

    sequence = maestro.train
    test = maestro.test
    validation = maestro.validate

    print(maestro._train_size)
    print(maestro._test_size)
    print(maestro._val_size)

    model = Sequential()

    model.add(layers.Bidirectional(layers.LSTM(1024, return_sequences=True, recurrent_dropout=0.4), merge_mode='sum'))
    model.add(layers.Bidirectional(layers.LSTM(512, return_sequences=True, recurrent_dropout=0.4), merge_mode='sum'))
    model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True, recurrent_dropout=0.4), merge_mode='sum'))
    model.add(layers.Bidirectional(layers.LSTM(131, return_sequences=True, recurrent_dropout=0.4), merge_mode='sum'))

    model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])
    #model.summary()

    checkpoint_path = "no_autoencoder_real_expressive/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    #model.fit(sequence, steps_per_epoch=maestro.train_steps, validation_data=validation, validation_steps=maestro.val_steps, epochs=500, callbacks=[cp_callback])

    #loss, acc = model.evaluate(sequence, steps=maestro.test_steps, verbose=1)
    #print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

    model.load_weights(checkpoint_path)

    loss, acc = model.evaluate(sequence, steps=maestro.test_steps, verbose=1)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

    #model.summary()

    def eval_audio(input_file, output_file):
        x, _ = tf.audio.decode_wav(tf.io.read_file(input_file))
        x = tf.squeeze(x)
        input_length = x.shape[0]
        #x = tf.pad(x, [[0, 1024 - (input_length % 1024)]])
        x = tf.reshape(x, [-1, 1024])
        x = tf.expand_dims(x, 0)
        y = model.predict(x)
        y = tf.squeeze(y)
        y = tf.reshape(y, [y.shape[0]*y.shape[1], 1])
        tf.io.write_file(output_file, tf.audio.encode_wav(y, 16000))

    def eval_midi(input_file, output_file):
        #read and process csv file
        csv_data = []
        with open(input_file) as f:
            reader = csv.reader(f, delimiter=',')
            next(reader, None)      #skip headers
            #iterate through csv
            for row in reader:
                foo = []
                for val in row[1:]:
                    foo.append(float(val))
                csv_data.append(np.asarray(foo, dtype=np.float32))

        #print(csv_data)

        value = tf.reshape(tf.convert_to_tensor(csv_data, dtype=tf.float32), [1, 156, 131])
        #value = tf.reshape(value, [156, 131])

        #value = tf.convert_to_tensor(csv_data, dtype=tf.float32)

        yhat = model.predict(value)

        yhat = np.where(yhat > 0.1, yhat, 0)
        #print(yhat.shape)
        #print(yhat)
        yhat1 = yhat.reshape((156, 131))
        df1 = pd.DataFrame(data=yhat1)
        #print(df1)
        df1.to_csv(output_file)

    def eval_aud_mid(input_file, output_file, velocity):
        #x, _ = tf.audio.decode_wav(tf.io.read_file(input_file))
        x = tf.squeeze(input_file)
        #input_length = x.shape[0]
        #x = tf.pad(x, [[0, 1024 - (input_length % 1024)]])
        x = tf.reshape(x, [-1, 1024])
        x = tf.expand_dims(x, 0)
        print("x shape: ", x.shape)
        yhat = model.predict(x)
        print("yhat shape: ", yhat.shape)

        if(velocity):
            yhat = np.where(yhat >= 0.1, yhat, 0)
        #yhat = np.where(yhat > 1, 1, yhat)
        else:
            yhat = np.where(yhat > 0.1, 1, 0)
        #print(yhat.shape)
        #print(yhat)
        yhat1 = yhat.reshape((156, 131))
        df1 = pd.DataFrame(data=yhat1)
        #print(df1)
        df1.to_csv(output_file)

    def eval_mid_aud(input_file, output_file):
        #read and process csv file
        csv_data = []
        with open(input_file) as f:
            reader = csv.reader(f, delimiter=',')
            next(reader, None)      #skip headers
            #iterate through csv
            for row in reader:
                foo = []
                for val in row[1:]:
                    foo.append(float(val))
                csv_data.append(np.asarray(foo, dtype=np.float32))

        #print(csv_data)

        value = tf.reshape(tf.convert_to_tensor(csv_data, dtype=tf.float32), [1, 156, 131])
        #value = tf.reshape(value, [156, 131])

        #value = tf.convert_to_tensor(csv_data, dtype=tf.float32)

        y = model.predict(value)

        y = tf.squeeze(y)
        y = tf.reshape(y, [y.shape[0]*y.shape[1], 1])
        tf.io.write_file(output_file, tf.audio.encode_wav(y, 16000))



    def eval_model():
        i = 0
        true = "outputs/test_set_song_true_"
        predicted = "outputs/test_set_song_predicted_"
        csv = ".csv"
        vel = "velocity_"
        wav = ".wav"
        for song in test:
            print("song[0] shape: ", song[0].shape)
            print("song[1] shape: ", song[1].shape)
            for j in range(16):
                #eval_aud_mid(song[0][j], predicted+str(i)+csv, 0)
                #eval_aud_mid(song[0][j], predicted+vel+str(i)+csv, 1)


                #df = pd.DataFrame(data=song[1][j].numpy())
                #df.to_csv(true+str(i)+csv)

                y = song[0][j]
                y = tf.squeeze(y)
                y = tf.reshape(y, [y.shape[0]*y.shape[1], 1])
                y = tf.cast(y, dtype=tf.float32)

                tf.io.write_file(true+str(i)+wav, tf.audio.encode_wav(y, 16000))

                print("Predicted song ", str(i))
                i += 1


    #eval_model()


    #eval_aud_mid("/nfs/guille/eecs_research/soundbendor/transcription/7234_songs/in/MIDI-Unprocessed_043_PIANO043_MID--AUDIO-split_07-06-17_Piano-e_1-03_wav--2_cutandresample_9point984secsplit0percentoverlap9_.wav", "/nfs/guille/eecs_research/soundbendor/transcription/midi_audio_diff_1not5.csv")

    #eval_mid_aud("/nfs/guille/eecs_research/soundbendor/transcription/7234/MIDI-Unprocessed_043_PIANO043_MID--AUDIO-split_07-06-17_Piano-e_1-03_wav--2_1024chunk_9point984secsplit0percentoverlap9_csv", "/nfs/guille/eecs_research/soundbendor/transcription/audio_midi_test_diff_same.wav")



