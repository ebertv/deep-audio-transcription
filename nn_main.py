from dataset import AudioClassificationDataset
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses, Sequential
import matplotlib.pyplot as plt
import os
import neptune.new as neptune


#hyperparameters (stored as a dict for neptune. This can include any information you want neptune to log for your experiment)
#PARAMS = {
#		"input_size"       : (156, 1024),
#		"batch_size"       : 16,
#		"epochs"           : 500,
#		"loss"             : "mse",
#        "optimizer"        : "Adam"
#		}

#class used for neptune logging
#class NeptuneMonitor(tf.keras.callbacks.Callback):
#	def on_epoch_end(self, epoch, logs={}):
#		neptune.send_metric("loss", epoch, logs["loss"])
#		neptune.send_metric("val_loss", epoch, logs["val_loss"])
#		neptune.send_metric("accuracy", epoch, logs["accuracy"])
#		neptune.send_metric("val_accuracy", epoch, logs["val_accuracy"])

#create neptune experiment and choose information to upload
#neptune.init("ebertv/tensectest", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NzkxZjBhNC1jYmU3LTQ5MWQtYjY5ZC0xNWEzMTRiYjQ4YjYifQ==")
#exp = neptune.create_experiment(params=PARAMS, source_files=["main.py"])
#neptune_callback = NeptuneMonitor()

run = neptune.init(project = 'ebertv/tensectest', api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NzkxZjBhNC1jYmU3LTQ5MWQtYjY5ZC0xNWEzMTRiYjQ4YjYifQ==", source_files=["main.py"])

run["algorithm"] = "BLSTM"

params = {
    "input_size"       : (156, 1024),
	"batch_size"       : 16,
	"epochs"           : 500,
	"loss"             : "mse",
    "optimizer"        : "Adam"
}
run["parameters"] = params

#class NeptuneMonitor(tf.keras.callbacks.Callback):
#	def on_epoch_end(self, epoch, logs={}):
#		neptune.send_metric("loss", epoch, logs["loss"])
#		neptune.send_metric("val_loss", epoch, logs["val_loss"])
#		neptune.send_metric("accuracy", epoch, logs["accuracy"])
#		neptune.send_metric("val_accuracy", epoch, logs["val_accuracy"])

class NeptuneMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        for metric_name, metric_value in logs.items():
            run['epoch/{}'.format(metric_name)].log(metric_value)

neptune_callback = NeptuneMonitor()

if __name__ == "__main__":

    print("Loading dataset...")

    #midi = []
    #audio = []
    #for file in os.listdir("../maestro_splits/10_sec/data_expressive/MIDI_10_sec"):
    #    if file.endswith(".csv"):
    #        midi.append(file)
    #for file in os.listdir("../maestro_splits/10_sec/data_expressive/MIDI_10_sec/in"):
    #    if file.endswith(".wav"):
    #       audio.append(file)

    #midi = sorted(midi)
    #audio = sorted(audio)

    #dataset = pd.DataFrame(audio)
    #dataset[1] = midi

    #os.remove("../maestro_splits/10_sec/data_expressive/MIDI_10_sec/dataset")
    #dataset.to_csv("../maestro_splits/10_sec/data_expressive/MIDI_10_sec/dataset", sep = ',', header=False, index=False)


    maestro = AudioClassificationDataset("/nfs/guille/eecs_research/soundbendor/transcription/maestro_splits/10_sec/data_expressive/MIDI_10_sec", "dataset")
    maestro.generate(1024)
    maestro.load(batch_size=16, output_size=131)


    print("Dataset loaded!") UPDATE DATASET SIZE TO 72011 from 7!!!!!

    sequence = maestro.train
    test = maestro.test
    validation = maestro.validate

    model = Sequential()

    model.add(layers.Bidirectional(layers.LSTM(1024, return_sequences=True, recurrent_dropout=0.4), merge_mode='sum'))
    model.add(layers.Bidirectional(layers.LSTM(512, return_sequences=True, recurrent_dropout=0.4), merge_mode='sum'))
    model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True, recurrent_dropout=0.4), merge_mode='sum'))
    model.add(layers.Bidirectional(layers.LSTM(131, return_sequences=True, recurrent_dropout=0.4), merge_mode='sum'))

    model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])
    #model.summary()

    checkpoint_path = "all_songs_right_number/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    model.fit(sequence, steps_per_epoch=maestro.train_steps, validation_data=validation, validation_steps=maestro.val_steps, epochs=500, callbacks=[neptune_callback, cp_callback])



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

    def eval_model():
        i = 0
        true = "outputs/test_set_song_true_"
        predicted = "outputs/test_set_song_predicted_"
        csv = ".csv"
        vel = "velocity_"
        wav = ".wav"
        for song in maestro.test:
            print("song[0] shape: ", song[0].shape)
            print("song[1] shape: ", song[1].shape)
            for j in range(16):
                eval_aud_mid(song[0][j], predicted+str(i)+csv, 0)
                eval_aud_mid(song[0][j], predicted+vel+str(i)+csv, 1)


                df = pd.DataFrame(data=song[1][j].numpy())
                df.to_csv(true+str(i)+csv)

                y = song[0][j]
                y = tf.squeeze(y)
                y = tf.reshape(y, [y.shape[0]*y.shape[1], 1])
                y = tf.cast(y, dtype=tf.float32)

                tf.io.write_file(true+str(i)+wav, tf.audio.encode_wav(y, 16000))

                print("Predicted song ", str(i))
                i += 1

    eval_model()


    #class Autoencoder(Model):
    #    def __init__(self):
    #        super(Autoencoder, self).__init__()

    #        self.forward_layer_1 = layers.LSTM(1024, activation='relu', return_sequences=True)
    #        self.backward_layer_1 = layers.LSTM(1024, activation='relu', return_sequences=True, go_backwards=True)
    #        self.forward_layer_2 = layers.LSTM(64, activation='relu', return_sequences=True)
    #        self.backward_layer_2 = layers.LSTM(64, activation='relu', return_sequences=True, go_backwards=True)
    #        self.forward_layer_3 = layers.LSTM(512, activation='relu', return_sequences=True)
    #        self.backward_layer_3 = layers.LSTM(512, activation='relu', return_sequences=True, go_backwards=True)
    #        self.forward_layer_4 = layers.LSTM(81, activation='relu', return_sequences=True)
    #        self.backward_layer_4 = layers.LSTM(81, activation='relu', return_sequences=True, go_backwards=True)
    #        self.forward_layer_5 = layers.LSTM(256, activation='relu', return_sequences=True)
    #        self.backward_layer_5 = layers.LSTM(256, activation='relu', return_sequences=True, go_backwards=True)
    #        self.forward_layer_6 = layers.LSTM(100, activation='relu', return_sequences=True)
    #        self.backward_layer_6 = layers.LSTM(100, activation='relu', return_sequences=True, go_backwards=True)
    #        self.forward_layer_7 = layers.LSTM(128, activation='relu', return_sequences=True)
    #        self.backward_layer_7 = layers.LSTM(128, activation='relu', return_sequences=True, go_backwards=True)
    #        self.forward_layer_8 = layers.LSTM(128, activation='relu', return_sequences=True)
    #        self.backward_layer_8 = layers.LSTM(128, activation='relu', return_sequences=True, go_backwards=True)
    #        self.forward_layer_9 = layers.LSTM(64, activation='relu', return_sequences=True)
    #        self.backward_layer_9 = layers.LSTM(64, activation='relu', return_sequences=True, go_backwards=True)
    #        self.forward_layer_10 = layers.LSTM(131, activation='relu', return_sequences=True)
    #        self.backward_layer_10 = layers.LSTM(131, activation='relu', return_sequences=True, go_backwards=True)





    #        self.encoder = tf.keras.Sequential([
    #            layers.Bidirectional(self.forward_layer_1, backward_layer=self.backward_layer_1, input_shape=(156, 1024), merge_mode = 'sum'),
    #            layers.Dense(1024)
    #            ])
    #        self.decoder  = tf.keras.Sequential([
    #            layers.Bidirectional(self.forward_layer_2, backward_layer=self.backward_layer_2, input_shape=(156, 1024), merge_mode = 'sum'),
    #            layers.Dense(131)
    #            ])

    #    def call(self, x):
    #        encoded = self.encoder(x)
    #        decoded = self.decoder(encoded)
    #        return decoded

    #autoencoder = Autoencoder()

    #autoencoder.compile(optimizer='adam', loss="mse", metrics=['accuracy'])
    #autoencoder.encoder.summary()
    #autoencoder.decoder.summary()

    #checkpoint_path = "aud_mid_velocity_2_layers/cp.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    #autoencoder.fit(sequence, steps_per_epoch=maestro.train_steps, validation_data=validation, validation_steps=maestro.val_steps, epochs=1000, callbacks=[neptune_callback, cp_callback])
    #autoencoder.eval(test, steps=maestro.test_steps)
    #history =.(sequence, sequence, epochs=250, validation_data=(validation, validation), batch_size=16, callbacks=[neptune_callback]) #when using this audio can just put sequence, not sequence and it will load both
    #need to pass steps_per_epoch, and validation_steps (train_steps, val_steps, test_steps, etc), dont need batch size because gave when loaded

    #model.evaluate also pass test_steps (look in discord)

    #yhat = autoencoder.predict(test, steps=maestro.test_steps, verbose=0)'''

