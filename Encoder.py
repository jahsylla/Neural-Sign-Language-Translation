import tensorflow as tf

# Features vector encoder  or CNN encoder
class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.gru1 = tf.keras.layers.GRU(self.enc_units,
                               return_sequences=True, recurrent_initializer='glorot_uniform')
        self.gru2 = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.gru1(x)
        output, state = self.gru2(x) # Attention la couche LSTM retourne 3 output contrairement à la couche GRU qui donne 2 output
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))



