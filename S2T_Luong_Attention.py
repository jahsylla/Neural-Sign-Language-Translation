import pandas as pd
import tensorflow as tf
import numpy as np
import os
import pickle
import time

# img_name from PHOENIX dataset 
l = os.listdir("train/")
img_name = [k.split(".")[0] for k in l]

# train
df_train = pd.read_csv("annotations/manual/PHOENIX-2014-T.train.corpus.csv", sep="|")
df_train = df_train[df_train['name'].isin(img_name)]

# val
df_val = pd.read_csv("annotations/manual/PHOENIX-2014-T.dev.corpus.csv", sep="|")
df_val = df_val[df_val['name'].isin(img_name)]

list_df = [df_train, df_val]
df = pd.concat(list_df)
df = df[["name", "translation"]]

dico_training = {k : f"<start> {v} <end>" for k,v in zip(df.name, df.translation)}
train_translation = list(dico_training.values())
train_img_name = list(dico_training.keys())

# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def TextVectozise(seq_text = ""):    
    # Choose the top 10000 words from the vocabulary
    top_k = 10000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(seq_text)
    train_seqs = tokenizer.texts_to_sequences(seq_text)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(seq_text)

    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    
    max_length = calc_max_length(train_seqs)

    return list(cap_vector), max_length, tokenizer



translation_vector, max_length, tokenizer = TextVectozise(train_translation)
img_to_translation_vector = {}
for img, cap in zip(train_img_name, translation_vector):
    img_to_translation_vector["train/" + img + ".npy"] = cap

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 1024

train_img_name = list(img_to_translation_vector.keys())
data_translation = list(img_to_translation_vector.values())

# Load the numpy files
def map_func(im_name_train, translation):
    img_tensor = np.load(im_name_train)
    return tf.constant(img_tensor/475, dtype=tf.float32), translation

dataset = tf.data.Dataset.from_tensor_slices((train_img_name, data_translation))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

for (image, label) in dataset.take(1):
    print(f'{image.numpy().shape} and {label.numpy().shape}')


class Encoder(tf.keras.Model):
    def __init__(self, units, batch_sz):
        super(Encoder, self).__init__()
        self.units = units
        self.batch_sz = batch_sz
        self.gru1 = tf.keras.layers.GRU(self.units,
                               return_sequences=True, recurrent_initializer='glorot_uniform')
        self.gru2 = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.gru1(x, initial_state = hidden)
        output, state = self.gru2(x)
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.units))


    
class Rnn_Global_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size,scoring_type):
        super(Rnn_Global_Decoder, self).__init__()
        

        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        
        
        self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
        
        self.wc = tf.keras.layers.Dense(units, activation='tanh')
        self.ws = tf.keras.layers.Dense(vocab_size)

        #For Attention
        self.wa = tf.keras.layers.Dense(units)
        self.wb = tf.keras.layers.Dense(units)
        
        #For Score 3 i.e. Concat score
        self.Vattn = tf.keras.layers.Dense(1)
        self.wd = tf.keras.layers.Dense(units, activation='tanh')

        self.scoring_type = scoring_type

        
    def call(self, sequence, features,hidden):
        
        # features : (64,49,256)
        # hidden : (64,512)
        
        embed = self.embedding(sequence)
        # embed ==> (64,1,256) ==> decoder_input after embedding (embedding dim=256)
       
        output, state = self.gru(embed)       
        #output :(64,1,512)

        score=0
        
        #Dot Score as per paper(Dot score : h_t (dot) h_s') (NB:just need to tweak gru units to 256)
        
        if(self.scoring_type=='dot'):
            xt=output #(64,1,512)
            xs=features #(256,49,64)  
            score = tf.matmul(xt, xs, transpose_b=True) 
               
          #score : (64,1,49)



        # General Score as per Paper ( General score: h_t (dot) Wa (dot) h_s')
        
        if(self.scoring_type=='general'):
            score = tf.matmul(output, self.wa(features), transpose_b=True)
          # score :(64,1,49)




        # Concat score as per paper (score: VT*tanh(W[ht;hs']))    
        #https://www.tensorflow.org/api_docs/python/tf/tile
        if(self.scoring_type=='concat'):
            tiled_features = tf.tile(features, [1,1,2]) #(64,49,512)
            tiled_output = tf.tile(output, [1,49,1]) #(64,49,512)

            concating_ht_hs = tf.concat([tiled_features,tiled_output],2) ##(64,49,1024)

            tanh_activated = self.wd(concating_ht_hs)
            score =self.Vattn(tanh_activated)
            #score :(64,49,1), but we want (64,1,49)
            score= tf.squeeze(score, 2)
            #score :(64,49)
            score = tf.expand_dims(score, 1)
          
          #score :(64,1,49)




        # alignment vector a_t
        alignment = tf.nn.softmax(score, axis=2)
        # alignment :(64,1,49)

        # context vector c_t is the average sum of encoder output
        context = tf.matmul(alignment, features)
        # context : (64,1,256)
        
        # Combine the context vector and the LSTM output
        
        output = tf.concat([tf.squeeze(context, 1), tf.squeeze(output, 1)], 1)
        # output: concat[(64,1,256):(64,1,512)] = (64,768)

        output = self.wc(output)
        # output :(64,512)

        # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
        logits = self.ws(output)
        # logits/predictions: (64,8239) i.e. (batch_size,vocab_size))

        return logits, state, alignment   
    

encoder = Encoder(units = units, batch_sz=64)
decoder = Rnn_Global_Decoder(embedding_dim = embedding_dim, units = units, vocab_size = len(tokenizer.word_index) + 1,scoring_type = "dot")


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# Define our metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

checkpoint_dir = 'training_checkpoints_Luong_attention2/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


loss_plot = []

@tf.function
def train_step(img_tensor, target, enc_hidden):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
#     hidden = encoder.initialize_hidden_state()

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

    with tf.GradientTape() as tape:
        features,enc_hidden = encoder(img_tensor, enc_hidden)

        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(dec_input, features, enc_hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)
          

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    #train_loss(loss)
    #train_accuracy(target, predictions)

    return total_loss


EPOCHS = 200
steps_per_epoch = len(train_img_name)//BATCH_SIZE

for epoch in range(EPOCHS):
    start = time.time()
    
    enc_hidden = encoder.initialize_hidden_state()
    total_loss_train = 0
    for (batch, (img_tensor, target)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(img_tensor, target, enc_hidden)
        total_loss_train += batch_loss
        
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))

    
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss_train / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))






