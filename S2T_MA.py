
import pandas as pd
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pickle
import collections
import time

# Feel free to change these parameters according to your system's configuration
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 512
units = 1024



# img_name from PHOENIX dataset 

# train
l = os.listdir("train_val/")
img_name = [k.split(".")[0] for k in l]
df_train = pd.read_csv("annotations/manual/PHOENIX-2014-T.train.corpus.csv", sep="|")
df_train = df_train[df_train['name'].isin(img_name)]

# val
df_val = pd.read_csv("annotations/manual/PHOENIX-2014-T.dev.corpus.csv", sep="|")
df_val = df_val[df_val['name'].isin(img_name)]

list_df = [df_train, df_val]
df = pd.concat(list_df)
df = df[["name", "translation"]]

dico_training = {k : f"<start> {v} <end>" for k,v in zip(df.name, df.translation)}
train_captions = list(dico_training.values())
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



translation_vector, max_length, tokenizer = TextVectozise(train_captions)
img_to_translation_vector = {}
for img, cap in zip(train_img_name, translation_vector):
	img_to_translation_vector["train_val/" + img + ".npy"] = cap


from Encoder import Encoder
from BahdanauAttention import BahdanauAttention
from Decoder import Decoder


im_name_train = list(img_to_translation_vector.keys())

translation_train = list(img_to_translation_vector.values())

steps_per_epoch = len(im_name_train)//BATCH_SIZE


# Load the numpy files
def map_func(im_name_train, translation_train):
    img_tensor = np.load(im_name_train)
    return tf.constant(img_tensor/475, dtype=tf.float32), translation_train

dataset = tf.data.Dataset.from_tensor_slices((im_name_train, translation_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

for (image, label) in dataset.take(1):
    print(f'{image.numpy().shape} and {label.numpy().shape}')


vocab_size = len(tokenizer.index_word) + 1
# Seq2Seq architecture
# Encoder
encoder = Encoder(units, BATCH_SIZE)
example_input_batch,_ = next(iter(dataset))
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape), "\n")
print()


# MA
attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape), "\n")

# Decoder
decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)
sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, sample_output)
print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
print()


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints_MA'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)





@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss




EPOCHS = 250

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                       batch,
                                                       batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))





