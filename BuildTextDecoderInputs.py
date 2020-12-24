
import pandas as pd
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pickle
import collections

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
train_captions = list(dico_training.values())
train_img_name = list(dico_training.keys())

print(df.head())

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


if __name__ == "__main__":
    translation_vector, max_length, tokenizer = TextVectozise(train_captions)
    img_to_translation_vector = {}
    for img, cap in zip(train_img_name, translation_vector):
    	img_to_translation_vector["train/" + img + ".npy"] = cap

    pickle.dump(img_to_translation_vector, open('Inputs/img_to_translation_vector.p', 'wb'))
    pickle.dump(max_length, open('Inputs/max_length.p', 'wb'))
    pickle.dump(tokenizer, open('Inputs/tokenizer.p', 'wb'))
    pickle.dump(translation_vector, open('Inputs/translation_vector.p', 'wb'))


    print(list(img_to_translation_vector.keys())[:5])
    print(list(img_to_translation_vector.values())[:5])



