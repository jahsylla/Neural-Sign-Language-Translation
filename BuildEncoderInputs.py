
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from tqdm import tqdm
import pickle

# Must to chage : variable path_img, PHOENIX_CSV, and where to save data in build_feature_vector function

path_img = "train_video/"
PHOENIX_CSV = pd.read_csv("annotations/manual/PHOENIX-2014-T.train.corpus.csv", sep="|")

# dico_training = {path_img + k + "/": f"<start> {v} <end>" for k,v in zip(PHOENIX_CSV.name, PHOENIX_CSV.orth)}
# train_captions = list(dico_training.values())
omen = pickle.load(open("omen.p",'rb'))
gricad = pickle.load(open("gricad.p",'rb'))
l1 = omen + gricad
l2 = os.listdir("train/")
l2 = [i.split(".")[0] for i in l2]


img_name_vector = []
for k in l1:
	if k not in l2:
		img_name_vector.append(k)


n = int(len(img_name_vector)/2)
l3 = img_name_vector[:n]
pickle.dump(img_name_vector[n:len(img_name_vector)], open("work_station2.p", "wb"))

img_name_vector = ["train_video/" + k + "/" for k in l3]

# Features extraction model VGG16
def features_extraction(X):
    base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    model_GAP = tf.keras.Model(inputs=base_model.input, outputs=avg)
    features= model_GAP.predict(X)
    del base_model, model_GAP, avg
    tf.keras.backend.clear_session()
    return features


# Image loader for transfert learning 
def load_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# Build feature vector
def build_feature_vector(max_length_frame = 475):
    """Build feature vector for video frames.
    because video matrice don't have same number of frames we pad all them with 0 in order they have same dim"""
    
    for vid in tqdm(img_name_vector):
        frame_name = os.listdir(vid)
        X = np.empty((len(frame_name), 512))
        for i, fn in enumerate(frame_name):
            img = load_image(image_path = vid + fn)
            features = features_extraction(img)
            X[i,] = features

        # Padding
        n = X.shape[0]
        pad = max_length_frame - n
        if n < max_length_frame:
        	Y = np.full((pad, 512), 0.)
        	X = np.concatenate([X, Y])
        	np.save("train/" + vid.split('/')[-2]+".npy", X)
        else:
        	np.save("train/" + vid.split('/')[-2]+".npy", X)

        if X.shape[0] != 475:
            print("Problem with: ", vid)


if __name__ == "__main__":
    build_feature_vector(max_length_frame = 475)