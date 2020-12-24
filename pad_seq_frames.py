import numpy
import os

file_names = os.listdir("train/")

max_seq_frame = 475
for fn in file_names:
    X = np.load("train/" + fn)
    n, _ = X.shape
    pad = max_seq_frame - n
    if n < max_seq_frame:
        Y = np.full((pad, 512), 0.)
    np.save("train/" + fn, np.concatenate([X, Y]))
    if np.concatenate([X, Y]).shape[0] != 475:
    	print("Problem with: ", fn)