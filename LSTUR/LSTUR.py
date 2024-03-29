#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import zipfile
from tqdm import tqdm
#import scrapbook as sb
from tempfile import TemporaryDirectory
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.lstur import LSTURModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

print("*** LSTUR model ***")
print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))


epochs = 5
seed = 40
batch_size = 32

# Options: demo, small, large
MIND_type = 'large'
print("*** Dataset:", MIND_type)

# ### Dowload data
cwd = os.getcwd()
data_path = cwd + '/data/' #tmpdir.name

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')

wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'lstur.yaml')


# Parameters
hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs)
print("*** Parameters:")
print(hparams)

iterator = MINDIterator

# Train the LSTUR model
model = LSTURModel(hparams, iterator, seed=seed)

print("*** Model Baseline")
print(model.run_eval(valid_news_file, valid_behaviors_file))

print("*** Fiting model")
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)

res_syn = model.run_eval(valid_news_file, valid_behaviors_file)
print("*** Metrics after training")
print(res_syn)

# Save the model
model_path = os.path.join(data_path, "model")
os.makedirs(model_path, exist_ok=True)

print("Model saving")
model.model.save_weights(os.path.join(model_path, "lstur_ckpt"))


# Output prediction file
group_impr_indexes, group_labels, group_preds = model.run_fast_eval(valid_news_file, valid_behaviors_file)

with open(os.path.join(data_path, 'prediction.txt'), 'w') as f:
    for impr_index, preds in zip(group_impr_indexes, group_preds):
        impr_index += 1
        pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
        pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
        f.write(' '.join([str(impr_index), pred_rank])+ '\n')

f = zipfile.ZipFile(os.path.join(data_path, 'prediction.zip'), 'w', zipfile.ZIP_DEFLATED)
f.write(os.path.join(data_path, 'prediction.txt'), arcname='prediction.txt')
f.close()

