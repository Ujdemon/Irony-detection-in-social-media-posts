# Import our dependencies
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
from keras import backend as K
import keras.layers as layers
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from sklearn.metrics import classification_report, mean_squared_error, roc_auc_score
from keras.engine import Layer
import numpy as np

from preprocessing import *
from tweet_utils import *

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

# Initialize session
sess = tf.Session()
K.set_session(sess)


# Load all files from a directory in a DataFrame.

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
                           )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


# Build model
def build_model(max_seq_length_post):
    in_id_post_enc = layers.Input(shape=(max_seq_length_post,), name="input_ids_post")
    # in_id_des_enc = layers.Input(shape=(max_seq_length_des,), name="input_ids_des")

    in_id_post = layers.Input(shape=(1,), dtype="string")
    # in_id_des = layers.Input(shape=(1,), dtype="string")

    post_embed = layers.Embedding(len(tknizer.word_index), 100, input_length=max_seq_length_post)(in_id_post_enc)
    # des_embed = layers.Embedding(len(tknizer.word_index), 100, input_length=max_seq_length_des)(in_id_des_enc)

    post_lstm = layers.LSTM(300, dropout=0.2, recurrent_dropout=0.5)(post_embed)
    # des_lstm = layers.LSTM(300, dropout=0.2, recurrent_dropout=0.5)(des_embed)

    elmo_output_post = ElmoEmbeddingLayer()(in_id_post)
    # elmo_output_des = ElmoEmbeddingLayer()(in_id_des)

    cat_output_post = layers.concatenate([elmo_output_post, post_lstm], axis=-1)
    # cat_output_des = layers.concatenate([elmo_output_des, des_lstm], axis=-1)

    batched_output_post = layers.BatchNormalization()(cat_output_post)
    # batched_output_des = layers.BatchNormalization()(cat_output_des)

    dense_post = layers.Dense(256, activation='relu')(batched_output_post)
    #     dense_title = tf.keras.layers.Dense(256, activation='relu')(cat_output_title)
    # dense_des = layers.Dense(256, activation='relu')(batched_output_des)

    # cat_output = layers.concatenate([dense_post, dense_des], axis=-1)
    cat_output = layers.Dropout(0.3)(dense_post)

    dense = layers.Dense(100, activation='relu')(cat_output)
    pred = layers.Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[in_id_post, in_id_post_enc], outputs=pred)
    print('Model Loaded')
    return model


df = pd.read_csv("election.csv", sep=',')
data = df.dropna(axis=0, how='any')

print('Total Sample after cleaning ::', len(data))

tweet_post = data['Tweets'].tolist()
# text_title1 = data['title'].tolist()
label1 = data['Task A'].tolist()
# label2 = data['Task B'].tolist()
# label3 = data['Task C'].tolist()


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np


def create_tokenizer(p):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(p)
    # tokenizer.fit_on_texts(t)
    return tokenizer


def preprocessed(tokenizer, docs, m_length):
    X = tokenizer.texts_to_sequences(docs)
    X = pad_sequences(X, padding='post', truncating='post', value=0, maxlen=m_length)
    return X


def tweet_cl(df_list):
    return [tweet_processor(t).lower() for t in df_list]


tweet_post = tweet_cl(tweet_post)
# text_title1 = tweet_cl(text_title1)
# text_des1 = tweet_cl(text_des1)


tknizer = create_tokenizer(tweet_post)

post_max_seq_length = max([len(post.split()) for post in tweet_post])

post_encoded = preprocessed(tknizer, tweet_post, post_max_seq_length)

# title_max_seq_length = max([len(title.split()) for title in text_title1])
# des_max_seq_length = max([len(post.split()) for post in text_des1])


text_post = np.array(tweet_post, dtype=object)[:, np.newaxis]
# text_title = np.array(text_title1, dtype=object)[:, np.newaxis]
# text_des = np.array(text_des1, dtype=object)[:, np.newaxis]

task_a_label_dict = {el: idx for idx, el in enumerate(data['Task A'].unique())}
# task_b_label_dict = {el:idx for idx, el in enumerate(data['Task B'].unique())}
# task_c_label_dict = {el:idx for idx, el in enumerate(data['Task C'].unique())}

label_conv1 = [task_a_label_dict[l] for l in label1]
# label_conv2 = [task_b_label_dict[l] for l in label2]
# label_conv3 = [task_c_label_dict[l] for l in label3]


train_post_text, test_post_text = train_test_split(text_post, test_size=0.2, random_state=2019)
train_labela, test_labela = train_test_split(label_conv1, test_size=0.2, random_state=2019)
# train_labelb, test_labelb = train_test_split(label_conv2, test_size=0.2, random_state=2019)
# train_labelc, test_labelc = train_test_split(label_conv3, test_size=0.2, random_state=2019)


train_post_enc, test_post_enc = train_test_split(post_encoded, test_size=0.2, random_state=2019)
# train_title_text, test_title_text = train_test_split(text_title, test_size=0.2, random_state=2019)
# train_des_enc, test_des_enc = train_test_split(des_encoded, test_size=0.2, random_state=2019)

# print(train_post_text.shape)
# print(train_post_enc.shape)

# vocab_len_enc = len(tknizer.word_index)
# post_len_enc = train_post_enc.shape[1]
# des_len_enc = train_des_enc.shape[1]
#
# print('Vocab size ::', vocab_len_enc)
# print('POST length ::', post_len_enc)
# print('POST_ELMO length ::', post_max_seq_length)
# print('DES length ::', des_len_enc)
# print('DES_ELMO length ::', des_max_seq_length)

# Build and fit
model = build_model(post_max_seq_length)

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
model.summary()

cb = EarlyStopping(monitor='val_loss', patience=3)
mc = ModelCheckpoint('ElmoSimpleModel.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

model.fit([train_post_text, train_post_enc],
          train_labela,
          validation_data=([test_post_text, test_post_enc], test_labela),
          epochs=10,
          batch_size=32,
          callbacks=[cb, mc],
          verbose=2)

post_save_preds = model.predict([test_post_text, test_post_enc])  # predictions after we clear and reload model

prediction = [1 if ps > 0.5 else 0 for ps in post_save_preds]

# print(prediction)
# In[ ]:

mse = mean_squared_error(test_labela, prediction)
print('Mean Squared Error = ' + str(mse))

roc_auc = roc_auc_score(test_labela, prediction)
print('ROC_AUC Report = ' + str(roc_auc))

rep = classification_report(test_labela, prediction, digits=4)
print('Classification Report \n' + str(rep))