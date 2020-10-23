#!/usr/bin/env python
# coding: utf-8

# ELMO

# In[1]:


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


from  sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

# Initialize session
config = tf.ConfigProto(gpu_options= tf.GPUOptions(allow_growth=True), allow_soft_placement=True)
sess = tf.Session(config=config)
K.set_session(sess)

# Load all files from a directory in a DataFrame.

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=False
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

    cat_output = layers.Dropout(0.3)(dense_post)

    dense = layers.Dense(100, activation='relu')(cat_output)
    pred = layers.Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs=[in_id_post, in_id_post_enc], outputs=pred)
    print('Model Loaded')
    return model





df = pd.read_csv("SemEval2018-T3-train-taskA.txt", sep='\t')
data = df.dropna(axis = 0, how ='any')

print('Total Sample after cleaning ::', len(data))

tweet_post = data['Tweet text'].tolist()
# text_title1 = data['title'].tolist()
label1 = data['Label'].tolist()



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
    X = pad_sequences(X, padding='post', truncating='post', value = 0, maxlen=m_length)
    return X


def tweet_cl(df_list):
    return [tweet_processor(t).lower() for t in df_list]

tweet_post = tweet_cl(tweet_post)



tknizer = create_tokenizer(tweet_post)

post_max_seq_length = max([len(post.split()) for post in tweet_post])

post_encoded = preprocessed(tknizer, tweet_post, post_max_seq_length)


text_post = np.array(tweet_post, dtype=object)[:, np.newaxis]

task_a_label_dict = {el:idx for idx, el in enumerate(data['Label'].unique())}


label_conv1 = [task_a_label_dict[l] for l in label1]


train_post_text, test_post_text = train_test_split(text_post, test_size=0.2, random_state=2019)
train_labela, test_labela = train_test_split(label_conv1, test_size=0.2, random_state=2019)



train_post_enc, test_post_enc = train_test_split(post_encoded, test_size=0.2, random_state=2019)



from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(hidden_layer_sizes=(5000,200,50), max_iter=500,activation = 'relu',solver='adam',random_state=1)
classifier.fit(train_post_enc,train_labela)

prediction_elmo = classifier.predict(test_post_enc)
rep = classification_report(test_labela, prediction_elmo, digits=4)
print('Classification Report \n' +str(rep))


# In[2]:


import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from bert.tokenization import FullTokenizer
from tqdm import tqdm_notebook
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, mean_squared_error, roc_auc_score

import nltk
from tweet_utils import *
from preprocessing import *

nltk_tokeniser = nltk.tokenize.TweetTokenizer()

# Initialize session
config = tf.ConfigProto(gpu_options= tf.GPUOptions(allow_growth=True), allow_soft_placement=True)
sess = tf.Session(config=config)

# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"



df = pd.read_csv("SemEval2018-T3-train-taskA.txt", sep='\t')
data = df.dropna(axis = 0, how ='any')

text_post1 = data['Tweet text'].tolist()

label = data['Label'].tolist()


def tweet_cl(df_list):
    return [tweet_processor(t).lower() for t in df_list]


text_post1 = tweet_cl(text_post1)


post_max_seq_length = max([len(post.split()) for post in text_post1])



text_post = np.array(text_post1, dtype=object)[:, np.newaxis]


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm_notebook(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples


# In[39]:


# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module()

def wordembedding(tokenizer):
    # load the whole embedding into memory
    embeddings_index = dict()
    for line in open('glove.6B.100d.txt',encoding="utf-8"):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((len(tokenizer.vocab), 100))
    for word, i in tokenizer.vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix



glove_word_embedding = wordembedding(tokenizer)
print(glove_word_embedding.shape)

from  sklearn.model_selection import train_test_split

train_post_text, test_post_text = train_test_split(text_post, test_size=0.2, random_state=2019)

train_label, test_label = train_test_split(label, test_size=0.2, random_state=2019)

train_post_examples = convert_text_to_examples(train_post_text, train_label)


test_post_examples = convert_text_to_examples(test_post_text, test_label)


# Convert to features
(train_input_ids_post, train_input_masks_post, train_segment_ids_post, train_labels_post) = convert_examples_to_features(tokenizer, train_post_examples, max_seq_length=post_max_seq_length)


(test_input_ids_post, test_input_masks_post, test_segment_ids_post, test_labels_post) = convert_examples_to_features(tokenizer, test_post_examples, max_seq_length=post_max_seq_length)


print(post_max_seq_length)

print('')





class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="first",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = False
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


# In[44]:


# Build model
def build_model(max_seq_length_post):

    in_id_post = tf.keras.layers.Input(shape=(max_seq_length_post,), name="input_ids_post")
    in_mask_post = tf.keras.layers.Input(shape=(max_seq_length_post,), name="input_masks_post")
    in_segment_post = tf.keras.layers.Input(shape=(max_seq_length_post,), name="segment_ids_post")
    bert_inputs_post = [in_id_post, in_mask_post, in_segment_post]




    post_embed = tf.keras.layers.Embedding(len(tokenizer.vocab), 100, input_length=max_seq_length_post, weights=[glove_word_embedding], trainable=False)(in_id_post)


    post_lstm = tf.keras.layers.LSTM(300, dropout=0.2, recurrent_dropout=0.5)(post_embed)


    bert_output_post = BertLayer(n_fine_tune_layers=4, pooling="mean")(bert_inputs_post)

    cat_output_post = tf.keras.layers.concatenate([bert_output_post, post_lstm], axis=-1)

    batched_output_post = tf.keras.layers.BatchNormalization()(cat_output_post)

    dense_post = tf.keras.layers.Dense(256, activation='relu')(batched_output_post)


    dense = tf.keras.layers.Dense(100, activation='relu')(dense_post)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    model = tf.keras.models.Model(inputs=[bert_inputs_post], outputs=pred)
    print('Model Loaded')
    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


# In[45]:


model = build_model(post_max_seq_length)


# In[46]:

adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
model.summary()


# In[47]:


train_inputs = [train_input_ids_post, train_input_masks_post, train_segment_ids_post]
test_inputs = [test_input_ids_post, test_input_masks_post, test_segment_ids_post]




# In[3]:


# print(train_labels_post)
print(train_labels_post.shape)


# In[4]:


# print(test_labels_post)
print(test_labels_post.shape)


# In[5]:


train_inputs = np.array(train_inputs)
test_inputs = np.array(test_inputs)
print(train_inputs.shape)
print(test_inputs.shape)


# In[6]:


train_inputs = train_inputs.reshape(3053,210*3)
test_inputs = test_inputs.reshape(764,210*3)
print(test_inputs.shape)
print(train_inputs.shape)


# In[7]:


classifier.fit(train_inputs,train_labels_post)
prediction_bert = classifier.predict(test_inputs.tolist())


# In[8]:


rep = classification_report(test_labels_post, prediction_bert, digits=4)
print('Classification Report \n' +str(rep))


# Average

# In[9]:


prediction_avg = (prediction_bert + prediction_elmo)/2
prediction_avg = [1 if i>=0.5 else 0 for i in prediction_avg]
rep = classification_report(test_labela, prediction_avg, digits=4)
print('Classification Report \n' +str(rep))


# In[10]:


rep = classification_report(test_labels_post, prediction_avg, digits=4)
print('Classification Report \n' +str(rep))


# Weighted Average

# In[11]:


prediction_wavg = (prediction_bert*0.6 + prediction_elmo*0.4)
# print(prediction_wavg)
prediction_wavg = [1 if i>=0.5 else 0 for i in prediction_wavg]
# print(prediction_wavg)
rep = classification_report(test_labels_post, prediction_wavg, digits=4)
print('Classification Report \n' +str(rep))


# In[12]:


prediction_wavg = prediction_bert*0.4+prediction_elmo*0.6
# print(prediction_wavg)
prediction_wavg = [1 if i>=0.5 else 0 for i in prediction_wavg]
# print(prediction_wavg)
rep = classification_report(test_labela, prediction_avg, digits=4)
print('Classification Report \n' +str(rep))


# In[13]:


rep = classification_report(test_labels_post, prediction_wavg, digits=4)
print('Classification Report \n' +str(rep))

