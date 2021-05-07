from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import time

path_to_file = "./datasets/adventure_compilation.txt"

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# The unique characters in the file
vocab = sorted(set(text))

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# Build the model
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=1)



model.load_weights(tf.train.latest_checkpoint('./training_checkpoints'))
model.build(tf.TensorShape([1, None]))



#quit = False

print("\n\n\nYou awake in a soft place.\n" +
"\n\n" +
"[type what you want to do, or use commands \n" +
"north	n	Move north\n" +
"south	s	Move south\n" +
"east	e	Move east\n" +
"west	w	Move west\n" +
"northeast	ne	Move northeast\n" +
"northwest	nw	Move northwest\n" +
"southeast	se	Move southeast\n" +
"southwest	sw	Move southwest\n" +
"up	u	Move up\n" +
"down	d	Move down\n" +
"look	l	Looks around at current location\n" +
"inventory	i	Shows contents of Inventory\n" +
"quit q quits game]\n")


def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # these characters are undefined
      if predicted_id > 95:
          continue

      # make sure its not a prompt symbol
      if predicted_id == 32:
          return (" " + ''.join(text_generated))

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])
      #print(predicted_id)
      #print(idx2char[predicted_id])


  return (" " + ''.join(text_generated))



while True:
    user_input = input(">")
    if(user_input == "quit" or user_input == "q"):
        break;
    print(generate_text(model, start_string=user_input))
