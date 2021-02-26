from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras import layers


def biLSTMAtt(lr):


  sequence_input = Input(shape=(max_len,), dtype="int32")
  embedded_sequences = Embedding(vocab_size+1, D, mask_zero=True)(sequence_input)

  (lstm, forward_h, forward_c, 
    backward_h, backward_c) = Bidirectional(LSTM(32, input_shape=(max_len, vocab_size), return_sequences=True,
                                                  return_state=True), name="bi_lstm_1")(embedded_sequences)

  state_h = Concatenate()([forward_h, backward_h])
  state_c = Concatenate()([forward_c, backward_c])

  context_vector, weights = Attention(return_attention=True)(lstm)

  dense1 = Dense(128, activation="relu")(context_vector)
  dropout = Dropout(0.4)(dense1)
  output = Dense(14, activation='softmax')(dropout)

  model = tf.keras.Model(inputs=sequence_input, outputs=output)
  
  # compile model
  opt = tf.keras.optimizers.RMSprop(lr=lr, momentum=0.01, centered=True)
  
  METRICS = [
	  tf.keras.metrics.TruePositives(name='tp'),
	  tf.keras.metrics.FalsePositives(name='fp'),
  	tf.keras.metrics.TrueNegatives(name='tn'),
	  tf.keras.metrics.FalseNegatives(name='fn'),
	  tf.keras.metrics.BinaryAccuracy(name='accuracy'),
	  tf.keras.metrics.Precision(name='precision'),
	  tf.keras.metrics.Recall(name='recall'),
	  tf.keras.metrics.AUC(name='auc'),
    ]

  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=METRICS)
  
  return model