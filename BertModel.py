
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models
from keras import layers
from keras import optimizers
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,classification_report


"""
## Configuration
"""

max_length = 100  # Maximum length of input sentence to the model.
batch_size = 32
epochs = 2

# Labels in our dataset.
labels = ["agreed", "disagreed", "unrelated"]


# There are more than 550k samples in total; we will use 100k for this example.
train_df = pd.read_csv("D:/Personal/DeepLearning 1/data/OSNA Project/option1-data/train.csv")
test_df = pd.read_csv("D:/Personal/DeepLearning 1/data/OSNA Project/option1-data/test.csv")





# Shape of the data
#print(f"Total train samples : {train_df.shape[0]}")






"""
One-hot encode training, validation, and test labels.
"""
train_df["label"] = train_df["label"].apply(
    lambda x: 0 if x == "agreed" else 1 if x == "disagreed" else 2
)
y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=3)


X_train_new, X_val, y_train_new, y_val = train_test_split(train_df, y_train, test_size = 1/3, random_state = 0)





"""
## Create a custom data generator
"""


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


"""
## Build the model
"""
# Create the model under a distribution strategy scope.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Encoded token ids from BERT tokenizer.
    input_ids = layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )
    # Loading pretrained BERT model.
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = False

    sequence_output, pooled_output = bert_model(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    )
    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bi_lstm = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True)
    )(sequence_output)
    bi_lstm_2 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(bi_lstm)
    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = layers.GlobalAveragePooling1D()(bi_lstm_2)
    max_pool = layers.GlobalMaxPooling1D()(bi_lstm_2)
    concat = layers.concatenate([avg_pool, max_pool])
    dropout = layers.Dropout(0.3)(concat)
    output = layers.Dense(3, activation="softmax")(dropout)
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    )

    model.compile(
        optimizer=optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )


print(f"Strategy: {strategy}")
model.summary()

"""
Create train and validation data generators
"""
train_data = BertSemanticDataGenerator(
    X_train_new[["title1_en", "title2_en"]].values.astype("str"),
    y_train_new,
    batch_size=batch_size,
    shuffle=True,
)
valid_data = BertSemanticDataGenerator(
    X_val[["title1_en", "title2_en"]].values.astype("str"),
    y_val,
    batch_size=19,
    shuffle=False,
)

"""
## Train the Model

Train the model to extract all the basic features from the dataset
since model is already trained and ready no need to train it again
Uses multiple GPU processing units to train the model since its a very large model

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)

"""
## Fine-tuning



# Unfreeze the bert_model.
bert_model.trainable = True
# Recompile the model to make the change effective.
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

"""
## Train the entire model end-to-end only do it if model weights are not available

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)
"""

test_data = BertSemanticDataGenerator(
        test_df[["title1_en", "title2_en"]].values.astype("str"), labels=None, batch_size=30, shuffle=False, include_targets=False,
    )

#Obtaining the classification report using bertmodel fully unlocked
valid_data = BertSemanticDataGenerator(
    X_val[["title1_en", "title2_en"]].values.astype("str"),
    y_val,
    batch_size=19,
    shuffle=False,
)
#model.save_weights("Bert_Embedding_model.h5")
model.load_weights("Bert_Embedding_model_unlocked.h5")

y_pred = model.predict(valid_data,verbose=1)

y_predict = []
y_true = []
for i in range(y_pred.shape[0]):
    y_predict.append(np.argmax(y_pred[i])) 
    

for i in range(y_val.shape[0]):
    y_true.append(np.argmax(y_val[i]))
#To get the test set predictions


print(classification_report(y_true,y_predict))

y_predict = model.predict(test_data,verbose=1)


test_predictions = []
for i in range(y_predict.shape[0]):
    test_predictions.append(np.argmax(y_predict[i]))
    


for i in range(len(test_predictions)):
    if test_predictions[i] == 2:
        test_predictions[i] = 'unrelated'
    elif test_predictions[i] == 0:
        test_predictions[i] = 'agreed'
    else:
        test_predictions[i] = 'disagreed'
        
    
test_predictions = np.array(test_predictions)

ids = test_df[:, 0].values

ids = ids.reshape((ids.shape[0],1)).astype('int32')
test_predictions = test_predictions.reshape((test_predictions.shape[0],1))

submission = np.concatenate((ids,test_predictions),axis=1)

submission = pd.DataFrame(submission,index=False)


submission.to_csv("submission.csv",header=['id','label'],index=False)



