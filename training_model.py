import numpy as np
from model import get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from helpers import get_word_ids, get_sequences_and_labels
from constants import *

def training_model(model_path, epochs=500, keypoints_path=KEYPOINTS_PATH, word_ids=None):
    if word_ids is None:
        word_ids = get_word_ids(WORDS_JSON_PATH)
    
    sequences, labels = get_sequences_and_labels(word_ids, keypoints_path)
    
    sequences = pad_sequences(sequences, maxlen=int(MODEL_FRAMES), padding='pre', truncating='post', dtype='float16')
    
    X = np.array(sequences)
    labels = np.array(labels)
    y = to_categorical(labels, num_classes=len(word_ids)).astype(int)

    X_train, X_val, y_train, y_val, labels_train, _ = train_test_split(
        X, y, labels, test_size=0.2, random_state=42, stratify=labels
    )
    class_counts = np.bincount(labels_train, minlength=len(word_ids))
    class_weight = {
        class_index: len(labels_train) / (len(word_ids) * count)
        for class_index, count in enumerate(class_counts)
        if count
    }

    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=25, restore_best_weights=True
    )
    
    model = get_model(int(MODEL_FRAMES), len(word_ids))
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=8,
        class_weight=class_weight,
        callbacks=[early_stopping],
    )
    
    model.summary()
    model.save(model_path)

if __name__ == "__main__":
    training_model(MODEL_PATH)
    