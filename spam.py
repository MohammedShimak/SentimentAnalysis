import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np

# Sample spam dataset
emails = [
    "Win money now claim your prize",
    "Congratulations you have won a lottery",
    "Claim your free gift voucher now",
    "Meeting at 3pm is confirmed",
    "Let's have lunch tomorrow",
    "Project deadline is extended",
    "Get cheap loans easy approval",
    "Buy medicines online without prescription"
]

labels = [1,1,1,0,0,0,1,1]  # 1 = Spam, 0 = Ham

# Tokenization
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(emails)

sequences = tokenizer.texts_to_sequences(emails)
padded = pad_sequences(sequences, maxlen=8)

# Split dataset
X_train = padded[:6]
X_test = padded[6:]

y_train = np.array(labels[:6])
y_test = np.array(labels[6:])

# Build LSTM Model
model = Sequential([
    Embedding(1000, 64, input_length=8),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ---- print model name ----
print('Model: "sequential"\n')

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=2,
    validation_split=0.2,
    verbose=0
)

# ---- Extract last epoch values ----
acc = history.history['accuracy'][-1]
loss = history.history['loss'][-1]
val_acc = history.history['val_accuracy'][-1]
val_loss = history.history['val_loss'][-1]

# ---- Print results in your format ----
print(f"accuracy: {acc:.4f} - loss: {loss:.4f} - val_accuracy: {val_acc:.4f} - val_loss: {val_loss:.4f}")

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy}")

# Prediction function
def predict_email(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=8)
    pred = model.predict(pad, verbose=0)[0][0]
    label = "Positive" if pred > 0.5 else "Negative"
    return f"{label} ({pred:.2f})"

# Sample prediction
sample = "Congratulations you won a free gift"
print("Sample Prediction:", predict_email(sample))
