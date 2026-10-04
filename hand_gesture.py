import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

data_path = "leapGestRecog"

images = []
labels = []

gestures = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down"
]

for person in os.listdir(data_path):
    person_path = os.path.join(data_path, person)

    if not os.path.isdir(person_path):
        continue

    for label, gesture in enumerate(gestures):
        gesture_path = os.path.join(person_path, gesture)

        if not os.path.exists(gesture_path):
            continue

        for file in os.listdir(gesture_path):
            image_path = os.path.join(gesture_path, file)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, (64, 64))
            images.append(image)
            labels.append(label)

X = np.array(images)
y = np.array(labels)

X = X / 255.0
X = X.reshape(-1, 64, 64, 1)

y = to_categorical(y, num_classes=10)

print("Total images:", len(X))
print("Image shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training model...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training Accuracy")
plt.legend()
plt.show()

predictions = model.predict(X_test[:6])

plt.figure(figsize=(12, 8))

for i in range(6):
    predicted_label = np.argmax(predictions[i])

    
    display_image = cv2.resize(
        X_test[i].reshape(64, 64),
        (256, 256),
        interpolation=cv2.INTER_CUBIC
    )

    plt.subplot(2, 3, i + 1)
    plt.imshow(display_image, cmap="gray")
    plt.title(
        "Predicted: " + gestures[predicted_label].split("_", 1)[1],
        fontsize=12
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

model.save("hand_gesture_model.keras")