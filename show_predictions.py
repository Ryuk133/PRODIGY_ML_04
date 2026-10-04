import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("hand_gesture_model.keras")

data_path = "leapGestRecog"

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

images = []
display_images = []

# Take one image from each of 6 different gestures
selected_gestures = [0, 1, 2, 4, 6, 9]

person_path = os.path.join(data_path, "00")

for label in selected_gestures:

    gesture_path = os.path.join(
        person_path,
        gestures[label]
    )

    files = os.listdir(gesture_path)

    image_path = os.path.join(
        gesture_path,
        files[0]
    )

    original = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:
        continue

    # Image used by CNN
    model_image = cv2.resize(
        original,
        (64, 64)
    )

    model_image = model_image / 255.0
    model_image = model_image.reshape(64, 64, 1)

    images.append(model_image)

    display_image = original.copy()

    display_image = cv2.normalize(
        display_image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Find the bright hand area
    _, threshold = cv2.threshold(
        display_image,
        40,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(
            contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(largest)

        # Add some space around the hand
        padding = 15

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(display_image.shape[1], x + w + padding)
        y2 = min(display_image.shape[0], y + h + padding)

        display_image = display_image[
            y1:y2,
            x1:x2
        ]

    display_images.append(display_image)

X_sample = np.array(images)

predictions = model.predict(X_sample, verbose=0)

plt.figure(figsize=(10, 7))

for i in range(len(X_sample)):

    predicted_label = np.argmax(predictions[i])

    image = display_images[i]

    plt.subplot(2, 3, i + 1)

    plt.imshow(
        image,
        cmap="gray",
        interpolation="nearest"
    )

    plt.title(
        "Predicted: " +
        gestures[predicted_label].split("_", 1)[1].title(),
        fontsize=12
    )

    plt.axis("off")

plt.suptitle(
    "Hand Gesture Recognition using CNN",
    fontsize=17
)

plt.tight_layout()

plt.savefig(
    "hand_gesture_predictions.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()