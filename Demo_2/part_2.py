from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# Use images with original shape 
X = lfw_people.images

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_classes: %d" % n_classes)

# Split into a training set and a test set using a stratified k fold
train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=42)

# Pre-process data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Add extra dimension
train_images = train_images[:, :, :, np.newaxis]
test_images = test_images[:, :, :, np.newaxis]
print("X_train shape:", train_images.shape)

import matplotlib.pyplot as plt
# Verify images are in correct format

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
    plt.xticks(())
    plt.yticks(())
    
plot_gallery(train_images, train_labels, h, w)
plt.show()

# Build the model 

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 37, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 37, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(7)
])

# Compile the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train by feeding the model

model.fit(train_images, train_labels, epochs=13)

# Evaluate accuracy of test dataset

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

predicted_classes = []

for prediction in predictions:
    predicted_classes.append(np.argmax(prediction))
    
print(classification_report(test_labels, predicted_classes, target_names=target_names))

