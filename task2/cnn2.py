
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Load and preprocess the bird.png image
bird_image = Image.open("/Users/giorgia/Desktop/hw2-3/bird.png")
bird_image = bird_image.resize((32, 32))  # Resize the image to match CIFAR-10 dimensions
bird_array = np.array(bird_image) / 255.0  # Convert image to array and normalize pixel values

# Convert the bird image to RGB (if it's in RGBA format)
if bird_array.shape[2] == 4:
    bird_array = bird_array[:, :, :3]

# Add the bird image to the training dataset
train_images = np.concatenate((train_images, [bird_array]), axis=0)
train_labels = np.concatenate((train_labels, [[2]]))  # Assuming '2' represents the 'bird' class

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Visualize the added bird image
plt.figure(figsize=(10, 10))
for i in range(26):
    plt.subplot(5, 6, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if i == 25:
        plt.imshow(bird_array)
        plt.xlabel("Added Bird")
    else:
        plt.imshow(train_images[i])
        plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Define and compile the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=70, validation_data=(test_images, test_labels))

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Make predictions on the bird image
bird_prediction = model.predict(np.expand_dims(bird_array, axis=0))
predicted_class_index = np.argmax(bird_prediction)
predicted_class = class_names[predicted_class_index]

print("Predicted Class for the Bird Image:", predicted_class)

#Predicted Class for the Bird Image: bird