# 🐶🐱 Dogs vs Cats Image Classification using CNN

A simple Deep Learning project that uses a **Convolutional Neural Network (CNN)** to classify images as either a **Dog** or a **Cat**.

The model is developed and trained using **Google Colab, Python, and TensorFlow/Keras**. The project demonstrates the complete image-classification workflow, including image preprocessing, CNN model building, training, evaluation, and prediction on new images.

---

## 📌 Project Overview

Image classification is a common application of Computer Vision and Deep Learning.

In this project, a CNN model is trained on images of cats and dogs. After training, the model can analyze a new image and predict whether it contains a:

* 🐱 Cat
* 🐶 Dog

The project also includes a basic **confidence threshold** so that images with uncertain predictions can receive a warning instead of immediately being classified.

---

## 🎯 Project Objectives

* Build an image classification model using CNN.
* Understand the basic working of Convolutional Neural Networks.
* Preprocess and normalize image data.
* Train a CNN using TensorFlow/Keras.
* Evaluate model performance using test data.
* Visualize training and validation accuracy.
* Visualize training and validation loss.
* Predict the class of a new image.
* Add a confidence threshold for uncertain predictions.

---

## 🛠️ Technologies Used

* **Python**
* **Google Colab**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Matplotlib**
* **CNN (Convolutional Neural Network)**
* **Kaggle Dogs vs Cats Dataset**

---

## 📂 Dataset

The project uses the **Dogs vs Cats** image dataset from Kaggle.

**Dataset:**
https://www.kaggle.com/datasets/salader/dogsvscats

The dataset contains images belonging to two classes:

```text
Cat
Dog
```

The images are resized to:

```text
128 × 128 pixels
```

before being passed to the CNN.

---

## 🧠 CNN Architecture

The CNN model consists of multiple convolutional and pooling layers followed by fully connected layers.

```text
Input Image
     ↓
Rescaling
     ↓
Conv2D (32 filters)
     ↓
MaxPooling2D
     ↓
Conv2D (64 filters)
     ↓
MaxPooling2D
     ↓
Conv2D (128 filters)
     ↓
MaxPooling2D
     ↓
Flatten
     ↓
Dense (128 neurons)
     ↓
Dropout (0.5)
     ↓
Dense (1 neuron)
     ↓
Sigmoid
     ↓
Cat / Dog
```

### Why CNN?

CNNs are especially useful for image-related tasks because they can automatically learn visual patterns such as:

* Edges
* Shapes
* Textures
* Fur patterns
* Facial features
* Other image characteristics

---

## 🔄 Data Preprocessing

Before training, the images are:

1. Loaded from the dataset.
2. Resized to `128 × 128`.
3. Converted into numerical arrays.
4. Pixel values are normalized from:

```text
0 – 255
```

to:

```text
0 – 1
```

This normalization is performed using:

```python
tf.keras.layers.Rescaling(1./255)
```

---

## 🏗️ Model Building

The CNN model is created using TensorFlow/Keras:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

---

## ⚙️ Model Compilation

The model uses the **Adam optimizer**, **Binary Crossentropy loss**, and **Accuracy** as the evaluation metric.

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

### Why Binary Crossentropy?

There are only two target classes:

```text
Cat
Dog
```

Therefore, binary classification is used.

---

## 🚀 Model Training

The model is trained using the training dataset and evaluated on the validation/test dataset.

```python
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10
)
```

The training history is stored so that accuracy and loss can be visualized later.

---

## 📊 Model Evaluation

The trained model is evaluated using the test dataset:

```python
loss, accuracy = model.evaluate(test_dataset)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
```

The project also visualizes:

### Training vs Validation Accuracy

The accuracy graph helps understand how the model's performance changes during training.

### Training vs Validation Loss

The loss graph helps identify whether the model is learning properly and whether overfitting may be occurring.

---

## 🔍 New Image Prediction

After training, a new image can be uploaded to Google Colab.

The image is resized and passed to the trained CNN model.

Example output:

```text
🐶 Dog (94.32% confidence)
```

or:

```text
🐱 Cat (91.45% confidence)
```

---

## ⚠️ Confidence Threshold

A basic confidence threshold is also used to handle uncertain predictions.

For example:

```python
threshold = 0.75

if prediction >= threshold:
    print(f"🐶 Dog ({prediction * 100:.2f}% confidence)")

elif prediction <= (1 - threshold):
    print(f"🐱 Cat ({(1 - prediction) * 100:.2f}% confidence)")

else:
    print("⚠️ Image clearly identify nahi ho rahi.")
    print("Please upload a clear cat or dog image.")
```

This provides a basic safeguard against low-confidence predictions.

> **Note:** A confidence threshold alone cannot reliably detect every unrelated image, because a binary CNN may still produce a high score for an image outside its training classes. A more robust solution would train an additional `Other` class or use a dedicated out-of-distribution detection approach.

---

## 💾 Saving the Trained Model

The trained model can be saved in Keras format:

```python
model.save("dogs_vs_cats_cnn.keras")
```

This allows the trained model to be reused later without training it again.

---

## 📁 Project Structure

```text
Dogs-vs-Cats-CNN/
│
├── Dogs_vs_Cats_CNN.ipynb
├── dogs_vs_cats_cnn.keras
└── README.md
```

The main project is implemented in the Google Colab notebook:

```text
Dogs_vs_Cats_CNN.ipynb
```

---

## 📈 Results

The final accuracy and loss depend on the actual training run, dataset split, number of epochs, and hardware.

After running the notebook, add your actual result here:

```text
Test Accuracy: XX.XX%
Test Loss: X.XXXX
```

You can also add screenshots of:

* Sample dataset images
* CNN model summary
* Training accuracy graph
* Validation accuracy graph
* Loss graph
* Prediction results

---

## 🎓 What I Learned

Through this project, I learned:

* Basics of Computer Vision
* Image preprocessing
* Image normalization
* CNN architecture
* Convolutional layers
* Max Pooling
* Flattening
* Dense layers
* Dropout
* Sigmoid activation
* Binary classification
* Model training and validation
* Accuracy and loss visualization
* Making predictions on new images

---

## 🔮 Future Improvements

This project can be improved by:

* Adding **Data Augmentation**
* Adding more training images
* Using **Early Stopping**
* Using a learning-rate scheduler
* Increasing image resolution
* Adding an **Other** class for non-cat/non-dog images
* Using Transfer Learning with models such as MobileNetV2, VGG16, or ResNet
* Creating a Streamlit web application for image prediction

---

## 👩‍💻 Author

**Aliya Yousuf**

Bachelor's in Computational Mathematics

Interested in Data Analysis, Python, Machine Learning and Artificial Intelligence.

---

## ⭐ Conclusion

This project demonstrates how a **Convolutional Neural Network can be used for binary image classification**.

The complete workflow starts from loading and preprocessing images, followed by building and training a CNN model, evaluating its performance, and finally making predictions on new images.

This project provides a practical introduction to **Deep Learning and Computer Vision using TensorFlow/Keras**.
