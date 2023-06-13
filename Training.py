# Import library
# Mengimpor library yang dibutuhkan: os untuk mengoperasikan sistem, zipfile untuk mengekstrak file zip,
# numpy untuk manipulasi array, matplotlib untuk visualisasi, ImageDataGenerator untuk preprocessing citra,
# Sequential, Conv2D, MaxPooling2D, Flatten, dan Dense untuk membangun arsitektur model CNN,
# serta optimizers untuk optimizer dalam pelatihan model.
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import optimizers

# Mengekstrak dataset data.zip ke dalam direktori saat ini.
# z = zipfile.ZipFile("data.zip", "r")
# z.extractall()

# Split dataset into train and validation set
# Membuat direktori train/no, train/yes,
# validation/no, dan validation/yes sebagai tempat penyimpanan dataset yang telah dipisahkan.
# os.makedirs("train/no")
# os.makedirs("train/yes")
# os.makedirs("validation/no")
# os.makedirs("validation/yes")

# Memisahkan dataset menjadi train dan validation set dengan rasio 8:2,
# kemudian memindahkan file-file dataset ke dalam direktori yang sesuai.
# for file in os.listdir("no"):
#   if np.random.rand(1) < 0.8:
#      os.rename("no/" + file, "train/no/" + file)
# else:
#    os.rename("no/" + file, "validation/no/" + file)

# for file in os.listdir("yes"):
#   if np.random.rand(1) < 0.8:
#      os.rename("yes/" + file, "train/yes/" + file)
# else:
#    os.rename("yes/" + file, "validation/yes/" + file)

# Data preprocessing
# Melakukan data preprocessing pada citra menggunakan ImageDataGenerator. Pada train dataset,
# citra di-rescale menjadi rentang 0-1, dan dilakukan augmentasi citra seperti shear, zoom, dan horizontal flip.
# Sedangkan pada validation dataset, hanya dilakukan rescale.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Membuat objek train_generator dan validation_generator menggunakan flow_from_directory untuk
# mengambil data dari direktori train dan validation.
train_generator = train_datagen.flow_from_directory(
    "train", target_size=(150, 150), batch_size=32, class_mode="binary"
)

validation_generator = val_datagen.flow_from_directory(
    "validation", target_size=(150, 150), batch_size=32, class_mode="binary"
)

# Define CNN model
# Membangun arsitektur model CNN dengan menambahkan layer-layer Conv2D, MaxPooling2D, Flatten, dan Dense.
# Input shape adalah (150,150,3) yang sesuai dengan dimensi citra dan jumlah kanalnya.
# Layer output menggunakan fungsi aktivasi sigmoid untuk menghasilkan prediksi biner (tumor atau bukan).
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile model
# Melakukan kompilasi model dengan loss function binary_crossentropy,
# optimizer RMSprop dengan learning rate 1e-4, dan metrics acc (akurasi).
model.compile(
    loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4), metrics=["acc"]
)

# Train model
# Melatih model dengan memanggil metode fit pada objek model.
# Melakukan iterasi selama 50 epoch pada train dataset dengan batch size 32.
# Validasi model pada validation dataset.
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size,
)

# Evaluate model
# Melakukan evaluasi model dengan memanggil metode evaluate pada objek model menggunakan validation dataset
# dan menghitung rata rata akurasi training
test_loss, test_acc = model.evaluate(validation_generator)
train_acc = history.history["acc"]
avg_train_acc = sum(train_acc) / len(train_acc)

print("Average Train Accuracy:", avg_train_acc)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

# simpan model ke fil
# Menyimpan model yang telah dilatih ke dalam file brain_tumor_cnn_model.h5.
model.save("brain_tumor_cnn_model.h5")

# plot grafik akurasi pelatihan
# Melakukan plot grafik akurasi pelatihan menggunakan matplotlib.
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()
