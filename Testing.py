# Mengimport library TensorFlow, NumPy, dan Matplotlib yang akan digunakan untuk melakukan prediksi dan
# menampilkan gambar.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
# Memuat model yang sudah dilatih sebelumnya dengan nama file "brain_tumor_cnn_model.h5".
model = tf.keras.models.load_model("brain_tumor_cnn_model.h5")

# Load the image and preprocess it
# Memuat gambar yang akan diprediksi dengan nama file "test.jpg" dan ukuran target 150x150 piksel.
img = tf.keras.utils.load_img("test.jpg", target_size=(150, 150))

# Mengubah gambar menjadi array numpy.
img_arr = tf.keras.utils.img_to_array(img)

# Menambah dimensi pada array untuk sesuai dengan input model yang diminta.
img_arr = np.expand_dims(img_arr, axis=0)

# Melakukan normalisasi pada array gambar
img_arr = img_arr / 255.0

# Make a prediction
# Memprediksi status tumor pada gambar.
prediction = model.predict(img_arr)

# Get the predicted class and accuracy
# Menentukan status tumor pada gambar dan akurasinya berdasarkan hasil prediksi.
# Jika nilai prediksi lebih besar dari 0,5, maka tumor akan terdeteksi, jika tidak,
# maka tidak ada tumor pada gamba
if prediction[0][0] > 0.5:
    tumor_status = "Tumor detected"
    accuracy = prediction[0][0]
else:
    tumor_status = "No tumor detected"
    accuracy = 1 - prediction[0][0]

# Print the accuracy
# Menampilkan akurasi prediksi pada gambar dengan format 2 angka di belakang koma.
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Display the image and prediction status
# Menampilkan gambar beserta status tumor dan akurasi prediksinya.
plt.imshow(img)
plt.title(tumor_status)
plt.axis("off")
plt.show()
