from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

model = load_model('model/model.h5')
img_path = 'test.jpg'

img = Image.open(img_path)
og_img = img

if img.mode != 'RGB':
    img = img.convert('RGB')

img = img.resize((128, 128))
img_array = np.array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predictions = np.squeeze(predictions)

threshold = 0.5
threshold_predictions = (predictions > threshold).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(og_img)
axes[0].set_title("Original Image")
axes[0].axis('off')

if predictions.shape[-1] == 1:
    predictions = predictions[:, :, 0]

axes[1].imshow(threshold_predictions, cmap='Reds')
axes[1].set_title("Mask")
axes[1].axis('off')

plt.show()
