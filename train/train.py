import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

"""
Parameters
"""
IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR =  './data/images_train'
VALIDATION_DATA_DIR = './data/images_validation'


datagen_kwargs = dict(rescale=1./255, validation_split=.20)
valid_datagen = ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    VALIDATION_DATA_DIR, 
    subset="validation", 
    shuffle=True,
    target_size=IMAGE_SHAPE
)

train_datagen = ImageDataGenerator(**datagen_kwargs)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_DIR, 
    subset="training", 
    shuffle=True,
    target_size=IMAGE_SHAPE)


image_batch_train, label_batch_train = next(iter(train_generator))
print("Image batch shape: ", image_batch_train.shape)
print("Label batch shape: ", label_batch_train.shape)

dataset_labels = sorted(train_generator.class_indices.items(), key=lambda pair:pair[1])
dataset_labels = np.array([key.title() for key, value in dataset_labels])
print(dataset_labels)


model = tf.keras.Sequential([
  hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", 
                 output_shape=[1280],
                 trainable=False),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])
model.build([None, 224, 224, 3])

model.summary()


model.compile(
  optimizer=optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])



steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
val_steps_per_epoch = np.ceil(valid_generator.samples/valid_generator.batch_size)

hist = model.fit(
    train_generator, 
    epochs=10,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=val_steps_per_epoch).history



final_loss, final_accuracy = model.evaluate(valid_generator, steps = val_steps_per_epoch)
SAVED_MODEL = "saved_models"

model.save('model.h5')

tf.keras.experimental.export_saved_model(model, SAVED_MODEL)


# Load SavedModel

load_model = tf.keras.experimental.load_from_saved_model(SAVED_MODEL, 
                                                            custom_objects={'KerasLayer':hub.KerasLayer})    
dirName= "tflite_models"


if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")



TFLITE_MODEL = "tflite_models/model.tflite"
TFLITE_QUANT_MODEL = "tflite_models/model_quant.tflite"

# Get the concrete function from the Keras model.
run_model = tf.function(lambda x : load_model(x))

# Save the concrete function.
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
)

# Convert the model
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converted_tflite_model = converter.convert()
with open('tflite_models/labels.txt', "w") as f:
    for line in dataset_labels:
        for item in line.split(' '):
            f.write(item + '\t')
        f.write("\n")

open(TFLITE_MODEL, "wb").write(converted_tflite_model)

# Convert the model to quantized version with post-training quantization
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open(TFLITE_QUANT_MODEL, "wb").write(tflite_quant_model)