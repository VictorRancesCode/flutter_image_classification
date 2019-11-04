### Train Your Model
* Create folder data
* In folder data create two folders
    * images_train
    * images_validation
* In folders, create folders for classification Example in image_train create:
    * dog (in this folder all images the dogs)
    * cat (in this folder all images the cats)
    * pig (in this folder all images the pigs)
    ....
* ok, in folder images_validation it would be completely the same but these images will serve to make tests.
* once we have everything, we execute:
```
$ pip install -r requierements.txt
$ python train.py
```

* ok, will create us folders
    * saved_models
    * tflite_models

* in folder tflite_models the files exist
    * labels.txt
    * model.tflite
    * model_quant.tflite
    
    
[Excellent Tutorial Tensorflow](https://github.com/frogermcs/TFLite-Tester/blob/master/notebooks/Testing_TFLite_model.ipynb)