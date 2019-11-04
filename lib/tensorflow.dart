import 'dart:io';

import 'package:tflite/tflite.dart';

class Tensorflow {
  Tensorflow({this.model, this.labels});
  final String model;
  final String labels;

  loadModel() async {
    Tflite.close();
    String res;
    res = await Tflite.loadModel(
      model: this.model,
      labels: this.labels,
    );
    print(res);
    return true;
  }

  Future<List> predictImage(File image) async {
    if (image == null) return [];

    var recognitions = await Tflite.runModelOnImage(
        path: image.path,
        imageMean: 0.0,
        imageStd: 255.0,
        numResults: 2,
        threshold: 0.2,
        asynch: true);
    return recognitions;
  }
}
