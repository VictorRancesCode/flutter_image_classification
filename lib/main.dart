import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_image_classification/tensorflow.dart';
import 'package:image_picker/image_picker.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(title: 'Flutter image Classification'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key, this.title}) : super(key: key);
  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  Tensorflow tensor =
      new Tensorflow(model: "assets/model.tflite", labels: "assets/labels.txt");
  File _image;
  List _recognitions = [];

  @override
  void initState() {
    super.initState();
    tensor.loadModel();
  }

  selectFromImagePicker() async {
    var image = await ImagePicker.pickImage(source: ImageSource.gallery);
    if (image == null) return;
    setState(() {
      _image = image;
    });
    List recognitions = await tensor.predictImage(image);
    setState(() {
      _recognitions=recognitions;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image == null ? Text('No image selected.') : Image.file(_image),
            new Container(
              height: 200.0,
              child: ListView.builder(
                itemCount: _recognitions.length,
                itemBuilder: (context, index) {
                  return ListTile(
                    title: Text(_recognitions[index]['label']),
                    subtitle: Text(
                        "Confidence ${_recognitions[index]['confidence'].toString()}"),
                  );
                },
              ),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        child: Icon(Icons.image),
        tooltip: "Pick Image from gallery",
        onPressed: selectFromImagePicker,
      ),
    );
  }
}
