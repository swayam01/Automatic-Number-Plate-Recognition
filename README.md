# India-Number-Plate-Recognition
This project demonstrates the use of [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to automatically recognise number plates of Indian vehicles.

Dataset used: https://www.kaggle.com/dataturks/vehicle-number-plate-detection
Developed using Tensorflow 1.15

## File description:

- Data-Images.zip: Contains the images of the cars, number plates and annotations in `.txt` files (YOLO format)
- Data_preparation.ipynb: A notebook demonstrating the process of preparing the dataset (`.csv` files) for creating TFRecords (otherwise TensorFlow Object Detection API won't work)
- Detection.ipynb: A notebook demonstrating the process of detecting number plates from video feed and crop the image to be sent for OCR (this notebook has to be stored inside object_detection folder)
- Recognition.ipynb: A notebook demonstrating the process of recognising digits of the number plate, generated from Detection.ipynb
- Indian_Number_plates.json: Configuration file which contains image download paths and annotations
- exported_graph: Contains the inference graph in `.pb` and `.tflite` formats which can be used to run inference on both CPU platforms and on-device platforms
- label_map.pbtxt: Contains the encodings of the dataset classes which,in this case, is 1: **license_plate**
- ssd_mobilenet_v1_pets.config: Training and evaluation pipeline configuration file as needed by TensorFlow Object Detection API
- test.record & train.record: `TFRecords` files of testing and training sets respectively
- test_labels.csv & train_labels.csv: `.csv` files as required by the `generate_tfrecord.py` script
- requirements.txt: Python dependencies to install


To kick-start the model training process, Run the train file located in object_detection/legacy/
python train.py –-logtostderr –train_sir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

To save frozen graph for inference, Run the following command (To be able convert an inference graph to its `.tflite` variant you need to enable _quantization aware training_ and you can specify that in the `.config` file itself.) :
```
tflite_convert \
    --output_file=detect.tflite \
    --graph_def_file=frozen_inference_graph.pb \
    --input_shapes=1,300,300,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
    --inference_type=QUANTIZED_UINT8 \
    --mean_values=128 \
    --std_dev_values=128 \
    --change_concat_input_ranges=false \
    --allow_custom_ops
```
    
**Note**: A collection of pre-trained detection models are avaiable [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), if you want to train it with another model.
