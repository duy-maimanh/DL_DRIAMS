# DL_DRIAMS Classification

> Our project is designed to classify species as either Susceptible or Resistant using advanced deep learning techniques. We employ both Basic Deep Neural Networks (DNN) and Convolutional Neural Networks (CNN) in our classification tasks.

## Data DL_DRIAMS
To download dataset, run:
```
download.sh
```
You may need using `chmod u+r+x download.sh` before run shell script above for by pass permission.

## PrePorcessing dataset
The code auto preprocessing data for first run with each species and save it as a pkl file in `processed_data`. You do not need to do anything.
## Training and Testing
To train the model on DNN and obtain the results on the testing set, run:
```
python main.py --model="dnn"
```
or train the model on CNN, run:
```
python main.py --model="cnn"
```
The model weight will be saved in `weight` folder.

## Visualization

To code will automatically visualization after save the weight. You can find it in `visualization` folder.

## Classifiy with specific medicine
To classify with other medicine you can replace `medicine` with the medicine you want. For example run:
```
python main.py --model="dnn" --medicine="Meropenem"
```
or
```
python main.py --model="cnn" --medicine="Ciprofloxacin"
```
<img src="image/dnn_1734577166143_ROC.png" width="40%" />

<img src="image/cnn_1734578784774_ROC.png" width="40%" />