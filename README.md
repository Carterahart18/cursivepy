# CursivePy

![](assets/demo.gif)

## About

"CursivePy" is a project that implements a Neural Network and teaches it to recognize simple handwritten digits. You can configure the network as you please with custom hidden layers, training batch sizes and iterations. You can save and load network weights and biases stored from pervious runs. Your network can then be tested against your own handwriting using a drawing program which simulates the training data.

The 28 x 28 handwritten digit recognition problem is often considered a "hello world" program for neural networks. It is a simple, well defined challenged that can be solved easily via a properly trained Neural Network but is very difficult to solve with conventional discrete programming methods.

## Training the network

To create a configuration of the network, run the following:

```shell
python src/train.py
```

This script will automatically download the training and testing data from [MNSIT](https://en.wikipedia.org/wiki/MNIST_database) and train the network to decipher the images in a dry run using the default configurations.

To save the network for use in testing, add the `-s` flag.

To configure the number of testing iterations use `-i <num>`.

To configure the image batch size of each iteration use `-b <num>`.

To configure the hidden layers of the network use `-l [layer_size...]`.

For more information, run `python src/train.py -h`.

Example output:

```
~/Desktop/CursivePy/src python train.py -i 20000 -b 250 -l 100 100 -s

  Training Accuracy  |  Testing Accuracy  |  Percent Complete
---------------------+--------------------+--------------------
       09.93%        |       10.32%       |       00.00%
       34.44%        |       35.03%       |       03.33%
       83.02%        |       83.35%       |       06.66%
       89.58%        |       90.09%       |       09.99%
       92.14%        |       92.20%       |       13.32%
       93.97%        |       93.67%       |       16.65%
       94.88%        |       94.40%       |       19.98%
       94.68%        |       94.16%       |       23.31%
       95.81%        |       95.22%       |       26.64%
        ...          |        ...         |        ...
       98.51%        |       96.84%       |       86.58%
       98.72%        |       96.94%       |       89.91%
       98.76%        |       96.98%       |       93.24%
       98.37%        |       96.55%       |       96.57%
       98.73%        |       96.98%       |       99.90%

Training Complete !

Would you like to save this network (y/n)?
```

## CursivePy Training Program

Once you have a `network.pkl` file saved to disk, you can run the main CursivePy application:

```shell
python src/main.py
```

By default it will find the file saved in the dist folder.

This GUI application has a 28 x 28 canvas which you can draw on have have the network guess against in real time. If the network's prediction is right its guess will appear in green and if wrong it will appear in red. The program assumes that the anser is the target number printed at the top of the window.

If you mis-draw you can clear the page. If you want the network to learn against your example, click "Submit". You will get a new random target number and the network will have back-propogated the its results, adjusting each neuron to better match your example.

To configure the rate of learning use `-l <num between 0 and 1>`. The default is 0.05. A higher learning rate means the network will change more drastically per submission, but this can lead to the network overfitting to one or a few examples.

For more information, run `python src/main.py -h`.

## Batch Testing

After you have saved a network to disk you can run the following to test mass batches of testing data aginst the network to measure overall performance:

```shell
python src/test.py
```

To load a specific network for use in testing, add the `-f <path>` flag.

To configure the number of testing iterations use `-i <num>`.

To configure the image batch size of each iteration use `-b <num>`.

For more information, run `python src/test.py -h`.
