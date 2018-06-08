# leaves-transfer-learning
Transfer learning on ResNet50 pre-trained on ILSVRC for Flavia 33 species leaves dataset.

<p>
Using Keras high API with
TensorFlow backend. Transfer Learning only on Softmax layer. 
The model achieves impressive results.
</p>


<u><h4>Dataset</h4></u>
<p>
<b>Flavia Leaves Dataset</b>
33 species represented by leaves images.
Data contains 1907 images which is split initialy
as 80:20 for train:test.
Train set split 80:20 again for
Cross-Validaiton.
Different species images seperated under sub-folders of "data/"
folder.
Source: http://flavia.sourceforge.net/
</p>


<u><h4>Metrics</h4></u>
<p>
<u>Loss</u>:
Sparse Categorical Cross-Enrtropy<br>
<u>Accuracy</u>:
Mean accuracy rate on all predictions
<br>
  </p>
<u><h4>Optimizer</h4></u>
<p>Adam with Learning rate 0.0001</p>

<hr>
Model achieves 0.992 accuracy on test set with 0.14 Loss.








