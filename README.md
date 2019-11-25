# bcg-hr-dl
Keras implementations of deep learning architectures used in our publication on
heart rate estimation from BCG data.

The models are implemented with `tf.keras` of Tensorflow 1.13, but should work
with most versions of Tensorflow.


## Usage

You can install all required packages (except Tensorflow) with the provided
requirements file:

```bash
git clone https://github.com/SamProell/bcg-hr-dl.git
cd bcg-hr-dl
pip install -r requirements.txt
```

The example jupyter notebooks highlight how to obtain and train the models.
In essence, you can import any network from the models subfolder and use
`create` to get a compiled Keras model:

```python
# from models import stacked_cnn_rnn_improved as architecture
from models import baseline_fcn as architecture

patchsize, n_channels = 400, 1
model = architecture.create(input_shape=(patchsize, n_channels), enlarge=1)
model.fit(x_data, y_data, batch_size=32)
```
