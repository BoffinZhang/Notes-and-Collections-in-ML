## Codes for *Deep Learning & Neural Network* Chapter 1
A single layer perceptron

### Changes
Since the original code are written in python 2, several problems may trouble you while using python 3.
Here I modified the code to run it with python 3.

1. In `Network.py`, I replaced all `xrange` with `range`. In python3, `xrange` merged into `range`.

2. In `mnist_loader.py`, all the data has to be returned in `list` type rather than `zip` type. Since len(zip) is no longer supported in python3.

```python
# /src/mnist_loader.py
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)
```