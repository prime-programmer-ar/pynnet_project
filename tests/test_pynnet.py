"""
Comprehensive Test Suite for pynnet v1.0.1
==========================================
Tests forward/backward shapes, gradient correctness, numerical stability,
and end-to-end training for all layers, losses, and optimizers.

Run with:  python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pynnet.layers.dense import dense
from pynnet.layers.conv2d import conv2d
from pynnet.layers.maxpool2d import maxpool2d
from pynnet.layers.flatten import flatten
from pynnet.layers.dropout import dropout
from pynnet.layers.batchnorm import batchnorm
from pynnet.layers.lstm import lstm
from pynnet.activation import (
    activation, softmax,
    relu, relu_derivative, sigmoid, sigmoid_derivative,
    tanh, tanh_derivative, linear, linear_derivative,
    leaky_relu, leaky_relu_derivative, softplus, softplus_derivative,
)
from pynnet.loss import (
    mse, mse_derivative,
    binary_cross_entropy, binary_cross_entropy_derivative,
    categorical_cross_entropy, cce_derivative_with_softmax,
)
from pynnet.optimizer import SGD, Adam, RMSprop, Momentum
from pynnet.network import sequential


# ======================================================================
# 1. DENSE LAYER TESTS
# ======================================================================

class TestDenseLayer:
    def test_forward_shape(self):
        layer = dense(input_size=4, output_size=3)
        x = np.random.randn(1, 4)
        out = layer.forward(x)
        assert out.shape == (1, 3)

    def test_backward_shape(self):
        layer = dense(input_size=4, output_size=3)
        x = np.random.randn(1, 4)
        layer.forward(x)
        grad = np.random.randn(1, 3)
        dx = layer.backward(grad)
        assert dx.shape == (1, 4)
        assert layer.weights_gradient.shape == (4, 3)
        assert layer.biases_gradient.shape == (1, 3)

    def test_weight_inits(self):
        for method in ['he', 'xavier', 'lecun', 'random', 'orthogonal']:
            layer = dense(input_size=4, output_size=3, weight_init=method)
            assert layer.weights.shape == (4, 3)

    def test_identity_init(self):
        layer = dense(input_size=4, output_size=4, weight_init='identity')
        np.testing.assert_array_equal(layer.weights, np.eye(4))

    def test_identity_init_non_square_raises(self):
        with pytest.raises(ValueError):
            dense(input_size=4, output_size=3, weight_init='identity')

    def test_orthogonal_rectangular(self):
        # This was a bug in v0.0.2 - should not crash
        layer = dense(input_size=10, output_size=3, weight_init='orthogonal')
        assert layer.weights.shape == (10, 3)
        layer2 = dense(input_size=3, output_size=10, weight_init='orthogonal')
        assert layer2.weights.shape == (3, 10)


# ======================================================================
# 2. ACTIVATION TESTS
# ======================================================================

class TestActivations:
    def test_sigmoid_stability(self):
        """Sigmoid should not overflow for extreme values."""
        x = np.array([-1000.0, -500.0, 0.0, 500.0, 1000.0])
        result = sigmoid(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.isclose(result[2], 0.5)

    def test_softplus_stability(self):
        x = np.array([-100.0, 0.0, 50.0, 1000.0])
        result = softplus(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_relu_forward_backward(self):
        x = np.array([-2, -1, 0, 1, 2], dtype=float)
        assert np.allclose(relu(x), [0, 0, 0, 1, 2])
        assert np.allclose(relu_derivative(x), [0, 0, 0, 1, 1])

    def test_activation_layer_shape(self):
        act = activation(relu, relu_derivative)
        x = np.random.randn(1, 5)
        out = act.forward(x)
        assert out.shape == x.shape
        grad = act.backward(np.ones_like(out), None)
        assert grad.shape == x.shape

    def test_softmax_forward(self):
        sm = softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        out = sm.forward(x)
        assert np.isclose(np.sum(out), 1.0)
        assert np.all(out > 0)


# ======================================================================
# 3. LOSS FUNCTION TESTS
# ======================================================================

class TestLossFunctions:
    def test_mse(self):
        y_true = np.array([[1.0, 0.0]])
        y_pred = np.array([[0.5, 0.5]])
        loss = mse(y_true, y_pred)
        assert loss > 0
        grad = mse_derivative(y_true, y_pred)
        assert grad.shape == y_true.shape

    def test_bce_clipping(self):
        """BCE should not produce NaN/Inf for extreme predictions."""
        y_true = np.array([[1.0]])
        y_pred = np.array([[0.0]])  # Would cause log(0) without clipping
        loss = binary_cross_entropy(y_true, y_pred)
        assert not np.isnan(loss) and not np.isinf(loss)

    def test_cce_has_log(self):
        """Regression test for the v0.0.2 bug where np.log was missing."""
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.7, 0.2, 0.1]])
        loss = categorical_cross_entropy(y_true, y_pred)
        expected = -np.log(0.7)
        assert np.isclose(loss, expected, atol=1e-6)

    def test_cce_derivative_with_softmax_shape(self):
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.7, 0.2, 0.1]])
        grad = cce_derivative_with_softmax(y_true, y_pred)
        assert grad.shape == y_true.shape


# ======================================================================
# 4. CONV2D TESTS
# ======================================================================

class TestConv2D:
    def test_forward_shape(self):
        layer = conv2d(num_filters=4, kernel_size=3, input_channels=1)
        x = np.random.randn(1, 8, 8)
        out = layer.forward(x)
        assert out.shape == (4, 6, 6)  # (8-3)/1 + 1 = 6

    def test_forward_with_padding(self):
        layer = conv2d(num_filters=2, kernel_size=3, input_channels=1, padding=1)
        x = np.random.randn(1, 8, 8)
        out = layer.forward(x)
        assert out.shape == (2, 8, 8)  # same padding

    def test_backward_shape(self):
        layer = conv2d(num_filters=4, kernel_size=3, input_channels=1)
        x = np.random.randn(1, 8, 8)
        out = layer.forward(x)
        grad = np.random.randn(*out.shape)
        dx = layer.backward(grad)
        assert dx.shape == x.shape
        assert layer.weights_gradient.shape == layer.weights.shape


# ======================================================================
# 5. MAXPOOL2D TESTS
# ======================================================================

class TestMaxPool2D:
    def test_forward_shape(self):
        layer = maxpool2d(pool_size=2)
        x = np.random.randn(4, 6, 6)
        out = layer.forward(x)
        assert out.shape == (4, 3, 3)

    def test_backward_shape(self):
        layer = maxpool2d(pool_size=2)
        x = np.random.randn(4, 6, 6)
        out = layer.forward(x)
        grad = np.random.randn(*out.shape)
        dx = layer.backward(grad)
        assert dx.shape == x.shape


# ======================================================================
# 6. FLATTEN TESTS
# ======================================================================

class TestFlatten:
    def test_forward_backward(self):
        layer = flatten()
        x = np.random.randn(4, 3, 3)
        out = layer.forward(x)
        assert out.shape == (1, 36)
        grad = np.random.randn(1, 36)
        dx = layer.backward(grad)
        assert dx.shape == (4, 3, 3)


# ======================================================================
# 7. DROPOUT TESTS
# ======================================================================

class TestDropout:
    def test_training_mode(self):
        np.random.seed(42)
        layer = dropout(rate=0.5)
        layer.training = True
        x = np.ones((1, 100))
        out = layer.forward(x)
        # Some values should be zero
        assert np.any(out == 0.0)

    def test_inference_mode(self):
        layer = dropout(rate=0.5)
        layer.training = False
        x = np.ones((1, 100))
        out = layer.forward(x)
        np.testing.assert_array_equal(out, x)


# ======================================================================
# 8. BATCHNORM TESTS
# ======================================================================

class TestBatchNorm:
    def test_forward_shape(self):
        layer = batchnorm(num_features=4)
        x = np.random.randn(1, 4)
        out = layer.forward(x)
        assert out.shape == (1, 4)

    def test_backward_shape(self):
        layer = batchnorm(num_features=4)
        x = np.random.randn(1, 4)
        layer.forward(x)
        grad = np.random.randn(1, 4)
        dx = layer.backward(grad)
        assert dx.shape == (1, 4)


# ======================================================================
# 9. LSTM TESTS
# ======================================================================

class TestLSTM:
    def test_forward_shape_sequences(self):
        layer = lstm(input_size=5, hidden_size=10, return_sequences=True)
        x = np.random.randn(8, 5)  # seq_len=8
        out = layer.forward(x)
        assert out.shape == (8, 10)

    def test_forward_shape_last(self):
        layer = lstm(input_size=5, hidden_size=10, return_sequences=False)
        x = np.random.randn(8, 5)
        out = layer.forward(x)
        assert out.shape == (1, 10)

    def test_backward_shape(self):
        layer = lstm(input_size=5, hidden_size=10, return_sequences=True)
        x = np.random.randn(8, 5)
        out = layer.forward(x)
        grad = np.random.randn(8, 10)
        dx = layer.backward(grad)
        assert dx.shape == (8, 5)


# ======================================================================
# 10. OPTIMIZER TESTS
# ======================================================================

class TestOptimizers:
    def _make_layer(self):
        l = dense(input_size=3, output_size=2)
        l.forward(np.random.randn(1, 3))
        l.backward(np.random.randn(1, 2))
        return l

    def test_sgd(self):
        opt = SGD(learning_rate=0.01)
        l = self._make_layer()
        w_before = l.weights.copy()
        opt.update(l)
        assert not np.array_equal(l.weights, w_before)

    def test_adam(self):
        opt = Adam(learning_rate=0.01)
        l = self._make_layer()
        opt.step()
        w_before = l.weights.copy()
        opt.update(l)
        assert not np.array_equal(l.weights, w_before)

    def test_rmsprop(self):
        opt = RMSprop(learning_rate=0.01)
        l = self._make_layer()
        w_before = l.weights.copy()
        opt.update(l)
        assert not np.array_equal(l.weights, w_before)

    def test_momentum(self):
        opt = Momentum(learning_rate=0.01)
        l = self._make_layer()
        w_before = l.weights.copy()
        opt.update(l)
        assert not np.array_equal(l.weights, w_before)


# ======================================================================
# 11. END-TO-END: MLP XOR
# ======================================================================

class TestEndToEnd:
    def test_xor_mlp(self):
        np.random.seed(42)
        X = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]], dtype=float)
        y = np.array([[[0]], [[1]], [[1]], [[0]]], dtype=float)

        model = sequential()
        model.add(dense(input_size=2, output_size=8, weight_init='he'))
        model.add(activation(relu, relu_derivative))
        model.add(dense(input_size=8, output_size=1, weight_init='xavier'))
        model.add(activation(sigmoid, sigmoid_derivative))
        model.compile(loss=mse, loss_derivative=mse_derivative,
                      optimizer=Adam(learning_rate=0.01))
        model.fit(X, y, epochs=500, verbose=False)

        preds = model.predict(X)
        for i in range(4):
            assert abs(preds[i][0][0] - y[i][0][0]) < 0.3, \
                f"XOR prediction {i} too far off: {preds[i][0][0]}"

    def test_cnn_forward_backward(self):
        """Test a minimal CNN pipeline doesn't crash."""
        np.random.seed(42)
        x = np.random.randn(1, 6, 6)  # 1 channel 6x6

        c = conv2d(num_filters=2, kernel_size=3, input_channels=1)
        p = maxpool2d(pool_size=2)
        f = flatten()
        d = dense(input_size=2 * 2 * 2, output_size=3)

        # Forward
        out = c.forward(x)       # (2, 4, 4)
        out = p.forward(out)     # (2, 2, 2)
        out = f.forward(out)     # (1, 8)
        out = d.forward(out)     # (1, 3)
        assert out.shape == (1, 3)

        # Backward
        grad = np.ones((1, 3))
        grad = d.backward(grad)
        grad = f.backward(grad)
        grad = p.backward(grad)
        grad = c.backward(grad)
        assert grad.shape == (1, 6, 6)

    def test_lstm_training(self):
        """Test LSTM can reduce loss on a simple pattern."""
        np.random.seed(42)
        seq = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        target = np.array([[0, 1]], dtype=float)

        rnn = lstm(input_size=2, hidden_size=4, return_sequences=False)
        d = dense(input_size=4, output_size=2)

        opt = Adam(learning_rate=0.01)
        losses = []
        for epoch in range(50):
            out = rnn.forward(seq)
            out = d.forward(out)
            loss = mse(target, out)
            losses.append(loss)
            grad = mse_derivative(target, out)
            grad = d.backward(grad)
            grad = rnn.backward(grad)
            opt.step()
            opt.update(d)
            opt.update(rnn)

        assert losses[-1] < losses[0], "LSTM loss should decrease"

    def test_model_save_load(self):
        """Test weight persistence."""
        import tempfile, os
        model = sequential()
        model.add(dense(input_size=2, output_size=3))
        model.add(dense(input_size=3, output_size=1))

        x = np.random.randn(2, 1, 2)
        preds_before = model.predict(x)

        path = os.path.join(tempfile.gettempdir(), 'test_pynnet_weights.npz')
        model.save_weights(path)

        model2 = sequential()
        model2.add(dense(input_size=2, output_size=3))
        model2.add(dense(input_size=3, output_size=1))
        model2.load_weights(path)

        preds_after = model2.predict(x)
        np.testing.assert_array_almost_equal(preds_before, preds_after)
        os.remove(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

# ======================================================================
# 12. METRICS TESTS
# ======================================================================

class TestMetrics:
    def test_accuracy_binary(self):
        from pynnet.metrics import accuracy
        y_true = np.array([[0], [1], [1], [0]])
        y_pred = np.array([[0.1], [0.9], [0.8], [0.4]])
        assert accuracy(y_true, y_pred) == 1.0

        y_pred_bad = np.array([[0.9], [0.1], [0.1], [0.9]])
        assert accuracy(y_true, y_pred_bad) == 0.0

    def test_accuracy_categorical(self):
        from pynnet.metrics import accuracy
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])
        assert accuracy(y_true, y_pred) == 1.0

        y_pred_bad = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.4, 0.3, 0.3]])
        assert accuracy(y_true, y_pred_bad) == 0.0
