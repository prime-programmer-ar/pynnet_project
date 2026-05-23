"""
LSTM (Long Short-Term Memory) Layer
=====================================

Implements a single-layer LSTM cell that processes sequences
one timestep at a time and returns the full sequence of hidden states.
"""

import numpy as np
from .base import layer


class lstm(layer):
    """Long Short-Term Memory (LSTM) recurrent layer.

    Processes a sequence of shape ``(seq_len, input_size)`` and
    produces hidden states of shape ``(seq_len, hidden_size)``.

    The LSTM gate equations follow the standard formulation:

    .. math::
        f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)   \\quad \\text{(forget gate)}
        i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i)   \\quad \\text{(input gate)}
        \\tilde{c}_t = \\tanh(W_c [h_{t-1}, x_t] + b_c)  \\quad \\text{(candidate)}
        o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o)   \\quad \\text{(output gate)}
        c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t
        h_t = o_t \\odot \\tanh(c_t)

    Args:
        input_size (int): Dimensionality of each input vector.
        hidden_size (int): Number of LSTM hidden units.
        return_sequences (bool): If ``True``, return hidden states for
            all timesteps ``(seq_len, hidden_size)``. If ``False``,
            return only the last hidden state ``(1, hidden_size)``.

    Attributes:
        weights (np.ndarray): Concatenated weight matrix of shape
            ``(input_size + hidden_size, 4 * hidden_size)`` for all gates.
        biases (np.ndarray): Concatenated bias vector of shape
            ``(1, 4 * hidden_size)``.
    """

    def __init__(self, input_size, hidden_size, return_sequences=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        concat_size = input_size + hidden_size

        # Xavier init for all gates combined: [W_f | W_i | W_c | W_o]
        std = np.sqrt(2.0 / (concat_size + hidden_size))
        self.weights = np.random.randn(concat_size, 4 * hidden_size) * std
        self.biases = np.zeros((1, 4 * hidden_size))

        # Bias initialization: set forget gate bias to 1.0
        # This helps the LSTM learn long-term dependencies early in training
        self.biases[0, :hidden_size] = 1.0

    @staticmethod
    def _sigmoid(x):
        """Numerically stable sigmoid."""
        result = np.zeros_like(x, dtype=np.float64)
        pos = x >= 0
        neg = ~pos
        result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[neg])
        result[neg] = exp_x / (1.0 + exp_x)
        return result

    def forward(self, input_data):
        """Forward pass: process the input sequence through the LSTM.

        Args:
            input_data (np.ndarray): Input sequence of shape
                ``(seq_len, input_size)``.

        Returns:
            np.ndarray: Hidden states. Shape depends on
                ``return_sequences``.
        """
        self.input = input_data
        seq_len = input_data.shape[0]
        H = self.hidden_size

        # Initialize hidden state and cell state
        h = np.zeros((1, H))
        c = np.zeros((1, H))

        # Cache for backward pass
        self.cache = []
        self.h_states = [h.copy()]
        self.c_states = [c.copy()]

        outputs = []

        for t in range(seq_len):
            x_t = input_data[t:t+1, :]  # (1, input_size)
            concat = np.concatenate([h, x_t], axis=1)  # (1, H + input_size)

            gates = np.dot(concat, self.weights) + self.biases  # (1, 4H)

            # Split into four gates
            f_gate = self._sigmoid(gates[:, :H])           # forget
            i_gate = self._sigmoid(gates[:, H:2*H])        # input
            c_cand = np.tanh(gates[:, 2*H:3*H])            # candidate
            o_gate = self._sigmoid(gates[:, 3*H:])          # output

            c = f_gate * c + i_gate * c_cand
            h = o_gate * np.tanh(c)

            self.cache.append({
                'concat': concat,
                'gates_raw': gates,
                'f': f_gate, 'i': i_gate,
                'c_cand': c_cand, 'o': o_gate,
                'c_prev': self.c_states[-1],
            })
            self.h_states.append(h.copy())
            self.c_states.append(c.copy())
            outputs.append(h.copy())

        if self.return_sequences:
            return np.vstack(outputs)  # (seq_len, H)
        else:
            return outputs[-1]  # (1, H)

    def backward(self, output_gradient, learning_rate=None):
        """Backward pass: backpropagation through time (BPTT).

        Args:
            output_gradient (np.ndarray): Gradient from upstream.
                Shape ``(seq_len, hidden_size)`` if ``return_sequences``,
                or ``(1, hidden_size)`` otherwise.
            learning_rate: Unused.

        Returns:
            np.ndarray: Gradient w.r.t. the input sequence,
                shape ``(seq_len, input_size)``.
        """
        seq_len = self.input.shape[0]
        H = self.hidden_size

        self.weights_gradient = np.zeros_like(self.weights)
        self.biases_gradient = np.zeros_like(self.biases)

        dh_next = np.zeros((1, H))
        dc_next = np.zeros((1, H))
        input_gradient = np.zeros_like(self.input)

        for t in reversed(range(seq_len)):
            cache = self.cache[t]

            if self.return_sequences:
                dh = output_gradient[t:t+1, :] + dh_next
            else:
                if t == seq_len - 1:
                    dh = output_gradient + dh_next
                else:
                    dh = dh_next

            c_t = self.c_states[t + 1]
            tanh_c = np.tanh(c_t)

            # Output gate
            do = dh * tanh_c
            do_raw = do * cache['o'] * (1 - cache['o'])

            # Cell state
            dc = dh * cache['o'] * (1 - tanh_c ** 2) + dc_next

            # Forget gate
            df = dc * cache['c_prev']
            df_raw = df * cache['f'] * (1 - cache['f'])

            # Input gate
            di = dc * cache['c_cand']
            di_raw = di * cache['i'] * (1 - cache['i'])

            # Candidate
            dc_cand = dc * cache['i']
            dc_cand_raw = dc_cand * (1 - cache['c_cand'] ** 2)

            # Concatenate gate gradients
            dgates = np.concatenate(
                [df_raw, di_raw, dc_cand_raw, do_raw], axis=1
            )  # (1, 4H)

            # Weight and bias gradients
            self.weights_gradient += np.dot(cache['concat'].T, dgates)
            self.biases_gradient += dgates

            # Gradient w.r.t. concatenated input
            d_concat = np.dot(dgates, self.weights.T)  # (1, H + input_size)

            dh_next = d_concat[:, :H]
            input_gradient[t:t+1, :] = d_concat[:, H:]

            # Gradient for cell state from forget gate
            dc_next = dc * cache['f']

        return input_gradient
