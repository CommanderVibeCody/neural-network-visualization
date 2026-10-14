"""Numpy feed-forward network for a whole population at once.

Genome layout = Keras get_weights() order flattened: [W1, b1, W2, b2, ...],
with W shaped (in, out) exactly like a tf.keras Dense kernel.
"""
import numpy as np

ACT = {
    "relu": lambda x: np.maximum(x, 0),
    "tanh": np.tanh,
    "sigmoid": lambda x: 1 / (1 + np.exp(-np.clip(x, -30, 30))),
    "elu": lambda x: np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1),
    "leaky_relu": lambda x: np.where(x > 0, x, 0.2 * x),
    "selu": lambda x: 1.0507 * np.where(x > 0, x, 1.67326 * (np.exp(np.minimum(x, 0)) - 1)),
    "linear": lambda x: x,
}


class Brain:
    def __init__(self, sizes, hidden_acts, out_acts):
        """sizes = [n_in, h1, ..., 2]; hidden_acts per hidden layer; out_acts per output neuron."""
        self.sizes = list(sizes)
        self.hidden_acts = list(hidden_acts)
        self.out_acts = list(out_acts)
        self.shapes = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.shapes += [(a, b), (b,)]
        self.n_params = int(sum(np.prod(s) for s in self.shapes))

    def random_genomes(self, n, rng):
        parts = []
        for s in self.shapes:
            if len(s) == 2:
                parts.append(rng.normal(0, 1 / np.sqrt(s[0]), (n, s[0] * s[1])))
            else:
                parts.append(np.zeros((n, s[0])))
        return np.hstack(parts)

    def unpack(self, genomes):
        """Return list of (W (N,in,out), b (N,out)) views."""
        layers, o = [], 0
        n = genomes.shape[0]
        for i in range(0, len(self.shapes), 2):
            a, b = self.shapes[i]
            W = genomes[:, o:o + a * b].reshape(n, a, b); o += a * b
            bias = genomes[:, o:o + b]; o += b
            layers.append((W, bias))
        return layers

    def forward(self, genomes, x, return_acts=False):
        """x: (N, n_in). Returns outputs (N, n_out) and optionally all layer activations."""
        acts = [x]
        h = x
        layers = self.unpack(genomes)
        for li, (W, b) in enumerate(layers):
            z = np.einsum("ni,nio->no", h, W) + b
            if li < len(layers) - 1:
                h = ACT[self.hidden_acts[li]](z)
            else:
                h = np.stack([ACT[a](z[:, j]) for j, a in enumerate(self.out_acts)], axis=1)
            acts.append(h)
        return (h, acts) if return_acts else h

    def weight_matrices(self, genome):
        """Kernels of one genome (for drawing), Keras-shaped (in, out)."""
        return [W[0] for W, _ in self.unpack(genome[None, :])]
