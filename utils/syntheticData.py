import numpy as np


class Flip:
    """"Provably End-to-end Label-noise Learning without Anchor Points"
        <https://arxiv.org/abs/2102.02400>
        Flipping the label in given noisy probability by 3 methods(symmetric, asymmetric, pair).
        Symmetric will generate the same noisy prob except diagonal;
        Asymmetric uses uniform distribution to generate asymmetric noisy prob;
        Pair will set 1 - noise_rate to diagonal and noise_rate to position of diagonal + 1.

        Args:
        noise_rate (float): The probability of flipping into other classes.
        num_classes (int): The class number of dataset.
        random_state (int): Random Seed to fix the same random result in  multi experiments.
    """
    def __init__(self, noise_rate: float, num_classes: int, random_state: int):
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.random_state = random_state

    def _noisify_label(self, noise_matrix, labels):
        """
        According to the noise_matrix to flip clean label randoml, return the corrupt label and
        the true transition matrix
        """
        # make sure the sum of each row prob is 1.0
        np.testing.assert_array_almost_equal(noise_matrix.sum(axis=1), np.ones(self.num_classes))

        corrupt_label = labels.copy()

        stas = np.unique(labels, return_counts=True)[1]
        true_transition_matrix = np.zeros_like(noise_matrix)

        rng = np.random.RandomState(self.random_state)
        for idx in np.arange(corrupt_label.shape[0]):
            i = labels[idx]
            flipped_label = rng.multinomial(1, noise_matrix[i, :], 1)
            flipped_label = np.argmax(flipped_label)
            true_transition_matrix[i, flipped_label] += 1
            corrupt_label[idx] = flipped_label
        true_transition_matrix = true_transition_matrix / stas[:, np.newaxis]
        print(true_transition_matrix)
        return corrupt_label, true_transition_matrix

    def symmetric(self, label):
        T = np.ones([self.num_classes, self.num_classes])
        T = self.noise_rate / (self.num_classes - 1) * T
        np.fill_diagonal(T, 1-self.noise_rate)

        corrupt_label, true_transition_matrix = self._noisify_label(T, label)
        return corrupt_label, true_transition_matrix

    def asymmetric(self, label):
        T = np.random.uniform(low=0.1, high=1., size=(self.num_classes, self.num_classes))
        np.fill_diagonal(T, 0.0)
        T = T / T.sum(axis=1, keepdims=True) * self.noise_rate  # weighted * nose_rate
        np.fill_diagonal(T, 1-self.noise_rate)

        corrupt_label, true_transition_matrix = self._noisify_label(T, label)
        return corrupt_label, true_transition_matrix

    def pair(self, label):
        T = np.zeros([self.num_classes, self.num_classes])
        np.fill_diagonal(T, 1 - self.noise_rate)
        row = [i for i in range(self.num_classes)]
        col = [i if i < self.num_classes else 0 for i in range(1, self.num_classes+1)]
        T[(row, col)] = self.noise_rate

        corrupt_label, true_transition_matrix = self._noisify_label(T, label)
        return corrupt_label, true_transition_matrix







