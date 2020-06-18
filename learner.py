import numpy as np

class learner:
    def __init__(self, Q_0, H, eta):
        self.Q_0 = Q_0
        self.H = H
        self.Q = Q_0
        self.eta = eta
    #enddef

    def drawn_hypothesis(self):
        Q_normalized = self.normalize_vector(self.Q)
        index_of_h = np.random.choice(np.arange(0, len(Q_normalized)), p=Q_normalized)
        return self.H[index_of_h]
    #enddef

    def update_scoring_function(self, list_of_examples):
        error_array = []
        for example in list_of_examples:
            for i, h in enumerate(self.H):
                self.Q[i] *= self.likelihood(h, example)
        return
    #enddef

    def likelihood(self, h, example):
        if self.prediction_is_correct(h, example):
            return 1
        return 1 - self.eta
    #enndef

    def prediction_is_correct(self, h, example):
        if (np.dot(h[:-1], example.x_s) - h[-1]) * example.y_s >= 0:
            return True
        else:
            return False
    #enndef

    def expected_error(self, list_of_examples):
        error = 0
        normalized_scoring_function = self.normalize_vector(self.Q)
        for i, h in enumerate(self.H):
            error += normalized_scoring_function[i] * self.error_given_h(h, list_of_examples)
        return error
    #enndef

    def error_given_h(self, h, list_of_examples):
        error_count = 0
        for example in list_of_examples:
            if not self.prediction_is_correct(h, example):
                error_count += 1
        return error_count / len(list_of_examples)
    #enddef

    def normalize_vector(self, P):
        return P / sum(P)
    #enddef


if __name__ == "__main__":
    pass

