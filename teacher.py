import numpy as np
import copy
import utils
import learner

class teacher:
    def __init__(self, H, examples, Q_0, eta_learner, delta, eps,
                 teacher_type, max_iter, mu, top_k=10):
        self.H = H
        self.examples = examples
        self.Q_0 = Q_0
        self.Q_of_h_given_S = copy.deepcopy(Q_0)
        self.eta_learner = eta_learner
        self.delta = delta
        self.teacher_type = teacher_type
        if teacher_type == "greedy":
            self.eta_teacher = eta_learner
        elif teacher_type == "plus_delta":
            self.eta_teacher = eta_learner + self.delta
        elif teacher_type == "minus_delta":
            self.eta_teacher = eta_learner - self.delta
        elif teacher_type == "random":
            self.eta_teacher = eta_learner
        elif teacher_type == "approximate_Q":
            self.eta_teacher = eta_learner
        elif teacher_type == "noise_feature" or teacher_type == "limited_ground_truth":
            self.eta_teacher = eta_learner

        else:
            print("Wrong teacher type -- ", self.teacher_type)
            print("Please choose one of the following: [greedy, plus_delta, minus_delta, random]" )
            exit(0)
        self.learner = learner.learner(Q_0=Q_0, H=H, eta=self.eta_teacher)
        self.eps = eps
        self.max_iter = max_iter
        self.mu = mu
        self.top_k = top_k
        self.h_star_index = utils.find_h_star_index_given_examples(H, examples)
        self.h_star = self.H[self.h_star_index]
        self.examples_train = utils.remove_all_inconsistent_examples(self.h_star, examples)
        self.list_of_error_h = utils.error_for_every_h(H, self.examples_train)
        if teacher_type == "approximate_Q":
            self.Q_0 = self.get_perturbed_initial_distribution()
        self.C_eps = self.get_C_eps()


    def get_selected_examples_for_demonstration(self):
        selected_examples = []
        iter = 0
        while True:
            print("iter = ", iter)
            maximum = -np.inf
            selected_example = None
            for example in self.examples_train:
                new_max_value = self.obj_function(example)
                if new_max_value > maximum:
                    selected_example = copy.deepcopy(example)
                    maximum = copy.deepcopy(new_max_value)
            #add selected example
            selected_examples.append(selected_example)
            self.Q_of_h_given_S = self.get_Q_of_h_given_S(selected_example)
            print("Example ID = ", selected_example.id)
            iter += 1
            if maximum > self.C_eps or iter > self.max_iter:
                break
        return selected_examples
    #enddef

    def get_C_eps(self):

        return np.dot(self.list_of_error_h, self.Q_0) - self.eps * self.Q_0[self.h_star_index]
    #enddef

    def get_perturbed_initial_distribution(self):
        delta, Q_0_perturbed = utils.get_delta_initD_given_mu(self.mu, self.H, self.examples_train, self.top_k)
        self.Q_0 = copy.deepcopy(Q_0_perturbed)
        self.delta = delta
        return self.Q_0
    #enddef

    def obj_function(self, example=None):
        if example is not None:
            Q_of_h_given_S = self.get_Q_of_h_given_S(example)
        else:
            Q_of_h_given_S = copy.deepcopy(self.Q_of_h_given_S)
        obj_func_value = np.dot((self.Q_0 - Q_of_h_given_S), self.list_of_error_h)
        return obj_func_value
    #enndef

    def get_Q_of_h_given_S(self, example):
        Q = copy.deepcopy(self.Q_of_h_given_S)
        for i, h in enumerate(self.H):
            Q[i] *= self.learner.likelihood(h, example)
        return Q
    #enddef

    def normalize_vector(self, P):
        return P / sum(P)
    #enddef


if __name__ == "__main__":
    pass






