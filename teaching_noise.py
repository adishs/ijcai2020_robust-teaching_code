import learner
import teacher
import dataset
import example
import utils
import plot
import plot_results
import numpy as np
import copy
np.set_printoptions(precision=5)


path_prefix = "input/"
features_file = path_prefix + "images-embedding.yaml"
hypotheses_file = path_prefix + "hypotheses_with-target.yaml"
ds = dataset.Dataset(features_file)


class teaching:
    def __init__(self, Q_0, hypotheses, examples, eta, delta, eps, teacher_type, exp_iter, mu, top_k):
        self.Q_0 = Q_0
        self.hypotheses = hypotheses
        self.examples = examples
        self.eta = eta
        self.delta = delta
        self.eps = eps
        self.teacher_type = teacher_type
        self.exp_iter = exp_iter
        self.accumulator = {}

        self.max_iter = 150
        self.mu = mu
        self.top_k = top_k


        self.learner = learner.learner(Q_0=Q_0, H=hypotheses, eta=eta)

        self.teacher = teacher.teacher(H=hypotheses, examples=examples, Q_0=Q_0,
                          eta_learner=eta, delta=delta, eps=eps, teacher_type=teacher_type, max_iter=self.max_iter,
                                       mu=self.mu, top_k=self.top_k)
    #enddef

def accumulator_function(tmp_dict, dict_accumulator):
    for key in tmp_dict:
        if key in dict_accumulator:
            dict_accumulator[key] += np.array(tmp_dict[key])
        else:
            dict_accumulator[key] = np.array(tmp_dict[key])
    return dict_accumulator
#enddef

def calculate_average(dict_accumulator, number_of_iterations):
    for key in dict_accumulator:
        dict_accumulator[key] = dict_accumulator[key]/number_of_iterations
    return dict_accumulator
#enddef



if __name__ == "__main__":

    final_dict_accumulator = {}
    number_of_iterations = 10

    exp_iter = 0
    #input all hypotheses
    hypotheses = utils.get_hypotheses(hypotheses_file)
    #get h_star
    h_star = hypotheses[0]
    #get all examples
    examples = utils.input_examples(ds)
    #initial distribution
    Q_0 = np.ones(len(hypotheses)) / len(hypotheses)


    for iteration in range(0, number_of_iterations):

        ############Approxomate Q teacher################
        teacher_type = "approximate_Q" #minus_delta, plus_delta, approximate_Q
        mu_approximate_Q = np.arange(0, 1, 0.05)

        picked_examples_by_zero_out_teacher = None
        array_error = []
        accumulator = {}
        n_random = None
        for i, mu in enumerate(mu_approximate_Q):
            # teachers_examples = copy.deepcopy(examples)
            mu = np.round(mu, 2)
            print(mu)
            teaching_object = teaching(Q_0=Q_0, hypotheses=hypotheses,
                                      examples=examples, eta=0.5, delta=0, eps=0.001,
                                      teacher_type=teacher_type, exp_iter=exp_iter, mu=mu, top_k=10)
            print("=====Teaching start ---{}---====delta={}====".format(str.upper(teacher_type), teaching_object.teacher.delta))
            # print(teaching_object.teacher.Q_0)
            # input("ENter")
            if teaching_object.teacher.delta is None:
                continue
            teaching_examples = teaching_object.teacher.get_selected_examples_for_demonstration()
            learner_appr_Q = copy.deepcopy(teaching_object.learner)
            learner_appr_Q.update_scoring_function(teaching_examples)
            delta = np.round(teaching_object.teacher.delta, 2)
            accumulator["{}_delta={}_error".format(teacher_type, str(delta))] = [learner_appr_Q.expected_error(examples)]
            accumulator["{}_delta={}_n_examples".format(teacher_type, str(delta))] = [len(teaching_examples)]


        ############ Plus_Delta Teacer ###################################
        teacher_type = "plus_delta" #minus_delta, plus_delta, approximate_Q
        deltas_plus = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        picked_examples_by_zero_out_teacher = None

        n_random = None
        for i, delta in enumerate(deltas_plus):
            # teachers_examples = copy.deepcopy(examples)
            teaching_object = teaching(Q_0=Q_0, hypotheses=hypotheses,
                                      examples=examples, eta=0.5, delta=delta, eps=0.001,
                                      teacher_type=teacher_type, exp_iter=exp_iter, mu=[], top_k=10)
            print("=====Teaching start ---{}---====delta={}====".format(str.upper(teacher_type), delta))
            teaching_examples = teaching_object.teacher.get_selected_examples_for_demonstration()
            if delta == deltas_plus[0]:
                n_random = len(teaching_examples)
            elif delta == deltas_plus[-1]:
                picked_examples_by_zero_out_teacher = teaching_examples
            learner_plus_delta = copy.deepcopy(teaching_object.learner)
            learner_plus_delta.update_scoring_function(teaching_examples)
            accumulator["{}_delta={}_error".format(teacher_type, str(delta))] = [learner_plus_delta.expected_error(examples)]
            accumulator["{}_delta={}_n_examples".format(teacher_type, str(delta))] = [len(teaching_examples)]



        ######### Delta_Minus Teacher ###################
        teacher_type = "minus_delta"
        deltas_minus = [0, 0.1, 0.2, 0.3, 0.4, 0.45]
        for i, delta in enumerate(deltas_minus):
            # teachers_examples = copy.deepcopy(examples)
            teaching_object = teaching(Q_0=Q_0, hypotheses=hypotheses,
                                      examples=examples, eta=0.5, delta=delta, eps=0.001,
                                      teacher_type=teacher_type, exp_iter=exp_iter, mu=[], top_k=10)
            print("=====Teaching start ---{}---====delta={}====".format(str.upper(teacher_type), delta))
            teaching_examples = teaching_object.teacher.get_selected_examples_for_demonstration()
            learner_plus_delta = copy.deepcopy(teaching_object.learner)
            learner_plus_delta.update_scoring_function(teaching_examples)
            accumulator["{}_delta={}_error".format(teacher_type, str(delta))] = [learner_plus_delta.expected_error(examples)]
            accumulator["{}_delta={}_n_examples".format(teacher_type, str(delta))] = [len(teaching_examples)]

        ############# Random Teacher ######################

        teachers_examples = copy.deepcopy(examples)
        teaching_object = teaching(Q_0=Q_0, hypotheses=hypotheses,
                                  examples=examples, eta=0.5, delta=0, eps=0.001,
                                  teacher_type=teacher_type, exp_iter=exp_iter, mu=[], top_k=10)
        learner_random_0_5 = copy.deepcopy(teaching_object.learner)
        learner_random_1 = copy.deepcopy(teaching_object.learner)
        learner_random_1_5 = copy.deepcopy(teaching_object.learner)
        n_random_0_5 = int(np.round(n_random * 0.5))
        n_random_1 = int(np.round(n_random * 1))
        n_random_1_5 = int(np.round(n_random * 1.5))

        examples_random_0_5 = np.random.choice(examples, n_random_0_5)
        examples_random_1 = np.random.choice(examples, n_random_1)
        examples_random_1_5 = np.random.choice(examples, n_random_1_5)

        learner_random_0_5.update_scoring_function(examples_random_0_5)
        learner_random_1.update_scoring_function(examples_random_1)
        learner_random_1_5.update_scoring_function(examples_random_1_5)

        error_random_0_5 = learner_random_0_5.expected_error(examples)
        error_random_1 = learner_random_1.expected_error(examples)
        error_random_1_5 = learner_random_1_5.expected_error(examples)

        accumulator["random_error_0.5"]= [error_random_0_5]
        accumulator["random_n_examples_0.5"] = [n_random_0_5]
        accumulator["random_error_1"]= [error_random_1]
        accumulator["random_n_examples_1"] = [n_random_1]
        accumulator["random_error_1.5"] = [error_random_1_5]
        accumulator["random_n_examples_1.5"] = [n_random_1_5]

        final_dict_accumulator = accumulator_function(accumulator, final_dict_accumulator)

    final_dict_accumulator = calculate_average(final_dict_accumulator, number_of_iterations)
    plot_results.plot_2_a_b_e_f(final_dict_accumulator)

