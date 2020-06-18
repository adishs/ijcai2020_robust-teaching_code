import learner
import teacher
import dataset
import utils
import numpy as np
import copy
import plot_results
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
    #get all examples
    examples = utils.input_examples(ds)
    # initial distribution
    Q_0 = np.ones(len(hypotheses)) / len(hypotheses)


    for iteration in range(0, number_of_iterations):

        #Define deltas
        deltas_for_noise_feature = np.arange(0, 0.201, 0.02)
        teacher_type = "noise_feature"#"noise_feature"  #limited_ground_truth

        accumulator = {}
        for delta in deltas_for_noise_feature:
            delta = np.round(delta, 2)
            teachers_examples = utils.get_teachers_examples(teacher_type, delta, examples)
            #Compute error of training examples
            #creating teaching class object
            teaching_object = teaching(Q_0=Q_0, hypotheses=hypotheses,
                                      examples=teachers_examples, eta=0.5, delta=delta, eps=0.001,
                                      teacher_type=teacher_type, exp_iter=exp_iter, mu=[], top_k=10)
            print("=====Teaching start ---{}---====delta={}====".format(str.upper(teacher_type), delta))
            teaching_examples = teaching_object.teacher.get_selected_examples_for_demonstration()

            if delta == deltas_for_noise_feature[0]:
                n_random = len(teaching_examples)
            learner_local = copy.deepcopy(teaching_object.learner)
            learner_local.update_scoring_function(teaching_examples)
            print(learner_local.expected_error(examples))
            print(delta)
            # input("===")
            accumulator["{}_delta={}_error".format(teacher_type, str(delta))] = [
                learner_local.expected_error(examples)]
            accumulator["{}_delta={}_n_examples".format(teacher_type, str(delta))] = [
                len(teaching_examples)]


        #Define deltas
        deltas_for_limited_ground_truth = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        teacher_type = "limited_ground_truth"

        for delta in deltas_for_limited_ground_truth:
            teachers_examples = utils.get_teachers_examples(teacher_type, delta, examples)
            #Compute error of training examples
            #creating teaching class object
            teaching_object = teaching(Q_0=Q_0, hypotheses=hypotheses,
                                      examples=teachers_examples, eta=0.5, delta=delta, eps=0.001,
                                      teacher_type=teacher_type, exp_iter=exp_iter, mu=[], top_k=10)
            print("=====Teaching start ---{}---====delta={}====".format(str.upper(teacher_type), delta))
            teaching_examples = teaching_object.teacher.get_selected_examples_for_demonstration()

            learner_local = copy.deepcopy(teaching_object.learner)
            learner_local.update_scoring_function(teaching_examples)
            print(learner_local.expected_error(examples))
            print(delta)
            # input("===")
            accumulator["{}_delta={}_error".format(teacher_type, str(delta))] = [
                learner_local.expected_error(examples)]
            accumulator["{}_delta={}_n_examples".format(teacher_type, str(delta))] = [
                len(teaching_examples)]

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

        accumulator["random_error_0.5"] = [error_random_0_5]
        accumulator["random_n_examples_0.5"] = [n_random_0_5]
        accumulator["random_error_1"] = [error_random_1]
        accumulator["random_n_examples_1"] = [n_random_1]
        accumulator["random_error_1.5"] = [error_random_1_5]
        accumulator["random_n_examples_1.5"] = [n_random_1_5]

        final_dict_accumulator = accumulator_function(accumulator, final_dict_accumulator)

    final_dict_accumulator = calculate_average(final_dict_accumulator, number_of_iterations)
    plot_results.plot_2_c_d_g_h(final_dict_accumulator)