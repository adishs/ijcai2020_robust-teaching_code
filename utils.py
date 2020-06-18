import numpy as np
import math
import copy
import random
import example
import os


def remove_all_inconsistent_examples(h_star, examples):
    copy_examples = copy.deepcopy(examples)
    for e in copy_examples:
        if not prediction_is_correct(h_star, e):
            print("Inconsistent example id=", e.id)
            copy_examples.remove(e)
    return copy_examples
#enddef

def remove_more_than_one_h_star(H, examples):
    h_array = [H[0]]
    for h in H[1:]:
        count = 0
        for e in examples:
            if not prediction_is_correct(h, e):
                count +=1
        if count != 0:
            h_array.append(h)
    return h_array
#enddef

def get_hypotheses(hypotheses_file):
    hypotheses_fp = open(hypotheses_file, 'r')
    initial_hypotheses = []
    for line in hypotheses_fp:
        line = line.split()
        initial_hypothesis = [float(i) for i in line]
        initial_hypotheses.append(initial_hypothesis)
    return initial_hypotheses
#enddef

def input_examples(dataset):
    examples = []
    i = 0
    for x_s, y_s in zip(dataset.features, dataset.labels):
        exm = example.example(id=i, x_s=x_s, y_s=y_s)
        examples.append(exm)
        i += 1
    return examples
#enddef

def prediction_is_correct(h, example):
    if (np.dot(h[:-1], example.x_s) - h[-1]) * example.y_s >= 0:
        return True
    else:
        return False
#enndef

def error_for_every_h(H, examples):
    error_array = []
    for h in H:
        error_count = 0
        for example in examples:
            if not prediction_is_correct(h, example):
                error_count += 1
        error_array.append(error_count / len(examples))
    return error_array
#enddef

def find_h_star_index_given_examples(H, examples):
    error_array = np.array(error_for_every_h(H, examples))
    indexes = np.where(error_array == error_array.min())[0]
    # print(indexes)
    index = np.random.choice(indexes, size=1)[0]
    return index
#enddef

def shift_data_points_given_radius(r, examples):
    perturbed_examples = []
    for e in examples:
        angle = random.uniform(0, 2*math.pi)
        new_coords = [r * math.cos(angle) + e.x_s[0], r * math.sin(angle) + e.x_s[1]]
        perturbed_e = copy.deepcopy(e)
        perturbed_e.x_s = new_coords
        perturbed_examples.append(perturbed_e)
    return perturbed_examples
#enddef

def get_teachers_examples(teaching_type, delta, examples):
    teachers_examples = None
    if teaching_type == "noise_feature":
        teachers_examples = shift_data_points_given_radius(delta, examples)
    elif teaching_type == "limited_ground_truth":
        known_set_size = int(np.round((len(examples) * (1-delta))))
        teachers_examples = np.random.choice(examples, size=known_set_size, replace=False)
    else:
        print("Wrong teaching type!!", teaching_type)
        exit(0)
    return teachers_examples
#enddef

def get_n_best_hypoteses(H, examples, top_k=10):
    best_hypotheses_index = []
    hyp_error_dict = {}
    error_array = error_for_every_h(H, examples)
    for i, error in enumerate(error_array):
        hyp_error_dict[i] = error
    sorted_dict = {k: v for k, v in sorted(hyp_error_dict.items(), key=lambda item: item[1])}

    for i, key in enumerate(sorted_dict.keys()):
        if i > top_k:
            break
        else:
            best_hypotheses_index.append(key)
    return best_hypotheses_index
#enddef


def write_into_file(accumulator, exp_iter, teacher_type):
    directory = 'results/{}'.format(teacher_type)
    filename = "convergence" + '_' + str(exp_iter) + '.txt'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filepath = directory + '/' + filename
    print("output file name  ", filepath)
    f = open(filepath, 'w')
    for key in accumulator:
        f.write(key + '\t')
        temp = list(map(str, accumulator[key]))
        for j in temp:
            f.write(j + '\t')
        f.write('\n')
    f.close()
#enddef

def get_delta_initD_given_mu(mu, H, examples, top_k=10):
    Q_init = np.ones(len(H))/len(H)
    Q_init_perturbed = np.ones(len(H))
    best_hypotheses_index = get_n_best_hypoteses(H, examples, top_k)
    Q_init_perturbed[best_hypotheses_index] *= (1 + mu)
    all_indexes = np.arange(0, len(H))
    index_of_not_best_hypothesis = np.setdiff1d(all_indexes, best_hypotheses_index)
    Q_init_perturbed[index_of_not_best_hypothesis] *= (1 - mu)
    Q_init_perturbed /= sum(Q_init_perturbed)
    min_delta = None
    for delta in np.arange(0.0, 1, 0.01):
        if ((1-delta) * Q_init <= Q_init_perturbed).all() and ((1+delta) * Q_init >= Q_init_perturbed).all():
            min_delta = delta
            break
    return min_delta, Q_init_perturbed
#enddef


if __name__ == "__main__":
    pass






