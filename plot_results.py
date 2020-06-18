import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.serif'] = ['times new roman']
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
import os


################ plot settings #################################
mpl.rc('font',**{'family':'serif','serif':['Times'], 'size': 30})
mpl.rc('legend', **{'fontsize': 22})
mpl.rc('text', usetex=True)
# fig_size = (5.5 / 2.54, 4 / 2.54)
fig_size = [6.5, 4.8]

################# the path where output will be added
file_path_out = 'output/'

def plot_2_a_b_e_f(dict_to_plot, show_plots=False):

    ##### Preprocessing
    if not os.path.isdir(file_path_out):
        os.makedirs(file_path_out)
    small_number = 1

    plt.figure(1, figsize=fig_size)

    err_array_plus = []
    err_array_minus = []
    err_array_approximate_Q = []
    deltas_plus = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    deltas_minus = [0, 0.1, 0.2, 0.3, 0.4]
    delta_approximate_Q = np.array([0, 0.09, 0.18, 0.28, 0.39, 0.51, 0.63, 0.77, 0.92])
    for delta in deltas_plus:
        err_array_plus.append(dict_to_plot["plus_delta_delta={}_error".format(delta)])
    for delta in deltas_minus:
        err_array_minus.append(dict_to_plot["minus_delta_delta={}_error".format(delta)])
    for delta in delta_approximate_Q:
        delta = np.round(delta, 2)
        if delta == 0:
            delta = '0.0'
        err_array_approximate_Q.append(dict_to_plot["approximate_Q_delta={}_error".format(delta)])

    n_iter_array_plus = []
    n_iter_array_minus = []
    n_iter_array_approximate_Q = []

    for delta in deltas_plus:
        n_iter_array_plus.append(dict_to_plot["plus_delta_delta={}_n_examples".format(delta)])
    for delta in deltas_minus:
        n_iter_array_minus.append(int(np.round(dict_to_plot["minus_delta_delta={}_n_examples".format(delta)])))
    for delta in delta_approximate_Q:
        delta = np.round(delta, 2)
        if delta == 0:
            delta = '0.0'
        n_iter_array_approximate_Q.append(int(np.round(dict_to_plot["approximate_Q_delta={}_n_examples".format(delta)])))


    ##################################################################################
    #### 2.a
    ##################################################################################

    plt.figure(1)

    plt.plot(delta_approximate_Q[::2], err_array_approximate_Q[::2], label=r"$\widetilde{\textnormal{OPT}}$",
             color='b', lw=2, marker="^")
    plt.plot(delta_approximate_Q[::2], list(err_array_plus[0]) * len(delta_approximate_Q[::2]), label=r"OPT",
             color='g', lw=2, marker='o')

    plt.plot(delta_approximate_Q[::2], list(dict_to_plot["random_error_0.5"]) * len(delta_approximate_Q[::2]),
             label=r"Rnd:$\frac{1}{2}$",
             color='#8b0000', ls="-.", lw=1.5)
    plt.plot(delta_approximate_Q[::2], list(dict_to_plot["random_error_1"]) * len(delta_approximate_Q[::2]),
             label=r"Rnd:$1$",
             color='#8b0000', ls=":", lw=2)
    plt.plot(delta_approximate_Q[::2], list(dict_to_plot["random_error_1.5"]) * len(delta_approximate_Q[::2]),
             label=r"Rnd:$\frac{3}{2}$",
             color='#8b0000', ls="--", lw=1.5)

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fancybox=False, shadow=False)
    plt.ylabel("Error of the learner")
    plt.xlabel(r'$\delta$')
    plt.ylim(ymax=0.101)
    plt.xticks(delta_approximate_Q[::2])
    plt.savefig(file_path_out + "2.a" + '.pdf', bbox_inches='tight')


    ##################################################################################
    #### 2.b
    ##################################################################################

    plt.figure(2)

    plt.plot(deltas_plus[:-1], err_array_plus[:-1], label=r"$\widetilde{\textnormal{OPT}}$",
             color='b', lw=2, marker="^")
    plt.plot(deltas_plus[:-1], list(err_array_plus[0]) * 5, label=r"OPT",
             color='g', lw=2, marker='o')

    plt.plot(deltas_plus[:-1], list(dict_to_plot["random_error_0.5"]) * 5, label=r"Rnd:$\frac{1}{2}$",
             color='#8b0000', ls="-.", lw=1.5)
    plt.plot(deltas_plus[:-1], list(dict_to_plot["random_error_1"]) * 5, label="Rnd:$1$",
             color='#8b0000', ls=":", lw=2)
    plt.plot(deltas_plus[:-1], list(dict_to_plot["random_error_1.5"]) * 5, label=r"Rnd:$\frac{3}{2}$",
             color='#8b0000', ls="--", lw=1.5)

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fancybox=False, shadow=False)
    plt.ylabel("Error of the learner")
    plt.xlabel(r'$\delta$')
    # plt.yticks(np.arange(0, 1.001, 0.2))
    # plt.xticks(np.arange(0, teaching_step+1, 3))
    # plt.ylim(ymax=1)
    plt.ylim(ymax=0.101)
    plt.xticks(deltas_plus[:-1])
    plt.savefig(file_path_out + "2.b" + '.pdf', bbox_inches='tight')


    ##################################################################################
    #### 2.e
    ##################################################################################

    plt.figure(3)

    delta_approximate_Q = np.round(delta_approximate_Q, 1)
    delta_approximate_Q[-1] -= 0.1

    plt.plot(delta_approximate_Q[::2], n_iter_array_approximate_Q[::2], label=r"$\widetilde{\textnormal{OPT}}$",
             color='b', lw=2, marker="^")
    plt.plot(delta_approximate_Q[::2], [n_iter_array_approximate_Q[0]] * (len(delta_approximate_Q[::2])), label=r"OPT",
             color='g', lw=2, marker="o")
    plt.plot(delta_approximate_Q[::2], [np.round(n_iter_array_plus[0]) * 0.5] * (len(delta_approximate_Q[::2])),
             label=r"Rnd:$\frac{1}{2}$",
             color='#8b0000', ls="-.", lw=1.5)
    plt.plot(delta_approximate_Q[::2],
             [np.round(n_iter_array_plus[0]) * 1 + small_number] * (len(delta_approximate_Q[::2])), label=r"Rnd:$1$",
             color='#8b0000', ls=":", lw=2)
    plt.plot(delta_approximate_Q[::2], [np.round(n_iter_array_plus[0]) * 1.5] * (len(delta_approximate_Q[::2])),
             label=r"Rnd:$\frac{3}{2}$",
             color='#8b0000', ls="--", lw=1.5)

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fancybox=False, shadow=False)
    plt.ylabel('Teaching set size')
    plt.xlabel(r'$\delta$')
    # plt.ylim(ymin=0)
    plt.ylim(ymax=100)
    plt.ylim(ymin=0)
    plt.xticks(delta_approximate_Q[::2])
    plt.savefig(file_path_out + "2.e" + '.pdf', bbox_inches='tight')


    ##################################################################################
    #### 2.f
    ##################################################################################

    plt.figure(4)
    plt.plot(deltas_minus, n_iter_array_minus, label=r"$\widetilde{\textnormal{OPT}}$",
             color='b', lw=2, marker="^")

    plt.plot(deltas_minus, [n_iter_array_minus[0]] * 5, label=r"OPT",
             color='g', lw=2, marker="o")

    plt.plot(deltas_minus, [np.round(n_iter_array_minus[0]) * 0.5] * 5, label=r"Rnd:$\frac{1}{2}$",
             color='#8b0000', ls="-.", lw=1.5)
    plt.plot(deltas_minus, [np.round(n_iter_array_minus[0]) * 1 + small_number] * 5, label=r"Rnd:$1$",
             color='#8b0000', ls=":", lw=2)
    plt.plot(deltas_minus, [np.round(n_iter_array_minus[0]) * 1.5] * 5, label=r"Rnd:$\frac{3}{2}$",
             color='#8b0000', ls="--", lw=1.5)

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fancybox=False, shadow=False)
    plt.ylabel('Teaching set size')
    plt.xlabel(r'$\delta$')
    plt.ylim(ymax=100)
    plt.ylim(ymin=0)
    plt.xticks(deltas_minus)
    plt.savefig(file_path_out + "2.f" + '.pdf', bbox_inches='tight')
    if show_plots:
        plt.show()
#enddef

def plot_2_c_d_g_h(dict_to_plot, show_plots=False):

    ##### Preprocessing
    if not os.path.isdir(file_path_out):
        os.makedirs(file_path_out)

    err_array_greedy_limited_ground_truth = []
    portion_of_knowledge_array = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    for portion in portion_of_knowledge_array:
        if portion == 0:
            portion = str('0')
        err_array_greedy_limited_ground_truth.append(dict_to_plot["limited_ground_truth_delta={}_error".format(portion)])

    n_iter_array_greedy_limited_ground_truth = []

    for portion in portion_of_knowledge_array:
        if portion == 0:
            portion = str('0')
        n_iter_array_greedy_limited_ground_truth.append(dict_to_plot["limited_ground_truth_delta={}_n_examples".format(portion)])

    x_axis_array = portion_of_knowledge_array

    random_0_5 = [dict_to_plot["random_error_0.5"]]
    random_1 = [dict_to_plot["random_error_1"]]
    random_1_5 = [dict_to_plot["random_error_1.5"]]

    ##################################################################################
    #### 2.c
    ##################################################################################

    plt.figure(1)
    plt.plot(x_axis_array, err_array_greedy_limited_ground_truth, label=r"$\widetilde{\textnormal{OPT}}$",
             color='b', lw=2, marker="^")
    plt.plot(x_axis_array, [err_array_greedy_limited_ground_truth[0]] * len(x_axis_array), label=r"OPT",
             color='g', ls=":", lw=2, marker="o")

    plt.plot(x_axis_array, random_0_5 * len(x_axis_array), label=r"Rnd:$\frac{1}{2}$", color='#8b0000', ls="-.", lw=1.5)
    plt.plot(x_axis_array, random_1 * len(x_axis_array), label="Rnd:$1$",
             color='#8b0000', ls=":", lw=2)
    plt.plot(x_axis_array, random_1_5 * len(x_axis_array), label=r"Rnd:$\frac{3}{2}$",
             color='#8b0000', ls="--", lw=1.5)

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fancybox=False, shadow=False)
    plt.ylabel("Error of the learner")
    plt.xlabel(r'Fraction of $\mathcal{|X|}$')
    # plt.xticks(list(np.arange(0.1, 1.001, 0.1)).reverse())
    # plt.xticks(np.arange(0, teaching_step+1, 3))
    plt.xticks(x_axis_array, ["1", '0.9', '0.8', '0.7', '0.6', '0.5'])
    plt.ylim(ymax=0.101)
    # plt.ylim(xmin=0.1)

    # plt.xticks(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    # plt.gca().invert_xaxis()
    plt.savefig(file_path_out + "2.c" + '.pdf', bbox_inches='tight')


    ##################################################################################
    #### 2.g
    ##################################################################################

    plt.figure(2)
    small_number = 1

    plt.plot(x_axis_array, n_iter_array_greedy_limited_ground_truth, label=r"$\widetilde{\textnormal{OPT}}$",
             color='b', lw=2, marker="^")
    plt.plot(x_axis_array, [n_iter_array_greedy_limited_ground_truth[0]] * len(x_axis_array), label=r"OPT",
             color='g', lw=2, marker="o")

    plt.plot(x_axis_array, [np.round(n_iter_array_greedy_limited_ground_truth[0]) * 0.5] * len(x_axis_array), label=r"Rnd:$\frac{1}{2}$",
             color='#8b0000', ls="-.", lw=1.5)
    plt.plot(x_axis_array, [np.round(n_iter_array_greedy_limited_ground_truth[0]) * 1 + small_number] * len(x_axis_array), label=r"Rnd:$1$",
             color='#8b0000', ls=":", lw=2)
    plt.plot(x_axis_array, [np.round(n_iter_array_greedy_limited_ground_truth[0]) * 1.5] * len(x_axis_array), label=r"Rnd:$\frac{3}{2}$",
             color='#8b0000', ls="--", lw=1.5)

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fancybox=False, shadow=False)
    plt.ylabel('Teaching set size')
    plt.xlabel(r'Fraction of $\mathcal{|X|}$')
    plt.ylim(ymax=100)
    plt.ylim(ymin=0)
    plt.xticks(x_axis_array, ["1", '0.9', '0.8', '0.7', '0.6', '0.5'])
    plt.savefig(file_path_out + "2.g" + '.pdf', bbox_inches='tight')


    ### Preprocessing

    err_array_greedy_noise_feature = []
    noise_array = np.arange(0, 0.201, 0.02)
    for delta in noise_array:
        delta = np.round(delta, 2)
        if delta == 0:
            delta = str('0.0')
        err_array_greedy_noise_feature.append(dict_to_plot["noise_feature_delta={}_error".format(delta)])


    n_iter_array_greedy_noise_feature = []

    for delta in noise_array:
        delta = np.round(delta, 2)
        if delta == 0:
            delta = str('0.0')
        n_iter_array_greedy_noise_feature.append(dict_to_plot["noise_feature_delta={}_n_examples".format(delta)])


    ##################################################################################
    #### 2.d
    ##################################################################################
    plt.figure(3)
    plt.plot(noise_array[::2], err_array_greedy_noise_feature[::2], label=r"$\widetilde{\textnormal{OPT}}$",
             color='b', lw=2, marker="^")
    plt.plot(noise_array[::2], [err_array_greedy_noise_feature[0]] * len(noise_array[::2]), label=r"OPT",
             color='g', lw=2, marker="o")

    plt.plot(noise_array[::2], random_0_5 * len(noise_array[::2]), label=r"Rnd:$\frac{1}{2}$", color='#8b0000', ls="-.",
             lw=1.5)
    plt.plot(noise_array[::2], random_1 * len(noise_array[::2]), label="Rnd:$1$",
             color='#8b0000', ls=":", lw=2)
    plt.plot(noise_array[::2], random_1_5 * len(noise_array[::2]), label=r"Rnd:$\frac{3}{2}$", color='#8b0000', ls="--",
             lw=1.5)

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fancybox=False, shadow=False)
    plt.ylabel("Error of the learner")
    plt.xlabel(r'$\delta$ ($\%$ shift w.r.t. input radius)')
    # plt.yticks(np.arange(0, 1.001, 0.2))
    # plt.xticks(np.arange(0, teaching_step+1, 3))
    plt.ylim(ymax=0.101)
    # plt.ylim(ymin=0)
    plt.xticks(noise_array[::2], ['0', '2', '4', '6', '8', '10'])
    plt.savefig(file_path_out + "2.d" + '.pdf', bbox_inches='tight')


    ##################################################################################
    #### 2.h
    ##################################################################################

    plt.figure(4)
    small_number = 1

    plt.plot(noise_array[::2], n_iter_array_greedy_noise_feature[::2], label=r"$\widetilde{\textnormal{OPT}}$",
             color='b', lw=2, marker="^")
    plt.plot(noise_array[::2], [n_iter_array_greedy_noise_feature[0]] * len(noise_array[::2]), label=r"OPT",
             color='g', lw=2, marker="o")
    plt.plot(noise_array[::2], [np.round(n_iter_array_greedy_noise_feature[0]) * 0.5] * len(noise_array[::2]),
             label=r"Rnd:$\frac{1}{2}$",
             color='#8b0000', ls="-.", lw=1.5)
    plt.plot(noise_array[::2], [np.round(n_iter_array_greedy_noise_feature[0]) * 1 + small_number] * len(noise_array[::2]),
             label=r"Rnd:$1$",
             color='#8b0000', ls=":", lw=2)
    plt.plot(noise_array[::2], [np.round(n_iter_array_greedy_noise_feature[0]) * 1.5] * len(noise_array[::2]),
             label=r"Rnd:$\frac{3}{2}$",
             color='#8b0000', ls="--", lw=1.5)

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fancybox=False, shadow=False)
    plt.ylabel('Teaching set size')
    plt.xlabel(r'$\delta$ ($\%$ shift w.r.t. input radius)')
    plt.ylim(ymin=0)
    plt.ylim(ymax=100)
    plt.xticks(noise_array[::2], ['0', '2', '4', '6', '8', '10'])
    plt.savefig(file_path_out + "2.h" + '.pdf', bbox_inches='tight')

    if show_plots:
        plt.show()
#enddef


if __name__ == "__main__":
    pass



