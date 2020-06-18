import numpy as np
import os
import matplotlib.pyplot as plt
import example
import matplotlib as mpl
mpl.rcParams['font.serif'] = ['times new roman']
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']

mpl.rc('font',**{'family':'serif','serif':['Times'], 'size': 24})
mpl.rc('font',**{'family':'serif','serif':['Times'], 'size': 24})
mpl.rc('legend', **{'fontsize': 24})
mpl.rc('text', usetex=True)


def plot_fig_a( examples, picked_examples_by_zero_out_teacher, colors, hypotheses, prefix, title):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.scatter([f[0] for f in [e.x_s for e in examples[4:]]],
        [f[1] for f in [e.x_s for e in examples[4:]] ], color='#F08080', marker='^', label=r"$-$", s=40)

    plt.scatter([f[0] for f in [e.x_s for e in examples[0:4]]],
        [f[1] for f in [e.x_s for e in examples[0:4]] ], color='#1F45FC', marker='o', label=r"$+$",
                s=40)

    plt.scatter([f[0] for f in [e.x_s for e in picked_examples_by_zero_out_teacher]],
                [f[1] for f in [e.x_s for e in picked_examples_by_zero_out_teacher]],
                color='r', marker='*')


    x_axis = [-3.0, 3.0]
    y_axis = [-3.0, 3.0]
    plt.xlim(x_axis)
    plt.ylim(y_axis)
    ax.set_xticks((x_axis[0], x_axis[1]))
    ax.set_yticks((y_axis[0], y_axis[1]))

    x_vals = np.array(ax.get_xlim())
    for i in range(len(hypotheses)):
        w0 = hypotheses[i]
        if i ==0:
            linewidth = 3
            label = r"$h^*$"
        else:
            linewidth = 1.5
            if i ==1:
                label = "$h$"
            else:
                label= ""
        y_vals = (w0[2]/w0[1]) + (-w0[0]/w0[1]) * x_vals
        plt.plot(x_vals, y_vals, color=colors[i], linewidth=linewidth, label=label)
    plt.arrow(2, 1.5 , 0, np.sqrt(0.1), head_width=0.1, width=0.01, head_length=0.1, color='orange')
    plt.arrow(1.5, 2.5, -np.sqrt(0.1/2), np.sqrt(0.1/2), head_width=0.1, width=0.01, head_length=0.1, color='orange')
    plt.arrow(-1.5, 2.5, np.sqrt(0.1/2), np.sqrt(0.1/2), head_width=0.1, width=0.01, head_length=0.1, color='orange')
    plt.arrow(2, -1.5, 0, np.sqrt(0.1), head_width=0.1, width=0.01, head_length=0.1, color='orange')
    plt.arrow(-1.55, -2.3, -np.sqrt(0.1/2), np.sqrt(0.1/2), head_width=0.1, width=0.01, head_length=0.1, color='orange')
    plt.arrow(1.75, -2.5, np.sqrt(0.1/2), np.sqrt(0.1/2), head_width=0.1, width=0.01, head_length=0.1, color='orange')
        # plt.arrow(x_vals, y_vals, x_vals, y_vals)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.31), ncol=2, fancybox=False, shadow=False, borderpad=0.1)


    # alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
    plt.title(title)

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    plt.savefig(prefix + title + "embedding-bad1_new.pdf", bbox_inches='tight')
    # plt.show()
    # plt.close()
#enddef

def plot_fig_b( examples, picked_examples_by_zero_out_teacher, colors, hypotheses, prefix, title):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.scatter([f[0] for f in [e.x_s for e in examples[0:14]]],
        [f[1] for f in [e.x_s for e in examples[0:14]] ], color='#F08080', marker='^', label=r"$-$", s=40)

    plt.scatter([f[0] for f in [e.x_s for e in examples[14:]]],
        [f[1] for f in [e.x_s for e in examples[14:]] ], color='#1F45FC', marker='o', label=r"$+$",
                s=40)

    plt.scatter([f[0] for f in [e.x_s for e in picked_examples_by_zero_out_teacher]],
                [f[1] for f in [e.x_s for e in picked_examples_by_zero_out_teacher]],
                color='r', marker='*')


    x_axis = [-3.0, 3.0]
    y_axis = [-3.0, 3.0]
    plt.xlim(x_axis)
    plt.ylim(y_axis)
    ax.set_xticks((x_axis[0], x_axis[1]))
    ax.set_yticks((y_axis[0], y_axis[1]))

    x_vals = np.array(ax.get_xlim())
    for i in range(len(hypotheses)):
        w0 = hypotheses[i]
        if i ==0:
            linewidth = 3
            label = r"$h^*$"
        else:
            linewidth = 1.5
            if i ==1:
                label = "$h$"
            else:
                label= ""
        y_vals = (w0[2]/w0[1]) + (-w0[0]/w0[1]) * x_vals
        plt.plot(x_vals, y_vals, color=colors[i], linewidth=linewidth, label=label)
        # plt.arrow(x_vals, y_vals, x_vals, y_vals)
        #plt.arrow(w0[2], 2, np.sqrt(0.1), 0., head_width=0.1, width=0.01, head_length=0.1, color='orange')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.31), ncol=2, fancybox=False, shadow=False, borderpad=0.1)

    alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
    # plt.title(title)

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    plt.savefig(prefix + title + "embedding-bad2_new.pdf", bbox_inches='tight')
    plt.show()
    plt.close()
#enddef


if __name__ == "__main__":
    pass