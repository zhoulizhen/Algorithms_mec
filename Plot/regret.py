
import matplotlib.pyplot as plt
import numpy as np


def regret_plot(ol,mab,adms):

    beta = 0.9
    num_samples = 20
    #
    # # step 1 generate random seed
    # np.random.seed(0)
    # raw_tmp = np.random.randint(32, 38, size=num_samples)
    x_index = np.arange(num_samples)
    # # raw_tmp = [35, 34, 37, 36, 35, 38, 37, 37, 39, 38, 37]  # temperature
    # print(raw_tmp)

    # step 2 calculate ema result and do not use correction
    raw_tmp = ol
    raw_tmp1 = mab
    raw_tmp2 = adms

    v_ema = []
    v_pre = 0
    for i, t in enumerate(raw_tmp):
        v_t = beta * v_pre + (1 - beta) * int(t)
        v_ema.append(v_t)
        v_pre = v_t
    print("v_mea:", v_ema)

    v_ema1 = []
    v_pre1 = 0
    for i, t in enumerate(raw_tmp1):
        v_t = beta * v_pre1 + (1 - beta) * int(t)
        v_ema1.append(v_t)
        v_pre1 = v_t
    print("v_mea:", v_ema1)

    v_ema2 = []
    v_pre2 = 0
    for i, t in enumerate(raw_tmp2):
        v_t = beta * v_pre2 + (1 - beta) * int(t)
        v_ema2.append(v_t)
        v_pre2 = v_t
    print("v_mea:", v_ema2)

    # step 3 correct the ema results
    v_ema_corr = []
    for i, t in enumerate(v_ema):
        v_ema_corr.append(t / (1 - np.power(beta, i + 1)))
    print("v_ema_corr", v_ema_corr)

    v_ema_corr1 = []
    for i, t in enumerate(v_ema1):
        v_ema_corr1.append(t / (1 - np.power(beta, i + 1)))
    print("v_ema_corr", v_ema_corr1)

    v_ema_corr2 = []
    for i, t in enumerate(v_ema2):
        v_ema_corr2.append(t / (1 - np.power(beta, i + 1)))
    print("v_ema_corr", v_ema_corr2)

    # step 4 plot ema and correction ema reslut
    # plt.plot(x_index, raw_tmp, label='raw_tmp')  # Plot some data on the (implicit) axes.
    # plt.plot(x_index, v_ema, label='v_ema')  # etc.
    x = []
    for i in range(0,400,1):
        x.append(str(i))


    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    COLOR1 = (151 / 255, 179 / 255, 25 / 255)
    COLOR2 = (229 / 255, 139 / 255, 123 / 255)
    COLOR3 = (246 / 255, 224 / 255, 147 / 255)
    COLOR4 = (120 / 255, 183 / 255, 201 / 255)
    COLOR5 = (70 / 255, 120 / 255, 142 / 255)

    A, = plt.plot(x, v_ema_corr, markersize=30, c=COLOR5, lw=6, ls='-', label='OL')
    B, = plt.plot(x, v_ema_corr1, markersize=30, c=COLOR2, lw=6, ls='--', label='MAB')
    C, = plt.plot(x, v_ema_corr2, markersize=30, c=COLOR4, lw=6, ls='-.', label='ADMS')

    font1 = {'family': 'Helvetica',
             'weight': 'normal',
             'size': 35,
             }

    legend = plt.legend(handles=[A,B,C], prop=font1, ncol=1, labelspacing=0.02, handlelength=1, handleheight=1,
                        handletextpad=0.5, columnspacing=0.5, borderaxespad=0.1, borderpad=0.2)
    legend._legend_box.align = "right"

    ax.set_axisbelow(True)

    # Add grid lines
    ax.grid(axis="y", color="#A8BAC4", lw=1.2)

    # Customize bottom spine
    ax.spines["bottom"].set_lw(1.2)
    ax.spines["bottom"].set_capstyle("butt")

    legend = plt.legend(handles=[A, B, C], prop=font1, ncol=1, labelspacing=0.02, handlelength=0.5,
                        handleheight=1, handletextpad=0.5, columnspacing=0.5, borderaxespad=0.1, borderpad=0.2)
    # legend._legend_box.align = "right"
    # ax.legend(labels=['Concurrent pulls = 10','Concurrent pulls = 30','Concurrent pulls = 50'])

    plt.tick_params(labelsize=35)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Helvetica') for label in labels]

    # plt.xlim((0, 28)) #x轴范围
    # plt.ylim((0, 1.4)) #y轴范围
    plt.xticks(np.arange(0, 450,100)) #
    # plt.yticks(np.arange(0, 30, 5))
    # plt.xscale('symlog')

    plt.xlabel('Number of requests', font1)
    plt.ylabel('Regret', font1)
    # plt.grid(b=True, ls=':') #生成网格
    # figure.subplots_adjust(left=0.5, right=0.3, bottom=0.2, top=0.82)
    fig.tight_layout()
    plt.savefig('../regret.eps', format='eps')
    plt.show()

if __name__ == '__main__':
    regret_plot()