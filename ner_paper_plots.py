import numpy as np
import matplotlib.pyplot as plt


def plot_mean_std(x_mean, x_std=None, x_n=None, c="r", label=None, ms=1.8,):
    if x_n is None:
        x_n = list(range(len(x_mean)))
    plt.plot(x_n, x_mean, color=c, marker="d", linewidth=1.8, markersize=ms, label=label)
    if x_std is not None:
        plt.plot(x_n, x_mean+x_std, color=c, linewidth=0.2, alpha=0.2)
        plt.plot(x_n, x_mean-x_std, color=c, linewidth=0.2, alpha=0.2)
        plt.fill_between(x_n, x_mean+x_std, x_mean-x_std, color=c, alpha=0.03)


def plot_results(x_local, x_local_std, x_g_mean, x_g_std, x_g_st_mean, x_g_st_std, x_d_mean, x_d_std, x_d_st_mean, x_d_st_std, title="", ylim=None, lloc=0):
    labels = ["[:n]", "0.001", "0.005", "0.01", "0.05", "0.1", "0.15", "0.2", "0.25", "0.34", "0.5"]
    x_n = list(range(len(labels)))
    # doc only have some alphas
    x_d = [0] + x_n[-6:]
    plt.figure(figsize=(8, 4))
    plot_mean_std(x_d_mean, x_d_std, x_d, c="#007C1C", label="doc")
    plot_mean_std(x_d_st_mean, x_d_st_std, x_d, c="#6CD700", label="doc stacked")
    # global only and stack have full range of alphas
    plot_mean_std(x_g_mean, x_g_std, x_n, c="#012071", label="global")
    plot_mean_std(x_g_st_mean, x_g_st_std, x_n, c="#0088CC", label="global stacked")
    # plot local as one line
    plot_mean_std(np.array(len(x_n)*[x_local]), np.array(len(x_n)*[x_local_std]), x_n, c="#71011E", label="local only", ms=0.)

    plt.xticks(x_n, labels, fontsize=15.5)
    plt.xlabel(r"$\alpha$", fontsize=25)
    plt.ylabel("F1-score (micro)", fontsize=15.5)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(loc=lloc, fontsize=14, ncol=3)
    plt.title(title)
    plt.savefig("%s.pdf" % title.lower().replace(" ", "_"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':

    # ## BERT
    x_local = 100*0.9126
    x_local_std = 100*0.0015
    x_g_mean    = 100*np.array([0.9128, 0.9125, 0.9121, 0.9135, 0.9134, 0.9150, 0.9158, 0.9167, 0.9182, 0.9200, 0.9189])
    x_g_std     = 100*np.array([0.0016, 0.0013, 0.0012, 0.0009, 0.0008, 0.0023, 0.0019, 0.0017, 0.0017, 0.0010, 0.0013])
    x_g_st_mean = 100*np.array([0.9206, 0.9198, 0.9206, 0.9196, 0.9193, 0.9203, 0.9190, 0.9193, 0.9188, 0.9170, 0.9182])
    x_g_st_std  = 100*np.array([0.0012, 0.0024, 0.0015, 0.0011, 0.0010, 0.0008, 0.0005, 0.0019, 0.0009, 0.0013, 0.0012])
    x_d_mean    = 100*np.array([0.9155, 0.9156, 0.9155, 0.9160, 0.9168, 0.9166, 0.9151])
    x_d_std     = 100*np.array([0.0031, 0.0018, 0.0021, 0.0013, 0.0020, 0.0008, 0.0008])
    x_d_st_mean = 100*np.array([0.9157, 0.9148, 0.9166, 0.9161, 0.9164, 0.9164, 0.9156])
    x_d_st_std  = 100*np.array([0.0010, 0.0008, 0.0005, 0.0010, 0.0010, 0.0005, 0.0012])
    plot_results(x_local, x_local_std, x_g_mean, x_g_std, x_g_st_mean, x_g_st_std, x_d_mean, x_d_std, x_d_st_mean, x_d_st_std, title="BERT", ylim=(90.5, 93.5), lloc=9)

    # ## ROBERTA
    x_local = 100*0.9183
    x_local_std = 100*0.0005
    x_g_mean    = 100*np.array([0.9145, 0.9122, 0.9141, 0.9143, 0.9147, 0.9166, 0.9188, 0.9205, 0.9202, 0.9228, 0.9234])
    x_g_std     = 100*np.array([0.0021, 0.0017, 0.0006, 0.0009, 0.0017, 0.0014, 0.0014, 0.0012, 0.0006, 0.0014, 0.0012])
    x_g_st_mean = 100*np.array([0.9183, 0.9206, 0.9205, 0.9212, 0.9206, 0.9223, 0.9206, 0.9218, 0.9205, 0.9203, 0.9197])
    x_g_st_std  = 100*np.array([0.0011, 0.0011, 0.0006, 0.0009, 0.0007, 0.0009, 0.0014, 0.0007, 0.0004, 0.0013, 0.0008])
    x_d_mean    = 100*np.array([0.9222, 0.9199, 0.9204, 0.9219, 0.9214, 0.9218, 0.9223])
    x_d_std     = 100*np.array([0.0006, 0.0008, 0.0011, 0.0017, 0.0005, 0.0011, 0.0010])
    x_d_st_mean = 100*np.array([0.9197, 0.9198, 0.9202, 0.9205, 0.9204, 0.9191, 0.9195])
    x_d_st_std  = 100*np.array([0.0013, 0.0009, 0.0013, 0.0008, 0.0007, 0.0014, 0.0009])
    plot_results(x_local, x_local_std, x_g_mean, x_g_std, x_g_st_mean, x_g_st_std, x_d_mean, x_d_std, x_d_st_mean, x_d_st_std, title="RoBERTa", ylim=(90, 93), lloc=8)

    # ## Elmo
    x_local = 100*0.9087
    x_local_std = 100*0.0008
    x_g_mean    = 100*np.array([0.9101, 0.9106, 0.9099, 0.9105, 0.9116, 0.9125, 0.9120, 0.9123, 0.9130, 0.9132, 0.9131])
    x_g_std     = 100*np.array([0.0010, 0.0012, 0.0022, 0.0027, 0.0023, 0.0003, 0.0013, 0.0004, 0.0011, 0.0021, 0.0005])
    x_g_st_mean = 100*np.array([0.9149, 0.9160, 0.9146, 0.9172, 0.9155, 0.9165, 0.9167, 0.9152, 0.9156, 0.9139, 0.9139])
    x_g_st_std  = 100*np.array([0.0009, 0.0007, 0.0013, 0.0001, 0.0003, 0.0010, 0.0013, 0.0018, 0.0005, 0.0015, 0.0007])
    x_d_mean    = 100*np.array([0.9074, 0.9081, 0.9092, 0.9082, 0.9091, 0.9099, 0.9079])
    x_d_std     = 100*np.array([0.0008, 0.0006, 0.0004, 0.0005, 0.0009, 0.0007, 0.0021])
    x_d_st_mean = 100*np.array([0.9152, 0.9118, 0.9127, 0.9125, 0.9132, 0.9138, 0.9123])
    x_d_st_std  = 100*np.array([0.0008, 0.0013, 0.0019, 0.0011, 0.0007, 0.0005, 0.0007])
    plot_results(x_local, x_local_std, x_g_mean, x_g_std, x_g_st_mean, x_g_st_std, x_d_mean, x_d_std, x_d_st_mean, x_d_st_std, title="ELMo", ylim=(90, 93), lloc=9)

    # ## Flair
    x_local = 100*0.8912
    x_local_std = 100*0.0012
    x_g_mean    = 100*np.array([0.9027, 0.9035, 0.9025, 0.9030, 0.9030, 0.9038, 0.9044, 0.9060, 0.9048, 0.9041, 0.9002])
    x_g_std     = 100*np.array([0.0007, 0.0010, 0.0021, 0.0015, 0.0010, 0.0006, 0.0014, 0.0014, 0.0005, 0.0001, 0.0008])
    x_g_st_mean = 100*np.array([0.9068, 0.9066, 0.9058, 0.9059, 0.9069, 0.9064, 0.9069, 0.9068, 0.9066, 0.9059, 0.9033])
    x_g_st_std  = 100*np.array([0.0009, 0.0014, 0.0011, 0.0007, 0.0000, 0.0008, 0.0001, 0.0005, 0.0005, 0.0007, 0.0003])
    x_d_mean    = 100*np.array([0.8987, 0.8980, 0.8980, 0.8986, 0.8988, 0.8967, 0.8951])
    x_d_std     = 100*np.array([0.0011, 0.0007, 0.0006, 0.0009, 0.0009, 0.0021, 0.0022])
    x_d_st_mean = 100*np.array([0.9016, 0.8999, 0.9014, 0.9007, 0.8997, 0.8986, 0.8987])
    x_d_st_std  = 100*np.array([0.0020, 0.0036, 0.0009, 0.0022, 0.0015, 0.0026, 0.0036])
    plot_results(x_local, x_local_std, x_g_mean, x_g_std, x_g_st_mean, x_g_st_std, x_d_mean, x_d_std, x_d_st_mean, x_d_st_std, title="Flair", ylim=(89, 92), lloc=9)

    # ## BERT - CONLL_GERMAN
    x_local = 100*0.8746
    x_local_std = 100*0.0005
    x_g_mean    = 100*np.array([0.8718, 0.8709, 0.8731, 0.8711, 0.8737, 0.8711, 0.8767, 0.8798, 0.8784, 0.8793, 0.8797])
    x_g_std     = 100*np.array([0.0016, 0.0023, 0.0010, 0.0000, 0.0035, 0.0035, 0.0018, 0.0013, 0.0023, 0.0020, 0.0011])
    x_g_st_mean = 100*np.array([0.8788, 0.8779, 0.8765, 0.8807, 0.8754, 0.8785, 0.8815, 0.8815, 0.8810, 0.8811, 0.8787])
    x_g_st_std  = 100*np.array([0.0014, 0.0034, 0.0002, 0.0016, 0.0033, 0.0017, 0.0008, 0.0008, 0.0013, 0.0031, 0.0011])
    x_d_mean    = 100*np.array([0.8807, 0.8812, 0.8794, 0.8788, 0.8815, 0.8806, 0.8794])
    x_d_std     = 100*np.array([0.0036, 0.0021, 0.0010, 0.0005, 0.0009, 0.0039, 0.0033])
    x_d_st_mean = 100*np.array([0.8801, 0.8816, 0.8794, 0.8755, 0.8805, 0.8811, 0.8801])
    x_d_st_std  = 100*np.array([0.0028, 0.0029, 0.0011, 0.0017, 0.0011, 0.0025, 0.0020])
    plot_results(x_local, x_local_std, x_g_mean, x_g_std, x_g_st_mean, x_g_st_std, x_d_mean, x_d_std, x_d_st_mean, x_d_st_std, title="BERT (German)", ylim=(86, 89), lloc=8)
