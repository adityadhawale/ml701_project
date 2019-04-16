import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def l1_lr_coef_path():
    coefs_l1 = np.load("coefs_l1_50.npy")
    cs_l1 = np.load("cs_l1_50.npy")

    sns.set(context="talk")
    plt.figure(dpi=200)
    plt.plot(np.log10(cs_l1), coefs_l1, marker='o')
    plt.xlabel('log(C)')
    plt.ylabel('Coefficients')
    plt.title('Logistic Regression Path')
    plt.show()

def l1_lr_weight_path():
    cv_l1 = np.load("cv_l1_50.npy")
    train_l1 = np.load("train_l1_50.npy")
    cs_l1 = np.load("cs_l1_50.npy")

    xs = np.log10(cs_l1)
    sns.set(context="talk")
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.figure(dpi=200)
    plt.plot(xs, train_l1, linewidth=3.0, label="Train Accuracy")
    plt.plot(xs, cv_l1, linewidth=3.0, label="CV Accuracy")
    plt.title(r"Varying $\lambda$ for Logistic Regression with L1 Reg.")
    plt.xlabel(r"log($\lambda$)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    plt.xlim(-3.2, 5.3)
    sns.despine()
    plt.legend()
    plt.text(xs[-1]+0.1, train_l1[-1]-0.01, s=str(np.round(train_l1[-1], 3)))
    plt.text(xs[-1]+0.1, cv_l1[-1]-0.01, s=str(np.round(cv_l1[-1], 3)))
    plt.show()
l1_lr_weight_path()

def l2_lr_hist_plot():
    coefs_l2 = np.load("coefs_l2.npy")
    cs_l2 = np.load("cs_l2.npy")
    print(cs_l2[0])
    plt.figure(dpi=200)
    plt.hist(coefs_l2[-1], bins=100, color="#34495e", label=r"$\lambda$="+"{:.2e}".format(1/cs_l2[-1]))
    plt.hist(coefs_l2[7], bins=100, color="#3498db", alpha=0.9, label=r"$\lambda$="+"{:.2e}".format(1/cs_l2[7]))
    plt.hist(coefs_l2[0], bins=100, color="#e74c3c", alpha=0.8, label=r"$\lambda$="+"{:.2e}".format(1/cs_l2[0]))
    plt.title(r"Effect of $\lambda$ on weight distribution")
    plt.xlabel("Weights")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def l2_lr_plot():
    cv_l2 = np.load("cv_l2.npy")
    train_l2 = np.load("train_l2.npy")
    cs_l2 = np.load("cs_l2.npy")

    xs = np.log10(cs_l2)
    sns.set(context="talk")
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.figure(dpi=200)
    plt.plot(xs, train_l2, linewidth=3.0, label="Train Accuracy")
    plt.plot(xs, cv_l2, linewidth=3.0, label="CV Accuracy")
    plt.title(r"Varying $\lambda$ for Logistic Regression with L2 Reg.")
    plt.xlabel(r"log($\lambda$)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    plt.xlim(-3.2, 5.3)
    sns.despine()
    plt.legend()
    plt.text(xs[-1]+0.1, train_l2[-1]-0.01, s=str(np.round(train_l2[-1], 3)))
    plt.text(xs[-1]+0.1, cv_l2[-1]-0.01, s=str(np.round(cv_l2[-1], 3)))
    plt.show()