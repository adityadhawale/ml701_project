nn = []
train = []
test = []
cv = []

nn.append(1.0)
train.append(1.0)
test.append(0.2081447963800905)
cv.append(0.22549019607843138)

nn.append(2.0)
train.append(0.6215686274509804)
test.append(0.18929110105580693)
cv.append(0.16666666666666666)

nn.append(3.0)
train.append(0.5006535947712418)
test.append(0.18778280542986425)
cv.append(0.19117647058823528)

nn.append(4.0)
train.append(0.47058823529411764)
test.append(0.19834087481146304)
cv.append(0.19117647058823528)

nn.append(5.0)
train.append(0.4418300653594771)
test.append(0.2171945701357466)
cv.append(0.20098039215686275)

nn.append(6.0)
train.append(0.4300653594771242)
test.append(0.21794871794871795)
cv.append(0.2107843137254902)

nn.append(7.0)
train.append(0.42549019607843136)
test.append(0.222473604826546)
cv.append(0.22058823529411764)

nn.append(8.0)
train.append(0.41830065359477125)
test.append(0.22624434389140272)
cv.append(0.23039215686274508)

nn.append(9.0)
train.append(0.40980392156862744)
test.append(0.23076923076923078)
cv.append(0.23039215686274508)

nn.append(10.0)
train.append(0.3954248366013072)
test.append(0.23001508295625941)
cv.append(0.22058823529411764)

nn.append(20.0)
train.append(0.3254901960784314)
test.append(0.21417797888386123)
cv.append(0.18627450980392157)

nn.append(40.0)
train.append(0.269281045751634)
test.append(0.19532428355957768)
cv.append(0.18627450980392157)

nn.append(80.0)
train.append(0.19607843137254902)
test.append(0.16440422322775264)
cv.append(0.14705882352941177)

nn.append(160.0)
train.append(0.13594771241830064)
test.append(0.12066365007541478)
cv.append(0.12745098039215685)

nn.append(320.0)
train.append(0.07908496732026143)
test.append(0.07767722473604827)
cv.append(0.08823529411764706)

nn.append(640.0)
train.append(0.030065359477124184)
test.append(0.026395173453996983)
cv.append(0.049019607843137254)




import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

# coefs_l2 = np.load("coefs_l2.npy")
# cs_l2 = np.load("cs_l2.npy")

coefs_l2 = nn
cs_l2 = train


print(cs_l2[0])
plt.figure(dpi=200)
# plt.hist(coefs_l2[-1], bins=100, color="#34495e", label=r"$\lambda$="+"{:.2e}".format(1/cs_l2[-1]))
# plt.hist(coefs_l2[7], bins=100, color="#3498db", alpha=0.9, label=r"$\lambda$="+"{:.2e}".format(1/cs_l2[7]))
# plt.hist(coefs_l2[0], bins=100, color="#e74c3c", alpha=0.8, label=r"$\lambda$="+"{:.2e}".format(1/cs_l2[0]))



xs = nn
sns.set(context="talk")
sns.set_style("whitegrid", {'axes.grid' : False})
plt.figure(dpi=200)
plt.plot(xs, train, linewidth=3.0, label="Train Accuracy")
#plt.plot(xs, test, linewidth=3.0, label="Test Accuracy")
plt.plot(xs, cv, linewidth=3.0, label="CV Accuracy")
plt.title(r"Effect of Num Neighbors on Accuracy")
plt.xlabel("Num Neighbors")
plt.ylabel("Accuracy")


plt.plot([8, 8], [0, max(train)], color="black", linewidth=.5)
# plt.ylim(0, 1.1)
# plt.xlim(0, 10)
sns.despine()
plt.legend()
plt.subplots_adjust(left=0.2)

# plt.text(xs[3]+0.1, train_l2[3]-0.01, s=str(np.round(train_l2[3], 3)))
# plt.text(xs[-1]+0.1, cv_l2[-1]-0.01, s=str(np.round(cv_l2[-1], 3)))
plt.show()



# def l2_lr_plot():
#     cv_l2 = np.load("cv_l2.npy")
#     train_l2 = np.load("train_l2.npy")
#     cs_l2 = np.load("cs_l2.npy")

#     xs = np.log10(cs_l2)
#     sns.set(context="talk")
#     sns.set_style("whitegrid", {'axes.grid' : False})
#     plt.figure(dpi=200)
#     plt.plot(xs, train_l2, linewidth=3.0, label="Train Accuracy")
#     plt.plot(xs, cv_l2, linewidth=3.0, label="CV Accuracy")
#     plt.title(r"Varying $\lambda$ for Logistic Regression with L2 Reg.")
#     plt.xlabel(r"log($\lambda$)")
#     plt.ylabel("Accuracy")
#     plt.ylim(0, 1.1)
#     plt.xlim(-3.2, 5.3)
#     sns.despine()
#     plt.legend()
#     plt.text(xs[-1]+0.1, train_l2[-1]-0.01, s=str(np.round(train_l2[-1], 3)))
#     plt.text(xs[-1]+0.1, cv_l2[-1]-0.01, s=str(np.round(cv_l2[-1], 3)))
#     plt.show()












