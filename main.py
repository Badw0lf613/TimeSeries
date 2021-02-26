import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import minimize
import pandas as pd

class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)

        # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[
                1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            return loss.ravel()

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
                           bounds=((1e-4, 1e4), (1e-4, 1e4)),
                           method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]

        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)

        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)

def y(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
    return y.tolist()

def writeXlsx(x,filename):
  data = pd.DataFrame(x)
  writer = pd.ExcelWriter(filename)		# 写入Excel文件
  data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
  writer.save()

train_X = np.array([1, 3, 7]).reshape(-1, 1)
train_y = np.array([77.83, 0.026, 0.003]).reshape(-1, 1)
test_X = np.arange(0, 10, 0.1).reshape(-1, 1)

gpr = GPR(optimize=True)
gpr.fit(train_X, train_y)
mu, cov = gpr.predict(test_X)
test_y = mu.ravel()
uncertainty = 1.96 * np.sqrt(np.diag(cov))
plt.figure()
plt.title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
plt.plot(test_X, test_y, label="predict")
plt.scatter(train_X, train_y, label="train", c="red", marker="x")
plt.legend()

if __name__ == '__main__':
    filepath = 'data\data.xlsx'
    df = pd.read_excel(filepath)
    print(df.head(2))
    # df = df.head(2)
    # for i in df.index:  # 逐行读取
    #     # pre = df._get_value(i, 'pre')
    #     # w2 = df._get_value(i, '2W')
    #     # w4 = df._get_value(i, '4W')
    #     # w6 = df._get_value(i, '6W')
    #     # w8 = df._get_value(i, '8W')
    #     # w10 = df._get_value(i, '10W')
    #     # w12 = df._get_value(i, '12W')
    #     # print(pre,w2,w4,w6,w8,w10,w12)
    #     row = df.loc[i].values[0:-1]
    #     print("row",row,type(row)) # numpy.ndarray
    #     rowbool = np.isnan(row)
    #     print("rowbool",rowbool)
    #     rowbool_rev = (rowbool == False) # 取反
    #     ind = np.where(rowbool_rev) # 所有为非空元素的下标
    #     train_X = np.array(ind).reshape(-1, 1)
    #     print("train_X",train_X)
    #     train_y = np.array(row[ind]).reshape(-1, 1)
    #     print("train_y",train_y)
    #     test_X = np.arange(-10, 10, 0.1).reshape(-1, 1)
    #     # print(test_X)
    #     gpr = GPR(optimize=True)
    #     gpr.fit(train_X, train_y)
    #     mu, cov = gpr.predict(test_X)
    #     test_y = mu.ravel()
    #     uncertainty = 1.96 * np.sqrt(np.diag(cov))
    #     plt.figure()
    #     plt.title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
    #     plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
    #     plt.plot(test_X, test_y, label="predict")
    #     plt.scatter(train_X, train_y, label="train", c="red", marker="x")
    #     plt.legend()