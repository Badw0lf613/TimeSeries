#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [dan dot marthaler at gmail dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGPs.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014
#================================================================================

import pyGPs
import numpy as np
import pandas as pd

# This demo will not only introduce GP regression model,
# but provides a gerneral insight of our tourbox.

# You may want to read it before reading other models.
# current possible models are:
#     pyGPs.GPR          -> Regression
#     pyGPs.GPC          -> Classification
#     pyGPs.GPR_FITC     -> Sparse GP Regression
#     pyGPs.GPC_FITC     -> Sparse GP Classification
#     pyGPs.GPMC         -> Muli-class Classification

def processXlsx(df,i,l,r,step):
    row = df.loc[i].values[0:-1]
    # print("row",row,type(row)) # numpy.ndarray
    rowbool = np.isnan(row)
    # print("rowbool",rowbool)
    rowbool_rev = (rowbool == False) # negate
    ind = np.where(rowbool_rev) # all non-zero index
    train_X = np.array(ind).reshape(-1, 1)
    # print("train_X",train_X)
    train_y = np.array(row[ind]).reshape(-1, 1)
    # print("train_y",train_y)
    test_X = np.arange(l, r, step).reshape(-1, 1)
    start = int((0-l)/step) # points num (r - l) / step
    # print("start",start)
    ind_final = [start,start+int(1/step),start+int(2/step),
                 start+int(3/step),start+int(4/step),
                 start+int(5/step),start+int(6/step),
                 start+int(7/step),start+int(8/step),
                 start+int(9/step),start+int(10/step),
                 start+int(11/step)]
    return train_X,train_y,test_X,ind_final

def writeXlsx(x,filename):
  data = pd.DataFrame(x)
  writer = pd.ExcelWriter(filename)		# write into xlsx
  data.to_excel(writer, 'page_1', float_format='%.5f')
  writer.save()

print('')
print('---------------------myGPR-------------------------')
filepath = 'data/finaldata/data_Zero_MaternRQ.xlsx'
df = pd.read_excel(filepath)
# print(df)
result = []
for i in df.index:  # df.index read by row row range(2)
    l = 0
    r = 12
    step = 0.001
    train_X,train_y,test_X,ind_final = processXlsx(df,i,l,r,step)
    print("train_X",train_X)
    print("train_y",train_y)
    print("test_X",test_X)
    print("ind_final",ind_final)
    # start regression
    print('More Advanced Example (using a non-zero mean and Matern7 kernel)')
    model = pyGPs.GPR()           # start from a new model
    m = pyGPs.mean.Zero()
    # k = 0.5 * pyGPs.cov.Linear() + pyGPs.cov.RBF() # Approximates RBF kernel
    # k = pyGPs.cov.Linear() * pyGPs.cov.RBF() # Approximates RBF kernel
    # k = pyGPs.cov.Matern() # Approximates RBF kernel
    # k = 0.5 * pyGPs.cov.Linear() + pyGPs.cov.Matern() # Approximates RBF kernel
    # k = pyGPs.cov.Matern() * pyGPs.cov.RQ() # Approximates RBF kernel
    # k = pyGPs.cov.RBF() * pyGPs.cov.RQ() # Approximates RBF kernel
    k = pyGPs.cov.Matern() + pyGPs.cov.RBF() # Approximates RBF kernel
    model.setPrior(kernel=k, mean=m)
    model.optimize(train_X, train_y)
    print('Optimized negative log marginal likelihood:', round(model.nlZ,3))
    ym, ys2, fmu, fs2, lp = model.predict(test_X)
    # print('ym[ind_final].flatten()',ym[ind_final].flatten())
    res = ym[ind_final].flatten()
    print('res before',res,type(res))
    # fix to zero
    res[res < 0] = 0
    print('res after',res)
    result.append(res) # to1-d
    if i % 10 == 0:
        model.plot()
    # post = model.posterior    # already known before
    # ym, ys2, fmu, fs2, lp = model.predict_with_posterior(post,test_X)
    # model.setNoise(log_sigma=np.log(0.1))
print("result",result)
writeXlsx(result, filename='data/result_Zero_MaternRQ_all.xlsx') # 05代表0.5
print('--------------------END OF DEMO-----------------------')