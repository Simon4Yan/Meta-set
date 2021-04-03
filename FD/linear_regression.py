import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# data preparation
acc = np.load('./learn/accuracy_mnist.npy')
data = np.load('./FD/fd_mnist.npy').reshape(-1, 1)

# Choose some sample sets as validation (also used in NN regression)
indice = 30
train_data = data[indice:]
train_acc = acc[indice:]
test_data = train_data[:indice]
test_acc = train_acc[:indice]

# linear regression
slr = LinearRegression()
slr.fit(train_data, train_acc)
test_pred = slr.predict(test_data)

# plot training dataset
plt.scatter(train_data, train_acc, color='#0000FF')
plt.plot(train_data, slr.predict(train_data), color='#FF0000')

ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('linear_regression_train.png')
plt.close()

# plot testing dataset
plt.scatter(test_data, test_acc, color='red')
plt.plot(test_data, slr.predict(test_data), color='blue')
plt.savefig('linear_regression_test.png')

print('*****'*5)
print('If you could observe the linear correlation from figures, then your implementations are all good!')
print('*****'*5)

# evaluation with metrics
print('Test on Validation Set..')
R2 = r2_score(test_acc, slr.predict(test_data))
RMSE = mean_squared_error(test_acc, slr.predict(test_data), squared=False)
MAE = mean_absolute_error(test_acc, slr.predict(test_data))
print('\nTest set: R2 :{:.4f} RMSE: {:.4f} MAE: {:.4f}\n'.format(R2, RMSE, MAE))

# analyze the statistical correlation
rho, pval = stats.spearmanr(test_data, test_acc)
print('\nRank correlation-rho', rho)
print('Rank correlation-pval', pval)

rho, pval = stats.pearsonr(test_data.reshape(-1, 1), test_acc.reshape(-1, 1))
print('\nPearsons correlation-rho', rho)
print('Pearsons correlation-pval', pval)

print('*****'*5)
print('\nAll done! Thanks!')
print('*****'*5)
