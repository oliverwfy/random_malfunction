from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

a = np.array([0.5, 0.2, 0.8, 0.9])
x = np.arange(len(a))

reg = LinearRegression(fit_intercept=False).fit(x.reshape(-1, 1), a-0.5)
print(reg.coef_)
print(reg.intercept_)

x_ls = np.linspace(0, len(a))
plt.scatter(x, a)
plt.plot(x_ls, x_ls*reg.coef_ + 0.5)
plt.show()






