#%%
import pandas as pd
import matplotlib.pyplot as plt

# ## Import dataset
#%%
data = pd.read_csv("Salary_Data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
y = y.reshape(30, 1)

#%%
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#%%
from Regressor import Regressor
model = Regressor()
model.fit(X, y)
y_pred = model.predict()

#%%
# Reverse Standard scaling
X = sc_X.inverse_transform(X)
y_pred = sc_y.inverse_transform(y_pred)

# %%
# Regression model visuals
 
plt.scatter(X, y, color = 'red')
plt.plot(X, model.predict(), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# %%
# Cost function plot

plt.plot(np.arange(1, len(model.cost_arr)+1), model.cost_arr, color = 'blue')
plt.show()
