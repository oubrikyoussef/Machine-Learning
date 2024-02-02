# Linear Regression


```python
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
```

# 1. Dataset


```python
x,y = make_regression(n_samples = 100 , n_features = 1 , noise = 10)
y = y + (1/2)*abs(y)
plt.scatter(x,y)
```




    <matplotlib.collections.PathCollection at 0x1db1cc90e50>




    
![png](output_3_1.png)
    



```python
y = y.reshape(y.shape[0],1)
#dimensions verification
print(x.shape,y.shape)
```

    (100, 1) (100, 1)
    


```python
X = np.hstack((x, np.ones(x.shape)))
X = np.hstack((x ** 2, X))
#dimensions verification
print(X.shape)

```

    (100, 3)
    


```python
theta = np.random.randn(3,1)
#dimensions verification
print(theta.shape)
```

    (3, 1)
    

# 2. Model


```python
def model(X,theta):
    return X.dot(theta)
```


```python
plt.scatter(x,model(X,theta),c='g')
plt.scatter(x,y)
```




    <matplotlib.collections.PathCollection at 0x1db1d076c10>




    
![png](output_9_1.png)
    


# 3. Cost Function


```python
def cost_function(X , theta , y) :
    m = len(y)
    return (1 / (2 * m)) * np.sum((model(X , theta) - y) ** 2)
```


```python
cost_function(X , theta , y)
```




    3277.9583611965336



# 4. Gradient Descent


```python
def grad(X, y, theta):
    m = len(y)
    return (1 / m) * (X.T.dot(model(X, theta) - y))
print(theta.shape)  
```

    (3, 1)
    


```python
def gradient_descent(X, y, theta, l_rate, n):
    cost_history = np.zeros(n)
    for i in range(n):
        theta = theta - l_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X , theta , y)
    return theta , cost_history
```


```python
theta_final , cost_history = gradient_descent(X, y, theta, l_rate=0.01, n=1000)
print(theta_final.shape)  
```

    (3, 1)
    


```python
plt.scatter(x,y)
predection = model(X,theta_final) 
plt.scatter(x,predection,c='g')
```




    <matplotlib.collections.PathCollection at 0x1db1d0f90d0>




    
![png](output_17_1.png)
    



```python
plt.plot(range(1000),cost_history)
```




    [<matplotlib.lines.Line2D at 0x1db1cc2c450>]




    
![png](output_18_1.png)
    



```python
def determination_coef(y , pred) :
    u = ((y - pred) ** 2).sum()
    v = ((y - y.mean()) ** 2).sum()
    return 1 - (u / v)
```


```python
determination_coef(y , predection)
```




    0.9746317696000203




```python

```
