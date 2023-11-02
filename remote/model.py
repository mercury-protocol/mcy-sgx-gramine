def run(data):
    import numpy as np
    from sklearn.linear_model import LinearRegression

    data = np.loadtxt(data, delimiter=",", dtype=int, skiprows=1).transpose()
    x = data[0].reshape((-1, 1))
    y = data[1]

    model = LinearRegression()
    model.fit(x, y)

    return model
