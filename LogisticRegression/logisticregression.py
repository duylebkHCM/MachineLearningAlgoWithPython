import numpy as np

def sigmoid(S):
    return 1/(1 + np.exp(-S))

def prob(w,X):
    return sigmoid(X.dot(w))

def loss(w,X,y,lam):
    a = prob(w,X)
    loss_0 = -np.mean(y*np.log(a) + (1-y)*(1- np.log(a)))
    regular = 0.5*lam/X.shape[0]*np.sum(w[1:]*w[1:])
    return loss_0 + regular

def logistic_regression(w_init, X, y, lam, lr = 0.1, nepoches = 2000):
    N,d = X.shape[0], X.shape[1]
    ep = 0
    w = w_old = w_init
    loss_hist = [loss(w_init, X, y, lam)]

    while ep < nepoches:
        ep += 1
        mix_ids = np.random.permutation(N)
        for i in mix_ids:
            xi = X[i]
            yi = y[i]
            ai = sigmoid(xi.dot(w))
            if i == 0:
                w = w - lr*((ai - yi)*xi)
            w = w - lr*((ai - yi)*xi + lam*w[1:])
            loss_hist.append(loss(w, X, y, lam))

        if np.linalg.norm(w - w_old)/d < 1e-6:
            break
        w_old = w

    return w, loss_hist

def main():
    np.random.seed(2)
    X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
    2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
    y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

    Xbar = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
    w_init = np.random.randn(Xbar.shape[1])
    lam = 0.01
    w, loss_hist = logistic_regression(w_init, Xbar, y, lam, lr = 0.05, nepoches=1000)
    print('Solution of Logistic Regression:', w)
    print('Final loss:', loss(w, Xbar, y, lam))

if __name__ == '__main__':
    main()