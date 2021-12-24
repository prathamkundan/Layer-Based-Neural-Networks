from Errors import *

def predict(model, X):
    output = X
    for layer in model:
        output = layer._forward(output)

    return output
    
def fit(model, X_train, Y_train, epochs = 500):
    error_over_epoch = []
    for i in range(epochs):
        error = 0
        for X, Y in zip(X_train,Y_train):
            y_pred = predict(model, X)
            error += mse(y_pred, Y)
            grad_y = mse_prime(calc = y_pred, actual = Y)
            
            for layer in reversed(model):
                grad_y = layer._backward(grad_y)
                
        error_over_epoch.extend(error)
        print(error)
    return(error_over_epoch)