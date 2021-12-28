import sys
from matplotlib.pyplot import axis
from Errors import *

def predict(model, X):
    '''
        Predicts the Output
        Args:
        -------------
        model: list
            A list of layer objects
        X: np.array
            Input values
    '''
    output = X
    # Forwarding input through every layer
    for layer in model:
        output = layer._forward(output)

    return output

def fit(model, X_train, Y_train, epochs = 500, batch_size = 1, shuffle = True):
    '''
        Builds the Model
        Args:
        -------------
        model: list
            A list of Layer objects
        X_train: nparray
            Training input values 
        y_train: nparray
            Training output values
        epochs: int (default: 500)
            Number of epochs :| 
        batch_size: int (default: 1)
            Batch size for SDE
        shuffle: bool default(True)
            For suffling training inputs
    '''
    error_over_epoch = []
    for j in range(epochs):
        idx = np.arange(len(X_train))
        if (shuffle): np.random.shuffle(idx)
        error = 0
        for i in range(0,len(idx) - batch_size + 1, batch_size):
            
            y_pred = predict(model, X_train[idx[i:i+batch_size]])
            
            error += mse(y_pred, Y_train[idx[i:i+batch_size]]).sum(axis = 0)
            
            grad_error = mse_prime(calc = y_pred, actual = Y_train[idx[i:i+batch_size]])
            
            grad_y = grad_error
            
            # Back propagating through every layer
            for layer in reversed(model):
                grad_y = layer._backward(grad_y)
                
        error_over_epoch.append(error)
        sys.stderr.write("\rEpoch %0*d of %d\tError:%.2f" %(len(str(epochs)), j+1, epochs, error))
        sys.stdout.flush()
    return(error_over_epoch)