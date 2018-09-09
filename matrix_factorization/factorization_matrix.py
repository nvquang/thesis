from keras.models import Sequential
from keras.layers import Dense
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
import numpy as np


from sklearn.model_selection import train_test_split


def create_model(list_of_input_dim, X_train, y_train, epochs, batch_size):
    model = Sequential()
    for i in range(len(list_of_input_dim) - 1):
        input_dim = list_of_input_dim[i]
        output_dim = list_of_input_dim[i+1]
        model.add(Dense(output_dim, input_dim=input_dim, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model


def factorization_matrix(matrix_w):
    U, s, VT = np.linalg.svd(matrix_w, full_matrices=True)

    # create m x n Sigma matrix
    Sigma = zeros((matrix_w.shape[0], matrix_w.shape[1]))
    # populate Sigma with n x n diagonal matrix

    n = min(matrix_w.shape[0], matrix_w.shape[1])
    Sigma[:n, :n] = diag(s)

    return U, Sigma, VT


def factorization_matrix_dim(matrix_shape):
    m = matrix_shape[0]
    n = matrix_shape[1]
    return [(m, m), (m, n), (n, n)]


def create_factorization_model(original_model):
    weights = []
    for weight_index in range(len(original_model.get_weights()) - 2):
        if weight_index % 2 == 0:
            weights.append(original_model.get_weights()[weight_index])

    new_model = Sequential()
    new_model_weights = []
    for weight in weights:
        dims = factorization_matrix_dim(weight.shape)
        new_model.add(
            Dense(dims[0][1], input_dim=dims[0][0], activation='relu'))
        new_model.add(
            Dense(dims[1][1], input_dim=dims[1][0], activation='relu'))
        new_model.add(
            Dense(dims[2][1], input_dim=dims[2][0], activation='relu'))

        U, Sigma, VT = factorization_matrix(weight)
        new_model_weights.append(U)
        new_model_weights.append(np.zeros((dims[0][1], )))
        new_model_weights.append(Sigma)
        new_model_weights.append(np.zeros((dims[1][1], )))
        new_model_weights.append(VT)
        new_model_weights.append(np.zeros((dims[2][1], )))

    new_model.add(Dense(1, activation='sigmoid'))

    new_model_weights.append(original_model.get_weights()[-2])
    new_model_weights.append(original_model.get_weights()[-1])

    print("#### PRINT dimestion of model")
    print("## ORIGINAL MODEL: ")
    for wwww in original_model.get_weights():
        print(wwww.shape)
    print("## New model: ")
    for wwww in new_model.get_weights():
        print(wwww.shape)
    print("## weights: ")
    for w in new_model_weights:
        print(w.shape)

    new_model.set_weights(new_model_weights)
    new_model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return new_model


def main():
    dataset = np.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=42)

    model = create_model(list_of_input_dim=[
                         8, 500, 100, 50, 4 ], X_train=X_train, y_train=y_train, epochs=5, batch_size=1)
    f_model = create_factorization_model(model)

    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))

    scores = f_model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (f_model.metrics_names[1], scores[1]))


if __name__ == "__main__":
    main()
