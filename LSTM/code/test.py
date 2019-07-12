import numpy as np

from lstm import LstmParam, LstmNetwork


class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """

    # define loss function
    @classmethod
    def loss(self, pred, label):

        return (pred[0] - label) ** 2

    # differential loss function
    @classmethod
    def bottom_diff(self, pred, label):

        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)

        return diff


def example_0():

    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 50
    lstm_param = LstmParam(mem_cell_ct, x_dim)

    # print(lstm_param.wg)
    # print(lstm_param.wg[0]) # -0.1~0.1 array
    # print(len(lstm_param.wg[0])) # input 150

    lstm_net = LstmNetwork(lstm_param)
    y_list = [-0.5, 0.2, 0.1, -0.5]
    print("y_list input:", y_list)

    # input transform to vector
    input_val_arr = [np.random.random(x_dim) for _ in y_list]
    # print(lstm_param)
    # print(input_val_arr) # (4, 50) array

    # iteration 100
    for cur_iter in range(100):
        print("iter", "%2s" % str(cur_iter), end=": ")

        # next epoch input
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])

        print("y_pred = [" + ", ".join(["%2.5f" % lstm_net.lstm_node_list[ind].state.h[0] for ind in range(len(y_list))]) + "]", end=", ")

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss:", "%.3e" % loss)

        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()


if __name__ == "__main__":

    example_0()


