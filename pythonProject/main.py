import numpy as np
#hello

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Производная от sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true и y_pred являются массивами numpy с одинаковой длиной
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:
    def __init__(self, w_count_in, b_count):
        self.w_count_in = w_count_in
        self.w_count = w_count_in * (b_count - 1) + b_count - 1
        self.w = []
        self.b = []
        self.h = [0] * (b_count - 1)
        self.d_ypred_d_h = [0] * len(self.h)
        self.d_ypred_d_w = [0] * len(self.h)
        self.d_ypred_d_b = 0
        self.d_h_d_w = [0] * len(self.h) * w_count_in
        self.d_h_d_b = [0] * len(self.h)

        for i in range(0, self.w_count):
            self.w.append(np.random.normal())

        for i in range(0, b_count):
            self.b.append(np.random.normal())

    def feedforward(self, x):
        k = 0
        for i in range(0, len(self.b) - 1):
            summ = 0
            for j in range(0, self.w_count_in):
                summ += self.w[j + k] * x[j]
            k += self.w_count_in
            self.h[i] = (sigmoid(summ + self.b[i]))

        summ = 0
        for i in range(0, len(self.b) - 1):
            summ += self.w[len(self.w) - len(self.b) + i] * self.h[i]
        o = sigmoid(summ + self.b[-1])

        return o

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 10000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Выполняем обратную связь (нам понадобятся эти значения в дальнейшем)

                sums = [0] * len(self.b)
                for i in range(0, len(self.b) - 1):
                    for j in range(0, len(self.w) // len(self.b)):
                        sums[i] += self.w[j + j * (len(self.w) // len(self.b))] * x[j]
                    self.h[i] = (sigmoid(sums[i] + self.b[i]))

                # +++
                for i in range(0, len(self.b) - 1):
                    sums[-1] += self.w[len(self.w) - len(self.b) - 1 + i] * self.h[i]
                o = sigmoid(sums[-1] + self.b[-1])

                y_pred = o

                # --- Подсчет частных производных
                # --- Наименование: d_L_d_w1 представляет "частично L / частично w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # h
                j, k = 0, 0
                for i in range(0, len(self.d_h_d_w)):
                    self.d_h_d_w[i] = x[j] * deriv_sigmoid(sums[k])
                    j += 1
                    if j == len(x):
                        j = 0
                        k += 1

                for i in range(0, len(self.d_h_d_b)):
                    self.d_h_d_b[i] = deriv_sigmoid(sums[i])

                # o
                for i in range(0, len(self.h)):
                    self.d_ypred_d_w[i] = (self.h[i] * deriv_sigmoid(sums[-1]))

                j = 0
                for i in range(self.w_count - len(self.b) + 1, self.w_count):
                    self.d_ypred_d_h[j] = (self.w[i] * deriv_sigmoid(sums[-1]))
                    j += 1

                d_ypred_d_b = deriv_sigmoid(sums[-1])

                j, c = 0, 0
                for i in range(0, self.w_count - self.w_count_in):
                    if c > self.w_count_in:
                        self.b[j] -= learn_rate * d_L_d_ypred * self.d_ypred_d_h[j] * self.d_h_d_b[j]
                        j += 1
                        c = 0
                    self.w[i] -= learn_rate * d_L_d_ypred * self.d_ypred_d_h[j] * self.d_h_d_w[i]

                    c += 1

                j = 0
                for i in range(self.w_count - len(self.b) + 1, self.w_count):
                    self.w[i] -= learn_rate * d_L_d_ypred * self.d_ypred_d_w[j]
                    j += 1
                self.b[-1] -= learn_rate * d_L_d_ypred * d_ypred_d_b

            # --- Подсчитываем общую потерю в конце каждой фазы
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print(f"Epoch {epoch} loss: {loss}")


# Определение набора данных
data = np.array([
    [-2, -1],  # Alice
    [25, 6],  # Bob
    [17, 4],  # Charlie
    [-15, -6],  # Diana
])

all_y_trues = np.array([
    1,  # Alice
    0,  # Bob
    0,  # Charlie
    1,  # Diana
])

# Тренируем нашу нейронную сеть!
network = OurNeuralNetwork(2, 3)
network.train(data, all_y_trues)

emily = np.array([-7, -3])
frank = np.array([20, 2])
print("Emily: %.9f" % network.feedforward(emily))  # 0.951 - F
print("Frank: %.9f" % network.feedforward(frank))  # 0.039 - M
input("Press any key to exit...")