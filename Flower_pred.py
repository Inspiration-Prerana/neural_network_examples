from matplotlib import pyplot as plt
import numpy as np

# each point is length, width, type (0, 1)

data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, .5,  1],
        [2,   .5,  0],
        [5.5,  1,  1],
        [1,    1,  0]]

mystery_flower = [4.5, 1]


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))


T = np.linspace(-5, 5, 100)
# plt.plot(T, sigmoid(T), c='r')
# plt.plot(T, sigmoid_p(T), c='b')


# scatter plot
def vis_data():
    plt.grid()

    for i in range(len(data)):
        c = 'r'
        if data[i][2] == 0:
            c = 'b'
        plt.scatter([data[i][0]], [data[i][1]], c=c)

    plt.scatter([mystery_flower[0]], [mystery_flower[1]], c='black')


# training loop
def train():
    learning_rate = 0.1
    iteration = 10000
    costs = []

    w1 = np.random.rand()
    w2 = np.random.rand()
    b = np.random.rand()
    for i in range(iteration):
        ri = np.random.randint(len(data))
        point = data[ri]

        z = point[0] * w1 + point[1] * w2 + b
        pred = sigmoid(z)

        target = point[2]
        cost = np.square(pred - target)

        if i % 100 == 0:
            c = 0
            for j in range(len(data)):
                p = data[j]
                p_pred = sigmoid(p[0] * w1 + p[1] * w2 + b)
                c += np.square(p_pred - p[2])
                costs.append(c)

        dcost_pred = 2 * (pred - target)
        dpred_dz = sigmoid_p(z)
        dz_dw1 = point[0]
        dz_dw2 = point[1]
        dz_db = 1

        dcost_dz = dcost_pred * dpred_dz
        # partial derivative
        dcost_dw1 = dcost_dz * dz_dw1
        dcost_dw2 = dcost_dz * dz_dw2
        dcost_db = dcost_dz * dz_db

        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        b = b - learning_rate * dcost_db
    return costs, w1, w2, b


costs, w1, w2, b = train()

# fig = plt.plot(costs)

# prediction for mystery_flower
z = w1 * mystery_flower[0] + w2 * mystery_flower[1] + b
pred = sigmoid(z)

print(pred)
print("close to 0 -> blue, close to 1 -> red")

# check out prediction in x,y plane
for x in np.linspace(0, 6, 20):
    for y in np.linspace(0, 3, 20):
        pred = sigmoid(w1 * x + w2 * y + b)
        c = 'b'
        if pred > .5:
            c = 'r'
        plt.scatter([x], [y], c=c, alpha=.2)
vis_data()
plt.show()
