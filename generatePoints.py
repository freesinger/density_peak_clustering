import numpy as np
import matplotlib.pyplot as plt

GENERATE_POINTS_DIST = './data/generatePoints_distance.txt'
GENERATE_POINTS = './data/generatePoints.txt'

r = np.random.RandomState(24)
p = r.randn(400, 2)
q = r.randn(400, 2) + 7
s = r.randn(400, 2) + 4

t = np.concatenate((p, q, s), axis=0)

with open(GENERATE_POINTS, 'w', encoding='utf-8') as f:
    for pos in range(len(t)):
        cor = t[pos]
        f.write(str(pos) + ' ' + str(cor[0]) + ' ' + str(cor[1]) + '\n')

d = lambda x, y: np.sqrt(np.power((x[0] - y[0]), 2) + np.power((x[1] - y[1]), 2))

with open(GENERATE_POINTS_DIST, 'w', encoding='utf-8') as f:
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            distance = d(t[i], t[j])
            f.write(str(i) + ' ' + str(j) + ' ' + str(distance) + '\n')

x = p[:, 0]
y = p[:, 1]
plt.plot(x, y, 'or', markersize=1, alpha=0.5, label='1')
# plt.show()

x = s[:, 0]
y = s[:, 1]
plt.plot(x, y, 'ob', markersize=1, alpha=0.5, label='2')

x = q[:, 0]
y = q[:, 1]
plt.plot(x, y, 'oc', markersize=1, alpha=0.5, label='3')
plt.legend()
# plt.axis([-3, 10, -3, 9])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Points Plot')
plt.savefig('./images/generatedPoints.png')
plt.show()