import numpy as np
import matplotlib.pyplot as plt

GENERATE_POINTS_DIST = './data/generatePoints_distance.txt'
GENERATE_POINTS = './data/generatePoints.txt'

r = np.random.RandomState(24)
o = r.randn(400, 2)
o[:, 0] += 2
o[:, 1] += 6
u = r.randn(400, 2)
u[:, 0] += 4
u[:, 1] -= 0.5
v = r.randn(400, 2)
v[:, 0] += 7
v[:, 1] -= 0.5
p = r.randn(400, 2)
q = r.randn(400, 2) + 3
# q[:, 0] += 3
# q[:, 1] += 9
s = r.randn(400, 2) + 6

t = np.concatenate((o, p, q, s, u, v), axis=0)

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

# Without labels
x, y = t[:, 0], t[:, 1]
plt.plot(x, y, 'ok', markersize=1, alpha=0.5)
# plt.axis([-3, 10, -3, 9])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Points Plot')
plt.savefig('./images/generatedPoints.png')
plt.close()

color = {0: 'c', 1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'y'}
cluster = [o, p, q, s, u, v]
for i in range(len(cluster)):
    cur = cluster[i]
    x, y = cur[:, 0], cur[:, 1]
    plt.scatter(x, y, s=1, c=color[i], alpha=0.7, label=i + 1)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Points with Lable')
plt.savefig('./images/generatedColoredPoints.png')
plt.show()