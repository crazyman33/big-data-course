# coding=utf-8
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 构造训练数据
x = np.arange(0., 10., 0.2)
m = len(x)  # 训练数据点数目
print
m
x0 = np.full(m, 1.0)
input_data = np.vstack([x0, x]).T  # 将偏置b作为权向量的第一个分量
target_data = 2 * x + 5 + np.random.randn(m)

# 两种终止条件
loop_max = 10000  # 最大迭代次数(防止死循环)
epsilon = 1e-3

# 初始化权值
np.random.seed(0)
theta = np.random.randn(2)

alpha = 0.001  # 步长(注意取值过大会导致振荡即不收敛,过小收敛速度变慢)
diff = 0.
error = np.zeros(2)
count = 0  # 循环次数
finish = 0  # 终止标志
minibatch_size = 5  # 每次更新的样本数
while count < loop_max:
    count += 1

    # minibatch梯度下降是在权值更新前对所有样例汇总误差，而随机梯度下降的权值是通过考查某个训练样例来更新的
    # 在minibatch梯度下降中，权值更新的每一步对多个样例求和，需要更多的计算

    for i in range(1, m, minibatch_size):
        sum_m = np.zeros(2)
        for k in range(i - 1, i + minibatch_size - 1, 1):
            dif = (np.dot(theta, input_data[k]) - target_data[k]) * input_data[k]
            sum_m = sum_m + dif  # 当alpha取值过大时,sum_m会在迭代过程中会溢出

        theta = theta - alpha * (1.0 / minibatch_size) * sum_m  # 注意步长alpha的取值,过大会导致振荡

    # 判断是否已收敛
    if np.linalg.norm(theta - error) < epsilon:
        finish = 1
        break
    else:
        error = theta
    print
    'loopcount = %d' % count, '\tw:', theta
print
'loop count = %d' % count, '\tw:', theta

# check with scipy linear regression
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, target_data)
print
'intercept = %s slope = %s' % (intercept, slope)

plt.plot(x, target_data, 'g*')
plt.plot(x, theta[1] * x + theta[0], 'r')
plt.show()