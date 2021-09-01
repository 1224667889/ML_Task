import matplotlib.pyplot as plt
from sklearn import datasets
import random

# 随机取点
x = 2 * random.random() - 1
y = 2 * random.random() - 1


def build_circle_figure(figure: plt.Figure, k=15) -> None:
    # 生成数据
    points, tags = datasets.make_circles(
        n_samples=10000,
        shuffle=True,
        noise=.1,
        random_state=4103,
        factor=.1
    )
    # 标色
    colors = ["g" if tag else "r" for tag in tags]
    plot1 = figure.add_subplot(211)
    plot1.set_title('data by make_circles()')
    plot1.scatter(
        x=points[:, 0],
        y=points[:, 1],
        s=100,
        marker="o",
        c=colors,
    )
    plot1.scatter(x, y, c='b', s=100, marker='*')
    import time
    t1 = time.time()
    a = b = 0
    dis_list = [((point[0]-x)**2+(point[1]-y)**2)**0.5 for point in points]
    # 改变距离近的点的颜色
    for i in range(k):
        if colors[dis_list.index(min(dis_list))] == 'r':
            a += 1
        else:
            b += 1
        colors[dis_list.index(min(dis_list))] = 'b'
        dis_list[dis_list.index(min(dis_list))] = 500
    print(time.time()-t1)
    plot2 = figure.add_subplot(212)
    plot2.set_title(f'data by make_circles() K={k} a:b={a}:{b}')
    plot2.scatter(
        x=points[:, 0],
        y=points[:, 1],
        s=100,
        marker="o",
        c=colors,
    )
    plot2.scatter(x, y, c='b', s=100, marker='*')


if __name__ == '__main__':
    fig = plt.figure()
    build_circle_figure(fig)
    plt.tight_layout()
    plt.show()
