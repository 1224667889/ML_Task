import matplotlib.pyplot as plt
from sklearn import datasets


def build_circle_figure(figure: plt.Figure, location=211) -> None:
    points, tags = datasets.make_circles(
        n_samples=400,
        shuffle=True,
        noise=.1,
        random_state=4103,
        factor=.1
    )
    colors = ["b" if tag else "m" for tag in tags]
    plot = figure.add_subplot(location)
    plot.set_title('data by make_circles()')
    plot.scatter(
        x=points[:, 0],
        y=points[:, 1],
        s=100,
        marker="o",
        c=colors,
    )


def build_make_moons(figure: plt.Figure, location=212) -> None:
    points, tags = datasets.make_moons(
        n_samples=400,
        shuffle=True,
        noise=.1,
        random_state=4103,
    )
    colors = ["b" if tag else "m" for tag in tags]
    plot = figure.add_subplot(location)
    plot.set_title('data by make_moons()')
    plot.scatter(
        x=points[:, 0],
        y=points[:, 1],
        s=100,
        marker="o",
        c=colors,
    )


if __name__ == '__main__':
    fig = plt.figure()
    build_circle_figure(fig)
    build_make_moons(fig)
    plt.tight_layout()
    plt.show()
