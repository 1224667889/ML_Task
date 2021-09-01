import matplotlib.pyplot as plt
import numpy


if __name__ == '__main__':
    with open("iris.data") as f:
        lines = f.readlines()
    dataset = []
    for line in lines[:-1]:
        dataset.append(line.split(","))
    dataset = numpy.array(dataset)
    colors = []
    for label in dataset[:, 4]:
        if "setosa" in label:
            colors.append("r")
        elif "versicolor" in label:
            colors.append("g")
        else:
            colors.append("b")

    plt.scatter(
        x=dataset[:, 0],
        y=dataset[:, 1],
        s=100,
        marker="o",
        c=colors
    )
    plt.scatter(
        x=dataset[:, 2],
        y=dataset[:, 3],
        s=20,
        marker="o",
        c=colors
    )
    plt.show()
