import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, title, plots):
        self.title = title
        self.plots = plots
        self.data = {}

    def set(self, name, data):
        self.data[name] = data

    def draw(self):
        plt.figure(self.plots)
        plt.clf()
        for plot in self.data:
            plt.plot(self.data[plot])
        plt.pause(0.0001)
