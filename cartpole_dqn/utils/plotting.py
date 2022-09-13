import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, title, plots):
        self.title = title
        self.plots = plots
        self.data = {
            "reward": [],
            "loss": []
        }

    def set(self, name, data):
        self.data[name] = data

    def push(self, episode, reward, loss):
        self.data["reward"].append(reward)
        self.data["loss"].append(loss)

    def draw(self):
        plt.figure(self.plots)
        plt.clf()
        for plot in self.data:
            plt.plot(self.data[plot], label=plot)
        plt.legend()
        plt.pause(0.0001)
