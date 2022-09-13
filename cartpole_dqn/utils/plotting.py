import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, title, plots):
        self.title = title
        self.plots = plots
        self.data = {
            "reward": [],
            "loss": []
        }

        self.fig, self.ax = plt.subplots(2, 1, layout='constrained')
        self.ax[0].set_title("loss")
        self.ax[1].set_title("reward")

        self.ax[0].grid(True)
        self.ax[1].grid(True)

    def set(self, name, data):
        self.data[name] = data

    def push(self, episode, reward, loss):
        self.data["reward"].append(reward)
        self.data["loss"].append(loss)

    def draw(self):
        self.ax[0].plot(self.data["loss"], color="orange")
        self.ax[1].plot(self.data["reward"], color="blue")
        plt.pause(0.1)
