import matplotlib.pyplot as plt
from IPython import display


plt.ion()


def plot(scores, mean_scores, file_name="plotting.png"):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.title("Training...")
    plt.xlabel("Number of games")
    plt.ylabel("Scores")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, scores[-1], str(mean_scores[-1]))
    plt.savefig(file_name)
