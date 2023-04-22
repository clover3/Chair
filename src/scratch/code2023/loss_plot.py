import matplotlib.pyplot as plt
import numpy as np
import re
from cpath import output_path
from misc_lib import path_join


def parse_line(line):
    # line = "INFO    chair   04-20 03:11:12 step 10 loss=92.023598  verbosity_loss=11.2127 acc_loss=0.2"
    m_step = re.search(r"step (\d+) ", line)
    m_loss = re.search(r" acc_loss=(\d+\.\d+)", line)
    step = int(m_step.group(1))
    loss = float(m_loss.group(1))
    return step, loss


def main():
    file_path = path_join(output_path, "loss_log.txt")
    f = open(file_path, "r")
    xy = map(parse_line, f)
    # print(list(xy))
    x, y = zip(*xy)
    plt.plot(list(x)[10:-1], list(y)[10:-1])  # plot the function
    plt.xlabel('x')  # add a label to the x-axis
    plt.ylabel('y')  # add a label to the y-axis
    plt.show()  # display the graph



if __name__ == "__main__":
    main()