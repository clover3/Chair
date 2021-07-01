import math


def main():

    def ce(y, p):
        if_one_loss = -math.log(p)
        if_zero_loss = -math.log(1-p)
        v = y * if_one_loss + (1-y) * if_zero_loss
        return v

    for y in [0, 1]:
        p = 0.01
        while p < 1:
            print("{0:.2f} {1:.2f} {2:.2f}".format(y, p, ce(y, p)))
            p += 0.05

    y = 1
    p = 0.001
    print("{0:.4f} {1:.4f} {2:.4f}".format(y, p, ce(y, p)))


if __name__ == "__main__":
    main()