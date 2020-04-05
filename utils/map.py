def map(x1, x2, y1, y2):
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    return m, c


if __name__ == '__main__':
    print(map(100, 180, .05, .2))