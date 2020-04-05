def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    intersection = (xB - xA) * (yB - yA)

    if intersection < 0:
        intersection = 0

    boxAArea = abs(boxA[2] - boxA[0]) * abs(boxA[3] - boxA[1])
    boxBArea = abs(boxB[2] - boxB[0]) * abs(boxB[3] - boxB[1])
    union = abs(boxAArea + boxBArea - intersection)

    iou = intersection / union
    return iou


if __name__ == '__main__':
    boxA = [72, 102, 416, 395]
    boxB = [70, 100, 400, 350]
    IOU = iou(boxA=boxA, boxB=boxB)
    print(IOU)
