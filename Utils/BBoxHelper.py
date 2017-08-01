import math

def IOU(box1, box2):
    cx1 = box1[0]
    cy1 = box1[1]
    w1  = box1[2]
    h1  = box1[3]

    cx2 = box2[0]
    cy2 = box2[1]
    w2  = box2[2]
    h2  = box2[3]

    interX = max(0, min(cx1 + w1 / 2., cx2 + w2 / 2.) - max(cx1 - w1 / 2., cx2 - w2 / 2.))
    interY = max(0, min(cy1 + h1 / 2., cy2 + h2 / 2.) - max(cy1 - h1 / 2., cy2 - h2 / 2.))

    iterArea = interX * interY

    area1 = w1 * h1
    area2 = w2 * h2

    IOU = iterArea / (area1 + area2 - iterArea)
    return IOU


def same_bbox(_box1, _box2):
    _thres = 0.0001
    if (abs(_box1[0] - _box2[0]) < _thres and \
        abs(_box1[1] - _box2[1]) < _thres and \
        abs(_box1[2] - _box2[2]) < _thres and \
        abs(_box1[3] - _box2[3]) < _thres):
        return True
    return False

# Check if center of box2 in box1
def Inside(box1, box2):
    cx1 = box1[0]
    cy1 = box1[1]
    w1 = box1[2]
    h1 = box1[3]

    cx2 = box2[0]
    cy2 = box2[1]

    if cx1 <= cx2 and cx2 <= cx1 + w1 and \
       cy1 <= cy2 and cy2 <= cy1 + h1:
       return True

    return False

def InterestBox(box1, box2, alpha, beta):
    cx1 = box1[0]
    cy1 = box1[1]
    w1 = box1[2]
    h1 = box1[3]

    cx2 = box2[0]
    cy2 = box2[1]
    w2 = box2[2]
    h2 = box2[3]

    interX = max(0, min(cx1 + w1 / 2., cx2 + w2 / 2.) - max(cx1 - w1 / 2., cx2 - w2 / 2.))
    interY = max(0, min(cy1 + h1 / 2., cy2 + h2 / 2.) - max(cy1 - h1 / 2., cy2 - h2 / 2.))

    iterArea = interX * interY

    area1 = w1 * h1
    area2 = w2 * h2

    ratio1 = iterArea / area1
    ratio2 = iterArea / area2

    ratioBox1 = w1 / h1
    ratioBox2 = w2 / h2
    temp1     = alpha / beta
    temp2     = beta / alpha
    if temp1 * ratioBox1 <= ratioBox2 and ratioBox2 <= temp2 * ratioBox1:
        return min(ratio1, ratio2), max(ratio1, ratio2)
    else:
        return 0, 0


def Distance(cx1, cy1, cx2, cy2):
    deltaX = cx2 - cx1
    deltaY = cy2 - cy1
    return math.sqrt(deltaX * deltaX + deltaY * deltaY)

def InterestBox1(box1, box2, alpha, beta):
    cx1 = box1[0]
    cy1 = box1[1]
    w1 = box1[2]
    h1 = box1[3]

    cx2 = box2[0]
    cy2 = box2[1]
    w2 = box2[2]
    h2 = box2[3]

    ratioBox1 = w1 / h1
    ratioBox2 = w2 / h2
    temp1 = alpha / beta
    temp2 = beta / alpha

    area1 = w1 * h1
    area2 = w2 * h2

    squareW = math.sqrt(area1)

    ratioArea = min(area1 / area2, area2 / area1)

    if temp1 * ratioBox1 <= ratioBox2 and ratioBox2 <= temp2 * ratioBox1 and ratioArea >= 0.5 and Distance(cx1, cy1, cx2, cy2) <= squareW:
        return True
    else:
        return False


def InterestBox2(box1, box2, alpha, beta):
    cx1 = box1[0]
    cy1 = box1[1]
    w1  = box1[2]
    h1  = box1[3]
    ratioBox1 = w1 / h1
    area1     = w1 * h1
    w1 = h1 = math.sqrt(area1)
    area1 = w1 * h1

    cx2 = box2[0]
    cy2 = box2[1]
    w2  = box2[2]
    h2  = box2[3]
    ratioBox2 = w2 / h2
    area2     = w2 * h2

    interX = max(0, min(cx1 + w1 / 2., cx2 + w2 / 2.) - max(cx1 - w1 / 2., cx2 - w2 / 2.))
    interY = max(0, min(cy1 + h1 / 2., cy2 + h2 / 2.) - max(cy1 - h1 / 2., cy2 - h2 / 2.))
    iterArea = interX * interY

    ratio1 = iterArea / area1
    ratio2 = iterArea / area2

    temp1     = alpha / beta
    temp2     = beta / alpha
    if temp1 * ratioBox1 <= ratioBox2 and ratioBox2 <= temp2 * ratioBox1:
        return ratio1, ratio2
    else:
        return 0, 0
