import numpy as np

def get_orientation(image):
    direction = image.direction

    orientation = []
    for i in range(3):
        row = direction[:,i]
        idx = np.where(np.abs(row)==np.max(np.abs(row)))[0][0]

        if idx == 0:
            if row[idx] < 0:
                orientation.append('L')
            else:
                orientation.append('R')
        elif idx == 1:
            if row[idx] < 0:
                orientation.append('P')
            else:
                orientation.append('A')
        elif idx == 2:
            if row[idx] < 0:
                orientation.append('S')
            else:
                orientation.append('I')
    return ''.join(orientation)