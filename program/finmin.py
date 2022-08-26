from detecta import detect_peaks
import matplotlib.pyplot as plt
import numpy as np
from tem import test1,test2


def findmin(list, min):
    list_result = []
    list_up = detect_peaks(list, mpd=1, valley=False, show=False, edge=None)
    list_down = detect_peaks(list, mpd=1, valley=True, show=False, edge=None)
    #     print(len(list_up),len(list_down))
    for index, value in enumerate(list_down):
        if index < len(list_up) and index < len(list_down) and (list[list_up[index]] - list[list_down[index]] >= min):
            list_result.append(value)
    plt.figure(figsize=(14, 7),dpi=120)
    plt.plot(list, marker=6, mfc='r', mec='r', ms=5, markevery=list_result)
    plt.show()
    print("Done！")
    print(list_up)
    print(list_down)
    print(list_result)
    return list_result


test3 = [0, 1, 2, 3, 4, 3, 3, 2, 2, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9]

# list_result = findmin(test2, 0.1)

r = g = b = np.zeros((10,5))

g[1][1] = 15
print(g)
b[1][1] = 20
print(g)