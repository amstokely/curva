import subprocess

import numpy as np


def generalizedCorrelation(
        x,
        y,
        numFrames
) -> float:
    numWindows = int(numFrames / 1024)
    psi = np.zeros([1024 + 1])
    psi[1] = -0.57721566490153
    for indx in range(1024):
        if indx > 0:
            psi[indx + 1] = psi[indx] + 1 / (indx)
    phi = np.zeros(9, dtype=np.float64)
    for tmpindx in range(1, 9):
        phi[tmpindx] = psi[tmpindx] - 1 / tmpindx
    mi = 0.0
    for window in range(numWindows):
        dxy = 0
        k = 8
        windowSize = int(numFrames/numWindows)
        diffX = np.zeros(windowSize, dtype=np.float64)
        diffY = np.zeros(windowSize, dtype=np.float64)
        tmpDiff = np.zeros(windowSize, dtype=np.float64)
        sortIndx = np.zeros(windowSize, dtype=np.int64)
        for step in range(window * 1024, (window + 1) * 1024):
            diffX.fill(0)
            diffY.fill(0)
            tmpDiff.fill(0)
            sortIndx.fill(0)
            for d in range(3):
                tmpDiff = np.abs(
                    x[d * numFrames:(d + 1) * numFrames] - x[step +
                                                             d * numFrames]
                )
                diffX = np.where(diffX > tmpDiff, diffX, tmpDiff)
                tmpDiff = np.abs(
                    y[d * numFrames:(d + 1) * numFrames] - y[step +
                                                             d * numFrames]
                )
                diffY = np.where(diffY > tmpDiff, diffY, tmpDiff)
            sortIndx = np.argsort(np.where(diffX > diffY, diffX, diffY))
            epsx = 0
            epsy = 0
            for kindx in range(1, k + 1):
                for d in range(3):
                    dist = np.abs(
                        x[d * numFrames + step] - x[
                            d * numFrames + sortIndx[kindx]]
                    )
                    if epsx < dist:
                        epsx = dist
                    dist = np.abs(
                        y[d * numFrames + step] - y[
                            d * numFrames + sortIndx[
                                kindx]]
                    )
                    if epsy < dist:
                        epsy = dist
            nx = len(np.nonzero(diffX <= epsx)[0]) - 1
            ny = len(np.nonzero(diffY <= epsy)[0]) - 1
            dxy += psi[nx] + psi[ny]
        dxy /= (numFrames / numWindows)
        mi += psi[int(numFrames/numWindows)] + phi[k] - dxy
    return mi / numWindows

for _ in range(100):
    numFrames = 1024
    x = np.random.random(3*numFrames)
    y = np.random.random(3*numFrames)
    np.save('x.npy', x)
    np.save('y.npy', y)
    corr = generalizedCorrelation(x, y, numFrames)
    corr = max(0, corr)
    corr = np.sqrt(1 - np.exp(-corr * (2.0 / 3)))
    output = list(map(
        lambda s: float(s),
        subprocess.Popen(
            ["./gpu_generalized_correlation_single_window_test"],
            stdout=subprocess.PIPE
        ).communicate()[0].decode('utf8').split('\n')[:-1]
    ))
    control = round(corr, 6)
    result = round(output[1], 6)
    print(control, result, abs(control - result))
