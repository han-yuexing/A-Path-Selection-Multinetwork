import matplotlib.pyplot as plt
import os
import numpy as np

def savePlot(opt, valAccHist, lossHist, saveDirPath, netName, epochs):
    plt.title("Accurancy")
    plt.xlabel("Epoches")
    plt.ylabel("Validation Accurancy")
    plt.plot(range(1, epochs + 1), valAccHist)
    plt.ylim((0, 1.))
    plt.xticks((np.arange(1, epochs + 1, 1.)))
    plt.legend()
    plotName = netName + 'Acc.png'
    plt.savefig(os.path.join(saveDirPath, plotName))
    plt.show()
    plt.close()

    plt.title("Loss")
    plt.xlabel("Epoches")
    plt.ylabel("Validation Loss")
    plt.plot(range(1, epochs + 1), lossHist)
    plt.ylim((0, 3.))
    plt.xticks((np.arange(1, epochs + 1, 1.)))
    plt.legend()
    plt.savefig(os.path.join(saveDirPath, 'beginNetLoss.png'))
    plt.show()
    plt.close()

def savetxt(saveDirPath, netName, valAccHist, lossHist):
    txtName = netName + '.txt'
    with open(os.path.join(saveDirPath, txtName), "w", encoding='utf-8') as file:
        file.write("Acc: \n")
        for hist in valAccHist:
            file.write(str(hist))
            file.write('\n')
        file.write("Loss: \n")
        for loss in lossHist:
            file.write(str(loss))
            file.write('\n')
        file.close()