import os
import sys
import logging

import re
import numpy as np
from matplotlib import pyplot as plt

font = {'family' : 'Times New Roman', 'color'  : 'black',\
         'weight' : 'normal', 'size' : 18}

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    log_file = os.path.join('log', '1-layer lstm_0.5.bak.txt')
    data = []
    with open(log_file, 'r') as my_file:
        for line in my_file.readlines():
            if not line.startswith('Epoch') and not line.startswith('2017'):
                line_data = re.findall(r'\d+\.?\d*', line)
                data.append(line_data)
                # print(line_data)

    log_file_8 = os.path.join('log', '8-layer lstm_0.5.txt')
    data_8 = []
    with open(log_file_8, 'r') as my_file_8:
        for line in my_file_8.readlines():
            if not line.startswith('Epoch') and not line.startswith('2017'):
                line_data = re.findall(r'\d+\.?\d*', line)
                data_8.append(line_data)
                # print(line_data)

    x = np.linspace(1, 50, 50)
    # print(x)

    train_mae = []
    train_pearson = []

    for i in range(len(x)):
        train_mae.append(float(data[i][2]))
        train_pearson.append(float(data[i][3]))

    train_mae_8 = []
    train_pearson_8 = []

    for i in range(len(x)):
        train_mae_8.append(float(data_8[i][2]))
        train_pearson_8.append(float(data_8[i][3]))

    # print(train_mae)
    # print(train_pearson)

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(x, train_mae, linewidth=2.2, color='blue', label='1-layer LSTM ($MAE$)')
    line3, = ax1.plot(x, train_mae_8, linewidth=2.2, color='green', label='8-layer LSTM ($MAE$)')

    ax1.set_ylabel('Mean Absolute Error ($MAE$)', fontdict=font)
    ax1.set_title('Performance on Training Set', fontdict=font)
    ax1.set_xlabel('Epochs', fontdict=font)
    ax1.set_ylim(0.11, 0.21)

    ax2 = ax1.twinx()
    line2, = ax2.plot(x, train_pearson, linewidth=2.2, color='red', label='1-layer LSTM ($r$)')
    line4, = ax2.plot(x, train_pearson_8, linewidth=2.2, color='cyan', label='8-layer LSTM ($r$)')

    ax2.set_ylabel('Pearson correlation coefficient ($r$)', fontdict=font)
    ax2.set_ylim(0.3, 1.0)
    
    # ax1.legend(['1-layer LSTM'], \
    #     prop={'size': 16, 'family' : 'Times New Roman'}, loc='upper right')

    ax1.grid(True)

    plt.legend(handles=[line1, line2, line3, line4], prop={'size': 12, 'family' : 'Times New Roman'}, loc='upper right')

    plt.tight_layout()
    plt.show()


