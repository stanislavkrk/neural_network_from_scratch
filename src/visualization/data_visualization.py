import matplotlib.pyplot as plt
import numpy as np

def data_input_visualization(data01, data02, data03):
    '''
    В цій функції візуалізуємо деякі літери, які були створені у функції data_input.
    Subplot приймає параметри: nrows, ncols, index
    imshow - data, cmap
    :param data01: Довільний літерал.
    :param data02: Довільний літерал.
    :param data03: Довільний літерал.
    :return: None
    '''
    plt.subplot(1, 3, 1)
    litera01 = np.array(data01).reshape(5,6)
    plt.imshow(litera01, cmap = 'coolwarm')
    plt.subplot(1, 3, 2)
    litera02 = np.array(data02).reshape(5,6)
    plt.imshow(litera02, cmap = 'coolwarm')
    plt.subplot(1, 3, 3)
    litera03 = np.array(data03).reshape(5,6)
    plt.imshow(litera03, cmap = 'coolwarm')
    plt.show()

def data_input_visualization_letters(data01, data02):
    '''
    В цій функції візуалізуємо деякі літери, в даному випадку порівнюємо очікуваний літерал та розпізнаний.
    Subplot приймає параметри: nrows, ncols, index
    imshow - data, cmap
    :param data01: Довільний літерал.
    :param data02: Довільний літерал.
    :return: None
    '''
    plt.subplot(1, 2, 1)
    litera01 = np.array(data01).reshape(5,6)
    plt.imshow(litera01, cmap = 'Blues')
    plt.title("Очікування:", fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    litera02 = np.array(data02).reshape(5,6)
    plt.imshow(litera02, cmap = 'Greens')
    plt.title("Класифіковано нейромережею:", fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.show()