import numpy as np


def sort_func(filename):
    file_parts = filename.split('_')
    try:
        return int(file_parts[-1].split('.')[0])
    except ValueError:
        return 0


def shrink_array(arr, factor):
    """Re-bin an array by an integer factor.
    In the new array, each bin is the sum of the original bins.
    """
    old_shape = arr.shape
    new_shape = (old_shape[0] // factor, old_shape[1] // factor)
    new_arr = np.zeros(new_shape)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            new_arr[i, j] = np.sum(arr[i*factor:(i+1)*factor, j*factor:(j+1)*factor])
    return new_arr


def shrink_array_roll(arr, factor: int, offset_0: int = 0, offset_1: int = 0):
    """Offset the array by a given amount and then re-bin it with integer factor.
    """
    offset_0 = offset_0 % factor
    offset_1 = offset_1 % factor
    rolled_array = np.roll(arr, (offset_0, offset_1), axis=(0, 1))
    return shrink_array(rolled_array, factor)


def rhombus_diagonal_idx(arr_shape, diagonal, size, offset, index):
    """Return a boolean mask for a rhombus-shaped rebinned array.

    Args:
        arr_shape (tuple): Shape of the array.
        diagonal (int): Index of the diagonal.
        size (int): Size of the rhombus.
        offset (int): Offset of the zero-index rhombus center along the chosen diagonal.
        index (int): Index of the rhombus within an array of adjacent rhombuses.
            Each rhombus is centered along the diagonal, and index provides some offset.

    Return:
        np.ndarray: Boolean mask of rhombus
    """
    x_center = offset + (size * index) + diagonal
    y_center = offset + (size * index)

    mask = np.zeros(arr_shape)
    for i in range(-size+1, size):
        for j in range(-size+1, size):
            if abs(i) + abs(j) < size:
                mask[x_center+i, y_center+j] = 1

    mask = mask.astype(bool)
    return mask


def rhombus_diagonal(arr, diagonal, size, offset, num):
    diag_values = np.zeros(num)
    for i in range(num):
        mask = rhombus_diagonal_idx(arr.shape, diagonal, size, offset, i)
        diag_values[i] = np.sum(arr[mask])

    return diag_values


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    arr_shape = (20, 20)
    diagonal = 0
    size = 3
    offset = 5
    index = 1

    np.random.seed(0)
    bg_data = np.random.rand(*arr_shape)
    mask = rhombus_diagonal_idx(arr_shape, diagonal, size, offset, index)

    # plt.imshow(bg_data)
    plt.imshow(mask)
    plt.contour(mask, levels=[0.5], colors='r')
    plt.tight_layout()
    plt.show()
