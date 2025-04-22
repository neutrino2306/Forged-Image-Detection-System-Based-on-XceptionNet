import tensorflow as tf
import numpy as np
from tfwavelets.dwtcoeffs import get_wavelet, Wavelet
from tfwavelets.nodes import dwt1d


def dwt2d_my(input_node, wavelet, levels=1, mode='zero'):
    coeffs = {}
    pad_mode = 'CONSTANT' if mode == 'zero' else mode.upper()
    filter_length = len(wavelet.decomp_lp.coeffs.numpy())

    # Calculate padding based on the filter length
    pad_size = filter_length // 2
    paddings = tf.constant([ [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    # 不在批次和通道维度填充

    # Apply padding to the input
    if mode != 'periodization':
        padded_input = tf.pad(input_node, paddings, mode=pad_mode)
    else:
        # Periodization needs special handling, which we need to define separately
        padded_input = input_node  # This is a placeholder

    last_level = padded_input
    m, n, channels = last_level.shape

    for level in range(levels):
        local_m, local_n = m // (2 ** level), n // (2 ** level)
        print("Shape of last_level before passing to dwt1d:", last_level.shape)

        # Perform the first pass of the 1D DWT
        first_pass = dwt1d(last_level, wavelet, 1)  # assuming dwt1d accepts a Filter object

        # Perform the second pass on the transposed result of the first pass
        second_pass = tf.transpose(
            dwt1d(
                tf.transpose(first_pass, perm=[1, 0, 2]),
                wavelet,  # Assuming horizontal pass uses high pass filter
                1
            ),
            perm=[1, 0, 2]
        )

        # Slice out the low-frequency part for the next level
        last_level = tf.slice(second_pass, [0, 0, 0], [local_m // 2, local_n // 2, -1])

        # Store the high-frequency coefficients
        coeffs[f'level_{level + 1}'] = {
            'HL': tf.slice(second_pass, [0, local_n // 2, 0], [local_m // 2, local_n // 2, -1]),
            'LH': tf.slice(second_pass, [local_m // 2, 0, 0], [local_m // 2, local_n // 2, -1]),
            'HH': tf.slice(second_pass, [local_m // 2, local_n // 2, 0], [local_m // 2, local_n // 2, -1])
        }

    # Store the final low-frequency coefficients
    coeffs['low_freq'] = last_level

    return coeffs


def test_dwt2d_with_wavelet(wavelet_name, height=296, width=296, channels=3):
    """
    Create a specified wavelet and test the dwt2d function using a random input tensor.

    Args:
        wavelet_name (str): Name of the wavelet to use for testing ('haar', 'db1', 'db2', 'db3', 'db4').
        batch_size (int): Number of images in the batch.
        height (int): Height of each image.
        width (int): Width of each image.
        channels (int): Number of channels in each image (e.g., 3 for RGB).

    Returns:
        Dictionary: A dictionary containing the low-pass and high-pass coefficients from the DWT.
    """
    # Get wavelet from specified name
    wavelet = get_wavelet(wavelet_name)
    if wavelet is None:
        raise ValueError(f"Wavelet '{wavelet_name}' is not recognized or supported.")

    # Generate a random input tensor
    input_tensor = tf.random.normal([height, width, channels], dtype=tf.float32)

    # Apply the dwt2d_my function
    dwt_result = dwt2d_my(input_tensor, wavelet, levels=1, mode='zero')

    return dwt_result


if __name__ == "__main__":
    wavelet_name = 'haar'  # Change this as needed to 'db1', 'db2', 'db3', or 'db4'
    dwt_results = test_dwt2d_with_wavelet(wavelet_name)
    print("DWT results:", dwt_results)
