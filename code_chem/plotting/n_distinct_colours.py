import math

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.colors import hsv_to_rgb


def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7

    number_of_distinct_colors_with_multiply_of_shades = int(
        math.ceil(number_of_distinct_colors / number_of_shades)
        * number_of_shades
    )

    # Uniform hue distribution
    linearly_distributed_nums = (
        np.arange(number_of_distinct_colors_with_multiply_of_shades)
        / number_of_distinct_colors_with_multiply_of_shades
    )

    # Create rising saw pattern
    arr_by_shade_rows = linearly_distributed_nums.reshape(
        number_of_shades,
        number_of_distinct_colors_with_multiply_of_shades // number_of_shades
    )

    arr_by_shade_columns = arr_by_shade_rows.T

    number_of_partitions = arr_by_shade_columns.shape[0]

    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # -------------------------------------------------------
    # CUSTOM HSV GENERATION INSTEAD OF matplotlib.cm.hsv
    # -------------------------------------------------------

    hues = nums_distributed_like_rising_saw

    # Slightly reduced saturation
    saturations = np.full_like(hues, 0.75)

    # Medium brightness only (avoid neon/light colors)
    values = np.full_like(hues, 0.75)

    # Darker shades for lower partitions
    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    lower_half = lower_partitions_half * number_of_shades

    values[:lower_half] *= np.linspace(0.55, 1.0, lower_half)

    # Slightly brighten upper half, but never too much
    values[lower_half:] *= np.linspace(
        0.85,
        1.0,
        len(values[lower_half:])
    )

    # -------------------------------------------------------
    # Reduce problematic yellow brightness
    # Yellow hue region ≈ 0.13 - 0.18
    # -------------------------------------------------------

    yellow_mask = (hues > 0.13) & (hues < 0.18)

    values[yellow_mask] *= 0.7
    saturations[yellow_mask] *= 0.6

    # Build HSV array
    hsv_colors = np.stack([hues, saturations, values], axis=1)

    rgb_colors = hsv_to_rgb(hsv_colors)

    # Add alpha channel
    alpha = np.ones((rgb_colors.shape[0], 1))
    rgba_colors = np.concatenate([rgb_colors, alpha], axis=1)

    return ListedColormap(rgba_colors)