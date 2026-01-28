import numpy as np
from typing import Any, Callable


def apply_along_axis_multi(
    func: Callable[..., np.ndarray], axis: int, *arrays: np.ndarray, **kwargs: Any
) -> np.ndarray:
    r"""Apply a function to 1D slices of multiple arrays along a given axis,
    also slicing any kwarg‐arrays of matching shape.

    This is a multi‐array generalization of `np.apply_along_axis`.  In addition
    to the positional `arrays`, any keyword argument whose value is a NumPy
    array of the same shape as the inputs will be sliced along `axis`.  All
    other kwargs are forwarded intact.

    Args:
      func:
        A function that takes N one‐dimensional slices (one from each array,
        plus any sliced‐kwargs) and returns a one‐dimensional NumPy array.
      axis:
        Axis along which `func` is applied.
      *arrays:
        Input arrays.  All must have identical shapes.
      **kwargs:
        Keyword arguments for `func`.  If a kwarg value is an ndarray of the
        same shape as the inputs, it will be sliced along `axis`.  Otherwise
        it’s passed through untouched.

    Returns:
      An array with the same shape as the inputs except that the dimension
      along `axis` is replaced by the length of the 1D output of `func`.

    Raises:
      ValueError:
        If the input arrays do not all have identical shapes.
    """
    # 1) Validate that all positional arrays share the same shape
    shape = arrays[0].shape
    for arr in arrays:
        if arr.shape != shape:
            raise ValueError("All input arrays must have identical shapes.")

    # 2) Normalize axis index
    axis = np.core.numeric.normalize_axis_index(axis, arrays[0].ndim)

    # 3) Move the target axis to the front (axis=0) for easy iteration
    moved_arrays = [np.moveaxis(arr, axis, 0) for arr in arrays]
    n_slices = moved_arrays[0].shape[0]

    # 4) Partition kwargs into those to slice and those to leave static
    sliced_kwargs = {}
    static_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray) and v.shape == shape:
            # move the same axis to front
            sliced_kwargs[k] = np.moveaxis(v, axis, 0)
        else:
            static_kwargs[k] = v

    # 5) Run the first slice to discover output‐slice shape & dtype
    def _build_kwargs_for_slice(idx: int):
        d = {k: sliced_kwargs[k][idx] for k in sliced_kwargs}
        d.update(static_kwargs)
        return d

    first_out = func(
        *(moved_arrays[j][0] for j in range(len(moved_arrays))),
        **_build_kwargs_for_slice(0)
    )
    first_out = np.asarray(first_out)

    # 6) Allocate the full output array
    out_shape = (n_slices,) + first_out.shape
    out = np.empty(out_shape, dtype=first_out.dtype)
    out[0] = first_out

    # 7) Loop over the remaining slices
    for i in range(1, n_slices):
        args_i = [moved_arrays[j][i] for j in range(len(moved_arrays))]
        kw_i = _build_kwargs_for_slice(i)
        out[i] = func(*args_i, **kw_i)

    # 8) Move the slice-axis back to its original position
    return np.moveaxis(out, 0, axis)
