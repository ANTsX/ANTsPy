__all__ = ['set_itk_default_num_threads', 'get_itk_default_num_threads']


from ants.internal import get_lib_fn


def set_itk_default_num_threads(num_threads: int) -> None:
    """
    Sets the default number of threads to be used when ants is calling ITK
    functions that support threading.

    The number of threads can be set via the environmental variable
    ``ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`` before ants is imported or via
    this function before the respective operation starts.

    Parameters
    ----------
    num_threads : int
        The number of threads to use by default.

    Examples
    --------
    >>> import ants
    >>> ants.get_itk_default_num_threads()
    20
    >>> ants.set_itk_default_num_threads(4)
    >>> ants.get_itk_default_num_threads()
    4
    """
    libfn = get_lib_fn('setGlobalDefaultNumberOfThreads')
    libfn(num_threads)


def get_itk_default_num_threads() -> int:
    """
    Gets the default number of threads ants is currently set to use when using
    ITK functions that support threaded operations.

    Returns
    -------
    num_threads : int
        The current number of default threads to use.

    Examples
    --------
    >>> import ants
    >>> ants.get_itk_default_num_threads()
    20
    """
    libfn = get_lib_fn('getGlobalDefaultNumberOfThreads')
    return libfn()
