import numpy as np

def __np_to_str(x):
    """
    Function that tranforms a numpy array to str

    Parameters
    ----------
        n : array 

    Returns
    -------
    xs  : str
        Returns a string with the following format:
            21.5,312.5,...,29.0
    """
 
    x = np.around(x, decimals=2)
    str_np = str(x)
    replace = ['\n', '[', ']']
    for r in replace:
        str_np = str_np.replace(r, '')
    xs = str_np.split()
    xs = ','.join(xs)
    return xs

def export(n, centroids, out='param.out'):
    """
    Function that writes the result of the learner script

    Parameters
    ----------
        n : int 
            A values that represents the number of centroids
        centroids: array
            An array containing the centroids 

    Returns
    -------
        side_effect : io
            Generates a file with the following format:
                2
                21.5,312.5,29.0
                225.75,1242.75,392.50
    """
    
    assert n == len(centroids)

    with open(out, 'w+') as f:
        lines = [str(n), '\n']
        for c in centroids:
            lines = lines + [__np_to_str(c), '\n']
        f.writelines(lines)
    f.close()

