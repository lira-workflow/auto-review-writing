from typing import List
from itertools import islice


# This function is the same as in `utils/misc.py`
def chunk_list(lst: List[str], n: int) -> List:
    """
    Splits a list into chunks of size `n`.

    Parameters:
    lst - the list of strings.
    n - the chunk size.

    Returns:
    a list containing the chunked lists.
    """

    return [list(islice(lst, i, i + n)) for i in range(0, len(lst), n)]
