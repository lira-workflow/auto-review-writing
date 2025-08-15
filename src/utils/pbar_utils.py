# This script is adapted from: https://stackoverflow.com/a/74377458 (and is the same as in the `utils` folder)
# To make the progress bars more informative (otherwise it immediately returns 100%
# for all parallel jobs)
import joblib, contextlib
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """
    Context manager to patch joblib to report into a tqdm progress bar given as argument.

    Parameters:
    tqdm_object - the tqdm object to parallelize.
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object

    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
