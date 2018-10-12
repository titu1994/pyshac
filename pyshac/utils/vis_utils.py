import os
import matplotlib.pyplot as plt
import pyshac.config.data as data
plt.style.use('seaborn-paper')


def plot_dataset(dataset, to_file='dataset.png',
                 title='Dataset evaluation',
                 eval_label='Score'):
    """
    Plots the training history of the provided dataset.
    Can be provided either the `Dataset` object itself,

    # Arguments:
        dataset (Dataset | str): A Dataset which has been
            restored or a string path to the root of the
            shac directory where the dataset is stored.
        to_file (str | None): A string file name if the
            dataset plot is to be stored in a file, or
            `None` if the image should not be saved to
            a file.
        title (str): String label used as the title of
            the plot.
        eval_label (str): String label used as the y axis
            label of the plot.

    # Raises:
        FileNotFoundError: If the provided dataset is a string
            that does not point to the root of a shac
            directory.
        ValueError: If a loaded dataset object could not be
            obtained to visualize.
    """
    if dataset is None:
        dataset = 'shac'

    if title is None:
        title = ''

    if eval_label is None:
        eval_label = ''

    if type(dataset) == str:
        dataset = data.Dataset.load_from_directory(dataset)

    elif isinstance(dataset, data.Dataset):
        pass

    else:
        raise ValueError("Dataset provided must be a string path to the root of "
                         "the `shac` directory or a restored dataset object.")

    _, scores = dataset.get_dataset()

    if len(scores) < 1:
        raise ValueError("Dataset provided has no history. Please run "
                         "`restore_dataset` to restore the history of the "
                         "dataset first.")

    fig = plt.figure()
    plt.plot(scores, label='Evaluation score')
    plt.legend()
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(eval_label)
    plt.show()

    if type(to_file) == str:
        head, tail = os.path.split(to_file)

        if len(head) == 0:
            fig.savefig(to_file)
        else:
            if not os.path.exists(head):
                os.makedirs(head)
            fig.savefig(to_file)
