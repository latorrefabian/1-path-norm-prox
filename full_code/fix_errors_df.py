import pandas
import os

from defaults import CV_ERRORS_DATA_FILE


def main():
    errors = pandas.read_pickle(CV_ERRORS_DATA_FILE)
    errors['file'] = errors['file'].map(lambda x: os.path.split(x)[-1])
    errors.to_pickle(CV_ERRORS_DATA_FILE)


if __name__ == '__main__':
    main()

