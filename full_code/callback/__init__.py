# import pdb
import time

from collections import defaultdict


class Callback:
    """Callback for iterative optimization methods

    Args:
        fns (list): list of callable objects that will be executed when
            the callback function is invoked.
        kwfns (dict): dictionary of callable objects that will be executed
            when the callback function is invoked.

    Attributes:
        functions (dict):
        values (collections.defaultdict):
        time_offset (float):

    """
    def __init__(self, *fns, skip=1, **kwfns):
        self.functions = {}
        self.functions.update(kwfns)
        self.functions.update({f.__name__: f for f in fns})
        self.values = defaultdict(list)
        self.time_offset = 0.0
        self.iter = 0
        self.skip = skip

    @staticmethod
    def _var_msg(name, value):
        """float variable print formatting

        Args:
            name (str): name of the variable
            value (float): numerical value of the variable

        Returns:
            (str): name-value pair in string format

        """
        return ' {0}: {1:>7.3e}'.format(name, value)

    def __str__(self):
        """print all current numeric variables sorted by key"""
        msg = ''

        for k, v in sorted(self.values.items()):
            if k == 'time' or k == 'iter':
                continue
            if type(v[-1]) is float:
                msg += self._var_msg(k, v[-1])

        return msg

    def __call__(self, namespace):
        """Evaluate the functions and store the values"""
        if not self.iter % self.skip == 0:
            self.iter += 1
            return

        self.values['iter'].append(self.iter)
        self.iter += 1

        self.values['time'].append(time.process_time() - self.time_offset)
        start = time.process_time()

        for name, function in self.functions.items():
            value = function(**namespace)
            self.values[name].append(value)

        self.time_offset = time.process_time() - start

