"""Functions for logger."""
import getpass
import pprint
import os
import time
import logging
import sys


def print_init_log_info(logger, input_args=None, username=True, sep_sign='-'):
    """Print initial information to logger."""
    logger.info('INITIALIZATION --- {}'.format(logger.name))

    if username:
        logger.info('Launched by {}'.format(getpass.getuser()))

    if input_args is not None:
        input_args_to_print = 'INPUT ARGS:\n'
        input_args_to_print += sep_sign * 80 + '\n'
        input_args_to_print += pprint.pformat(input_args) + '\n'
        input_args_to_print += sep_sign * 80
        logger.info(input_args_to_print)


def finish_logger(logger, sep_sign='#', print_done=True):
    """Print DONE, close logger and remove all handlers at the end."""
    if print_done:
        logger.info('DONE\n' + sep_sign * 80)
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def init_logger(logger_name, logpath, logging_stdout=True, time_UTC=True,
                log_format=('[%(asctime)s] {%(filename)s:%(lineno)d}'
                            ' %(levelname)s - %(message)s'),
                chmod_access=0o777):
    """
    Initialize logging configs.

    Parameters
    ----------
    logger_name: str, optional
        Logger name, which is also printed in initialization
    logpath: str, optional
        System path for log
    logging_stdout: bool, optional
        Use additional stdout logger in addition to file Handler
    time_UTC: bool, optional
        Use UTC or system time in logging
    log_format: str, optional
        Format for logging messages
    chmod_access: oct, optional
        Chmod access to created logpath if it doesn't exist

    """
    if not os.path.exists(os.path.dirname(logpath)):
        os.makedirs(os.path.dirname(logpath))

    if not os.path.exists(logpath):
        open(logpath, 'w').close()
        os.chmod(logpath, chmod_access)

    logger = logging.getLogger(logger_name)

    # check to exclude multiple handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # create a file handler
        handler = logging.FileHandler(logpath)
        handler.setLevel(logging.INFO)
        # create a logging format
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(handler)

        if logging_stdout:
            log_stdout = logging.StreamHandler(sys.stdout)
            log_stdout.setLevel(logging.INFO)
            formatter_stdout = logging.Formatter(log_format)
            log_stdout.setFormatter(formatter_stdout)
            logger.addHandler(log_stdout)

        if time_UTC:
            logging.Formatter.converter = time.gmtime

    return logger
