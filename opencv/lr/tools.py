import logging
from time import time
import sys
import json

import logging


# todo at the moment the logger logs into the same file
#  and it just check if a console or file handler exists wihtout checking the caller.
#  Later We wantto check if a handler from a specific caller exists and whter it writes to console or the file

def setup_logger(log_file="app.log", log_to_console=True, log_to_file=True):
    # Create a logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)  # Set the logger's level

    # Check if a file handler already exists
    file_handler_exists = any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)

    # Create a formatter to format log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create and add a file handler only if it doesn't exist
    if not file_handler_exists and log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Check if a console handler already exists
    console_handler_exists = any(handler.__class__ == logging.StreamHandler for handler in logger.handlers)

    # Create and add a console handler only if it doesn't exist
    if not console_handler_exists and log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def lr_require_initialization(method):
    def wrapper(cls, *args, **kwargs):
        if not getattr(cls, '_initialized', False):
            raise ValueError(f"LR: {cls.__name__} Class not initialized. Call initialize_class first.")
        return method(cls, *args, **kwargs)

    return wrapper


def lr_error_logger(logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log = {'input': [str(a) for a in args]}
                logger.error(
                    f"LR-LOGGER:Error occurred:"
                    f"\nfunction: {func.__name__}"
                    f"\nerror: {str(e)}"
                    f"\ninput: {json.dumps(log)}"
                    f"\n ===========")
                raise e

        return wrapper

    return decorator


def lr_timer(logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time()
            _ = func(*args, **kwargs)
            duration = time() - start
            logger.info(f" function: {func.__name__} - took {duration:.10f} seconds to run.")
            return _

        return wrapper

    return decorator


if __name__ == '__main__':
    logger = setup_logger()


    class MyClass:

        @classmethod
        @lr_error_logger(logger)
        @lr_timer(logger)
        def divide(cls, a, b):
            # raise ValueError("asd")
            return a / b


    MyClass.divide(10, 3)
