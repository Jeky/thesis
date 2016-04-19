import sys
import logging

_LOGGER_ = None
FORMAT = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')

def getLogger():
    global _LOGGER_
    if not _LOGGER_:
        _LOGGER_ = logging.getLogger()
        _LOGGER_.setLevel(logging.INFO)

        fileHandler = logging.FileHandler('log.txt')
        fileHandler.setFormatter(FORMAT)
        _LOGGER_.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(FORMAT)
        _LOGGER_.addHandler(consoleHandler)

    return _LOGGER_


def printFuncName(func):
    def _printFuncName(*argv, **kargw):
        globe.getLogger().info('Testing %s()', func.__name__)
        return func(*argv, **kargw)
    return _printFuncName


if __name__ == '__main__':
    getLogger().info('test')
