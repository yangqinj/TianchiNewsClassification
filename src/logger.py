"""
@author: Qinjuan Yang
@time: 2020-10-27 22:57
@desc: 
"""
import logging


_logger = None


def get_logger():
    global _logger
    if _logger:
        return _logger

    logging.basicConfig(format="%(levelname)s %(asctime)s %(filename)s %(lineno)d %(message)s")
    _logger = logging.getLogger("News Classification")
    _logger.setLevel(logging.INFO)
    return _logger
