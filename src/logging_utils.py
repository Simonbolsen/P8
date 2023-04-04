import logging

def setup_logger(level):
    root = logging.getLogger('')
    root.setLevel(level)
    
    return root
    