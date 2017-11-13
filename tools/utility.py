#coding: utf-8

import os,sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    null_fd = os.open(os.devnull, os.O_RDWR)
    save_fd = os.dup(1)
    os.dup2(null_fd, 1)
    yield
    os.dup2(save_fd, 1)
    os.close(null_fd)
    os.close(save_fd)
