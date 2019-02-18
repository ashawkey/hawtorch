import signal
import os
import time

def test_DelayedKeyboardInterrupt():
    from .utils import DelayedKeyboardInterrupt
    print("start")
    with DelayedKeyboardInterrupt():
        for i in range(5):
            print(i)
    print("finish")

