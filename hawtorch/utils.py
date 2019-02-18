import os
import sys
import signal

class DelayedKeyboardInterrupt(object):
    """ Delayed SIGINT 
    reference: https://stackoverflow.com/questions/842557/how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py/21919644
    with statement: 
        * call __init__, instantiate context manager class with parameters.
        * call __enter__, return of __enter__ is assigned to variable after `as`.
        * run `with` body.
        * call __exit__(exc_type, exc_value, exc_traceback). If return false, then raise error, else omit.
        * call __del__
    signal.signal(sig, handler)
        set handler for sig, return the old handler(signal.SIG_DFL)
    """
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received: 
            self.old_handler(*self.signal_received)