import os
import sys
import signal
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders

from .io import load_json

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


class EmailSender(object):
    def __init__(self, username=None, password=None, send_from=None, send_to=None, port=None, server=None):
        self.load_default()
        if username is not None: self.username = username 
        if password is not None: self.password = password
        if send_from is not None: self.send_from = username
        if send_to is not None: self.send_to = send_to
        if port is not None: self.port = port
        if server is not None: self.server = server

    def load_default(self, path="hawtorch/email.json"):
        args = load_json(path)
        self.username = args["username"]
        self.password = args["password"]
        self.sent_from = args["send_from"]
        self.sent_to = args["send_to"]
        self.server = args["server"]
        self.port = args["port"]

    
    def send(self, files=[], subject="[Model] Report", message="", use_tls=True):
        msg = MIMEMultipart()
        msg['From'] = self.send_from
        msg['To'] = COMMASPACE.join(self.send_to)
        msg['Date'] = formatdate(localtime=True)
        msg['Subject'] = subject

        msg.attach(MIMEText(message))

        for path in files:
            part = MIMEBase('application', "octet-stream")
            with open(path, 'rb') as file:
                part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition',
                            'attachment; filename="{}"'.format(os.path.basename(path)))
            msg.attach(part)

        smtp = smtplib.SMTP(self.server, self.port)
        if use_tls:
            smtp.starttls()
        smtp.login(self.username, self.password)
        smtp.sendmail(self.send_from, self.send_to, msg.as_string())
        smtp.quit()