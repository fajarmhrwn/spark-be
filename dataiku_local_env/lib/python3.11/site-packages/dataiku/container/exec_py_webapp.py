# encoding: utf-8
"""
Executor for containerized execution of python webapp.
"""

from dataiku.base.utils import ErrorMonitoringWrapper
from .runner import setup_log, setup_os_environ

if __name__ == "__main__":
    setup_log()
    setup_os_environ()

    with ErrorMonitoringWrapper():
        with open("start_webapp.py") as fd:
            exec(fd.read())