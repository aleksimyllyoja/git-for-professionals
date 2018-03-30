import sys
import time
import logging

import git
import argparse

from generate import straight_flames

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ProfessionalSystemEventHandler(FileSystemEventHandler):
    def __init__(self, g):
        self.g = g

    def dispatch(self, event):
        message = straight_flames()
        print "> ", message
        if not test_run:
            try:
                self.g.add("*")
                self.g.commit(m=message)
                self.g.push(force=True)
            except:
                pass

if __name__ == "__main__":
    for i in range(10):
        print straight_flames()
    """
    parser = argparse.ArgumentParser(description="Are you a professional?")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dir', type=str)

    args = parser.parse_args()

    test_run = args.test
    path = args.dir
    g = git.cmd.Git(path)

    if test_run:
        print "test run"

    event_handler = ProfessionalSystemEventHandler(g)

    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
    """
