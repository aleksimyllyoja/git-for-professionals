import sys
import time
import logging

import git

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

path = sys.argv[1] if len(sys.argv) > 1 else '.'

g = git.cmd.Git(path)

class ProfessionalSystemEventHandler(FileSystemEventHandler):
    def dispatch(self, event):
        try:
            g.add("*")
            g.commit(m="asd")
            g.push(force=True)
        except:
            pass

event_handler = ProfessionalSystemEventHandler()

observer = Observer()
observer.schedule(event_handler, path, recursive=True)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
