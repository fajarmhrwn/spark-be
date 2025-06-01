import dataiku
import os
import threading
from dotenv import load_dotenv
load_dotenv()

class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class Dataiku(metaclass=SingletonMeta):
    client = None

    def __init__(self):
        # This check is technically redundant with the metaclass, but it's good practice to include it.
        if Dataiku.client is None:
            dataiku.set_remote_dss(
                "https://ai-playground.app.weu-d1.delfi.slb-ds.com",
                "dkuaps-JjKc0w8HW663QSZw11ZTYGthmHFkBxYS",
                no_check_certificate=True
            )
            self.client = dataiku.api_client()
        else:
            self.client = Dataiku.client  # Use the existing client
if __name__ == "__main__":
  client = Dataiku().client
  project = client.get_default_project()
  print(client.list_project_keys())
  for d in project.list_datasets():
    print(d.name)
