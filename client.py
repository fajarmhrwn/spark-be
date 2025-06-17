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
    try:
        client = Dataiku().client
        project = client.get_default_project()
        print("=== Models ===")
        
        # Get list of all models in project
        models = project.list_saved_models()
        # for model in models:
            # print(f"- {model.get('name', 'N/A')} (Type: {model.get('type', 'N/A')})")
            # print(f"  ID: {model.get('id', 'N/A')}")
        
        # List ML tasks
        # print("\n=== ML Tasks ===")
        # ml_tasks = project.list_ml_tasks()
        # 1. Get the saved model metadata
        saved_model = project.get_saved_model("cDrOuPX1")

        # 2. Get list of versions (you must have at least one trained version)
        versions = saved_model.list_versions()

        if not versions:
            raise ValueError("No trained versions found for model 'cDrOuPX1'")

        # 3. Get the most recent version (or pick specific one)
        latest_version_id = versions[0]['id']  # or use logic to pick one
        trained_model = saved_model.get_version(latest_version_id)

        # 4. Get the predictor
        predictor = trained_model.get_predictor()

        # a = models
        # print(models[0].get("id", "mh8CKwbK"))
        # for task in ml_tasks:
            # print(f"- {task.get_name()}")
            
    except Exception as e:
        print(f"Error: {str(e)}")