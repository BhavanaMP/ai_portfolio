import os
import json


def save_temperature(temperature, file_path):
    """
    This function saves the posthoc trained temperature in a json file
    """
    with open(file_path, "w") as f:
        json.dump({"temperature": temperature}, f)


def load_temperature(file_path):
    """
    This function loads the trained temperature from the JSON file
    """
    if not os.path.exists(file_path):
        raise ValueError(f"use_tempscaling is set, but saved_temperature_path: {file_path} doesnt exist.")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data["temperature"]