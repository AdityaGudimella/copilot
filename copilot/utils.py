import json

from copilot import constants


def persist_str(key: str, value: str) -> None:
    """
    Persist a string value associated with a key to a JSON file.

    Args:
        key (str): The key to associate with the value.
        value (str): The string value to persist.

    Returns:
        None
    """
    file_path = constants.PERSISTENCE_SETTINGS_PATH
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if file exists
    if file_path.exists():
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Update or add the new key-value pair
    data[key] = value

    # Write the updated data back to the file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def retrieve_str(key: str) -> str:
    """
    Retrieve a string value associated with a key from a JSON file.

    Args:
        key (str): The key associated with the value to retrieve.

    Returns:
        Optional[str]: The retrieved string value, or None if the key is not found.
    """
    file_path = constants.PERSISTENCE_SETTINGS_PATH

    if not file_path.exists():
        raise ValueError(f"No settings file found at {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    try:
        return data[key]
    except KeyError:
        raise ValueError(f"No value found for key {key}")
