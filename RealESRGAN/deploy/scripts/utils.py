import yaml

def load_config(filepath):
    """Load yaml file.

    Args:
        filepath(str): Path to YAML config file.

    Returns:
        dict: config dict.
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)