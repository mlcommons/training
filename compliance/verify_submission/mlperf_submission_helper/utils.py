import os


def get_subdirs(parent_path):
  """Return a list of (name, path) tuples of direct subdirectories of

    parent_path, where each tuple corresponds to one subdirectory. Files
    in the parent_path are excluded from the output.
    """
  entries = os.listdir(parent_path)
  subdirs = [(entry, os.path.join(parent_path, entry))
             for entry in entries
             if os.path.isdir(entry)]
  return subdirs
