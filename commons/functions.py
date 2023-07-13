import urllib, os
from tqdm import tqdm
from pathlib import Path
urllib = getattr(urllib, 'request', urllib)

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)
    
def download_file(url, show_progress = True):
    import urllib.request
    initialize_folder()
    link = url
    file_name = link.split('/')[-1]
    if show_progress:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                desc=file_name) as t:  # all optional kwargs
            urllib.request.urlretrieve(link, filename=f"{get_model_path()}/{file_name}",
                        reporthook=t.update_to)
            t.total = t.n
    else:
         urllib.request.urlretrieve(link, filename=f"{get_model_path()}/{file_name}")
         
def get_home_path():
    """Get the home directory for storing weights and models.

    Returns:
        str: the home directory.
    """
    return str(os.getenv("HOME_PATH", default=str(Path.home())))

def get_model_path():
    return f"{get_home_path()}/.facedetect/models"

def initialize_folder():
    """Initialize the folder for storing weights and models.

    Raises:
        OSError: if the folder cannot be created.
    """
    print("checking home directory...")
    home = get_home_path()
    faceDetectHomePath = home + "/.facedetect"
    modelsPath = faceDetectHomePath + "/models"
    print(faceDetectHomePath)
    print(modelsPath)

    if not os.path.exists(faceDetectHomePath):
        os.makedirs(faceDetectHomePath, exist_ok=True)
        print("Directory ", home, "/.facedetect created")

    if not os.path.exists(modelsPath):
        os.makedirs(modelsPath, exist_ok=True)
        print("Directory ", home, "/.facedetect/models created")