"""Setup script for dlib facial landmark predictor model.

This module handles downloading and setting up the dlib shape predictor model
required for facial landmark detection.
"""

import os
import urllib.request
import bz2

def download_shape_predictor() -> None:
    """Download and extract the dlib shape predictor model if not present.

    Raises:
        RuntimeError: If download or extraction fails
    """
    predictor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                 'shape_predictor_68_face_landmarks.dat')
    
    if os.path.exists(predictor_path):
        return

    try:
        print("Downloading shape predictor model...")
        model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_path = predictor_path + ".bz2"

        # Download the compressed file
        urllib.request.urlretrieve(model_url, compressed_path)

        # Extract the file
        print("Extracting shape predictor model...")
        with bz2.BZ2File(compressed_path) as fr, open(predictor_path, 'wb') as fw:
            fw.write(fr.read())

        # Clean up the compressed file
        os.remove(compressed_path)
        print("Shape predictor model setup complete.")

    except Exception as e:
        raise RuntimeError(f"Failed to download shape predictor: {str(e)}")