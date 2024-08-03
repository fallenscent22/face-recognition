from pathlib import Path
import pickle
import face_recognition
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

# Ensure required directories exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def display_image_with_faces_and_names(image_path, face_locations, face_names):
    image = face_recognition.load_image_file(image_path)
    plt.imshow(image)
    ax = plt.gca()

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(left, top - 10, name, color='red', fontsize=12, weight='bold')

    plt.title(f"Faces found in {image_path.name}")
    plt.axis('off')
    plt.show()

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    training_path = Path("training")

    # Print contents of the training directory
    print(f"Contents of {training_path}:")
    for item in training_path.iterdir():
        print(item)

    # Check if training directory has subdirectories
    if not any(training_path.iterdir()):
        print("The training directory is empty or has no subdirectories.")
        return

    for filepath in training_path.glob("*"):
        print(f"Processing file: {filepath}")
        name = filepath.parent.name
        
        try:
            image = face_recognition.load_image_file(filepath)
            print(f"Loaded image size: {image.shape}")
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            continue

        face_locations = face_recognition.face_locations(image, model=model)
        print(f"Found {len(face_locations)} face(s) in {filepath}")

        face_encodings = face_recognition.face_encodings(image, face_locations)
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    if names and encodings:
        name_encodings = {"names": names, "encodings": encodings}
        with encodings_location.open(mode="wb") as f:
            pickle.dump(name_encodings, f)
        print("Encodings saved successfully.")
    else:
        print("No encodings to save.")

def recognize_faces_in_validation_images(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    try:
        with open(encodings_location, "rb") as f:
            name_encodings = pickle.load(f)

        known_names = name_encodings["names"]
        known_encodings = name_encodings["encodings"]

        validation_path = Path("validation")
        print(f"Contents of {validation_path}:")
        for item in validation_path.iterdir():
            print(item)

        for filepath in validation_path.glob("*"):
            print(f"Processing file: {filepath}")

            try:
                image = face_recognition.load_image_file(filepath)
                print(f"Loaded image size: {image.shape}")
            except Exception as e:
                print(f"Error loading image {filepath}: {e}")
                continue

            face_locations = face_recognition.face_locations(image, model=model)
            print(f"Found {len(face_locations)} face(s) in {filepath}")

            face_encodings = face_recognition.face_encodings(image, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

                face_names.append(name)

            display_image_with_faces_and_names(filepath, face_locations, face_names)

    except FileNotFoundError:
        print(f"File {encodings_location} not found. Please ensure the encoding process was successful.")
    except Exception as e:
        print(f"An error occurred while recognizing faces: {e}")

# Encode known faces and save to encodings.pkl
encode_known_faces()

# Recognize faces in validation images and display the results
recognize_faces_in_validation_images()