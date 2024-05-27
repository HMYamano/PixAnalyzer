import tkinter as tk
from tkinter import filedialog

import cv2
import PySimpleGUI as sg


def get_file_path():
    """Opens a file dialog to select an image file and returns its path.

    Returns:
        str: Path of the selected file.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path


def apply_threshold(image, lower_threshold, upper_threshold):
    """Applies thresholding to an image using the specified lower and upper thresholds.

    Args:
        image (np.array): The input image.
        lower_threshold (int): The lower threshold value.
        upper_threshold (int): The upper threshold value.

    Returns:
        np.array: The image after thresholding.
    """
    mask = cv2.inRange(image, lower_threshold, upper_threshold)
    threshold_image = cv2.bitwise_and(image, image, mask=mask)
    return threshold_image


def resize_image(image, height=480):
    """Resizes an image to a specified height while maintaining the aspect ratio.

    Args:
        image (np.array): The input image.
        height (int, optional): The desired height. Defaults to 480.

    Returns:
        np.array: The resized image.
    """
    aspect_ratio = image.shape[1] / image.shape[0]
    width = int(height * aspect_ratio)
    resized = cv2.resize(image, (width, height))
    return resized


def to_bytes(image):
    """Converts an image to bytes for display in GUI.

    Args:
        image (np.array): The input image.

    Returns:
        bytes: The image in byte format.
    """
    is_success, buf = cv2.imencode(".png", image)
    return buf.tobytes()


def select_threshold(image):
    """Creates a GUI for applying threshold to an image and returns the chosen thresholds.

    Args:
        image (np.array): The input image.

    Returns:
        tuple: A tuple containing the lower and upper threshold values.
    """
    processed_image = None
    original_image = resize_image(image)

    layout = [
        [sg.Image(data=to_bytes(original_image), key="-IMAGE-")],
        [
            sg.Text("Lower threshold:"),
            sg.Slider(
                (0, 255),
                key="-LOWER THRESHOLD-",
                orientation="h",
                size=(40, 15),
                default_value=100,
            ),
        ],
        [
            sg.Text("Upper threshold:"),
            sg.Slider(
                (0, 255),
                key="-UPPER THRESHOLD-",
                orientation="h",
                size=(40, 15),
                default_value=200,
            ),
        ],
        [sg.Button("Show Processed"), sg.Button("Show Original")],
        [sg.Button("Exit")],
    ]

    window = sg.Window("Threshold Application", layout, resizable=True)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "Exit":
            break

        elif event == "Show Processed" and image.any():
            processed_image = apply_threshold(
                image,
                int(values["-LOWER THRESHOLD-"]),
                int(values["-UPPER THRESHOLD-"]),
            )
            processed_image = resize_image(processed_image)
            window["-IMAGE-"].update(data=to_bytes(processed_image))

        elif event == "Show Original" and original_image is not None:
            window["-IMAGE-"].update(data=to_bytes(original_image))

    window.close()
    return int(values["-LOWER THRESHOLD-"]), int(values["-UPPER THRESHOLD-"])


if __name__ == "__main__":
    image_path = get_file_path()
    image = cv2.imread(image_path)
    lowth, upth = select_threshold(image)
    print()
