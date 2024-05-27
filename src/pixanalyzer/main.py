import datetime
import json
import os
import sys
import tempfile
import time
from io import BytesIO
from tkinter import Tk, filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PySimpleGUI as sg
import seaborn as sns
from tqdm import tqdm

# Import required libraries
# (Original comment was 'import libs')

sys.path.append("../pixanalyzer")
from pixanalyzer.libs import threshold as thresh


# Function to display a plot in the GUI
def draw_and_show_plot(window, fig, key):
    """Display a matplotlib plot in a PySimpleGUI window.

    Args:
        window (sg.Window): The PySimpleGUI window object.
        fig (matplotlib.figure.Figure): The matplotlib figure to display.
        key (str): The key associated with the image element in the PySimpleGUI layout.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        tmp_filename = tmpfile.name
    fig.savefig(tmp_filename, bbox_inches="tight", pad_inches=0.1, dpi=100)
    window[key].update(filename=tmp_filename)


# Function to apply thresholding to an image
def apply_threshold(image, lower_threshold, upper_threshold):
    """Apply thresholding to an image.

    Args:
        image (np.array): The input image.
        lower_threshold (int): The lower threshold value.
        upper_threshold (int): The upper threshold value.

    Returns:
        np.array: The thresholded image.
    """
    mask = cv2.inRange(image, lower_threshold, upper_threshold)
    threshold_image = cv2.bitwise_and(image, image, mask=mask)
    return threshold_image


# Main function of the script
def main():
    """Main function to run the video analysis GUI."""
    layout = [
        [sg.Button("Start Analysis")],
        [
            sg.Text("Remaining time"),
            sg.ProgressBar(
                max_value=100, orientation="h", size=(30, 20), key="-PROGRESS-"
            ),
            sg.Text("", key="-TIME-", size=(30, 1)),
        ],
        [sg.Image(filename="", key="-IMAGE1-")],
        [sg.Button("Exit")],
    ]

    window = sg.Window("PixAnalyzer", layout, resizable=True)

    while True:
        event, values = window.read(timeout=10)  # Poll every 10 ms

        if event == sg.WIN_CLOSED or event == "Exit":
            break

        if event == "Start Analysis":
            # File dialog to select a video file
            root = Tk()
            root.withdraw()
            video_path = filedialog.askopenfilename(
                title="Select a video file", filetypes=[("Video files", "*.mp4 *.avi")]
            )

            if not video_path:
                print("No file selected.")
                sys.exit()
            print("Start!!!!")

            # Load the video file
            cap = cv2.VideoCapture(video_path)

            # Read the first frame
            ret, first_frame = cap.read()
            if not ret:
                print("Cannot read video file")
                cap.release()
                window.close()
                exit()

            # Convert to grayscale
            first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

            # GUI to select ROI (Region of Interest)
            roi = cv2.selectROI(
                "Select ROI", first_frame, fromCenter=False, showCrosshair=True
            )
            cv2.destroyAllWindows()

            if not roi[2] and not roi[3]:  # Check if ROI has been selected
                print("No ROI selected.")
                cap.release()
                continue

            # Crop the selected region from the first frame
            first_frame_cropped = first_frame_gray[
                roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]
            ]
            roi_draw_image = first_frame.copy()
            rectangle1 = (int(roi[0]), int(roi[1]))
            rectangle2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
            print("ROI", rectangle1, rectangle2)
            cv2.rectangle(
                roi_draw_image, rectangle1, rectangle2, (0, 0, 255), thickness=10
            )

            # Save the image with the selected ROI
            directory, file_name_with_extension = os.path.split(video_path)
            file_name, _ = os.path.splitext(file_name_with_extension)
            cropped_img_path = os.path.join(directory, f"{file_name}_croparea_draw.png")
            cv2.imwrite(cropped_img_path, roi_draw_image)

            # Select the brightness threshold
            lower_th, upper_th = thresh.select_threshold(first_frame_cropped)

            # Apply the threshold and save the result
            th_cropped = apply_threshold(first_frame_cropped, lower_th, upper_th)
            cropped_th_img_path = os.path.join(
                directory, f"{file_name}_crop_threshold.png"
            )
            cv2.imwrite(cropped_th_img_path, th_cropped)

            # Initialize a list to store sum of pixel intensities for each frame
            sum_intensities = []

            # Get properties of the video file
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Setup for output video with differences
            output_path = os.path.join(directory, f"{file_name}_difference_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_path, fourcc, fps, (roi[2], roi[3]), isColor=False
            )

            # Initialize variables for processing frames
            prev_frame = apply_threshold(first_frame_cropped, lower_th, upper_th)
            accumulated_diff = np.zeros_like(first_frame_cropped, dtype=np.float64)

            # Record start time for progress estimation
            start_time = time.time()

            for i in range(total_frames - 1):
                window["-PROGRESS-"].update_bar((i + 1) * 100 / total_frames)
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cropped = gray[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
                cropped = apply_threshold(cropped, lower_th, upper_th)
                diff = cv2.absdiff(cropped, prev_frame)
                accumulated_diff += diff.astype(np.float64)
                sum_intensity = np.sum(diff)
                sum_intensities.append(sum_intensity)
                out.write(diff)
                prev_frame = cropped

                # Update elapsed and remaining time
                elapsed_time = time.time() - start_time
                remaining_time = (total_frames - i) * (elapsed_time / (i + 1))
                window["-PROGRESS-"].update_bar((i + 1) * 100 / total_frames)
                window["-TIME-"].update(f"Left processing time: {remaining_time:.2f}s")

                time.sleep(0.001)

            cap.release()
            out.release()

            # Plot pixel intensity vs time
            range_frames = total_frames - 1  # Number of frames to plot
            frame_count = np.arange(0, range_frames)
            time_count = np.arange(1, range_frames + 1) / (
                fps * 60
            )  # Divide by 60 to convert seconds to minutes

            # Calculate baseline using 10th percentile of first 30 seconds
            sum0_percentile = 25
            sum0_window = 30
            sum0 = np.percentile(sum_intensities[: sum0_window * fps], sum0_percentile)

            # Generate fold change over baseline
            d_sum = (np.array(sum_intensities) - sum0) / sum0

            # 任意単位での差分ピクセル強度
            # Plot Delta Pixel Intensity (a.u)
            fig1 = plt.figure(figsize=(9, 2))
            plt.plot(time_count, sum_intensities[:range_frames])
            plt.xlabel("Time (min)")
            plt.ylabel("Delta Pixel Intensity (a.u)")
            plt.title(file_name + " - Delta Pixel Intensity (a.u)")
            # plt.grid()
            draw_and_show_plot(window, fig1, "-IMAGE1-")

            # 基準値に対する差分ピクセル強度の倍数
            # Plot Delta Pixel Intensity (fc)
            fig2 = plt.figure(figsize=(9, 2))
            plt.plot(time_count, d_sum[:range_frames])
            plt.xlabel("Time (min)")
            plt.ylabel("Delta Pixel Intensity (fc)")
            plt.title(file_name + " - Delta Pixel Intensity (fc)")
            # plt.grid()
            # draw_and_show_plot(fig2, '-IMAGE2-')

            # 基準値に対する差分ピクセル強度のパーセンテージ
            # Plot Delta Pixel Intensity (%)
            fig3 = plt.figure(figsize=(9, 2))
            plt.plot(time_count, d_sum[:range_frames] * 100)
            plt.xlabel("Time (min)")
            plt.ylabel("Delta Pixel Intensity (%)")
            plt.title(file_name + " - Delta Pixel Intensity (%)")
            # plt.grid()
            # draw_and_show_plot(fig3, '-IMAGE3-')

            # ヒートマップ
            fig4 = plt.figure(
                figsize=(roi[2] // 10, roi[3] // 10)
            )  # Adjust size as needed
            sns.heatmap(accumulated_diff, cmap="viridis")
            plt.axis("off")
            # draw_and_show_plot(window, fig4, '-IMAGE4-')

            # グラフ保存
            gp1_path = os.path.join(directory, f"{file_name}_au_graph.png")
            gp2_path = os.path.join(directory, f"{file_name}_fc_graph.png")
            gp3_path = os.path.join(directory, f"{file_name}_percent_graph.png")
            gp4_path = os.path.join(directory, f"{file_name}_heatmap.png")
            fig1.savefig(gp1_path)
            fig2.savefig(gp2_path)
            fig3.savefig(gp3_path)
            fig4.savefig(gp4_path)

            # Export data to CSV
            results_file_name = os.path.join(
                directory, f"{file_name}_pixel_intensities.csv"
            )

            # Create dataframe
            data = {
                "Frame": frame_count,
                "Time (min)": time_count,
                "Delta Pixel Intensity (a.u)": sum_intensities[:range_frames],
                "Delta Pixel Intensity (fc)": d_sum[:range_frames],
                "Delta Pixel Intensity (%)": d_sum[:range_frames] * 100,
            }
            df = pd.DataFrame(data)

            df.to_csv(results_file_name)

            # with pd.ExcelWriter(excel_file_name) as writer:
            #     df.to_excel(
            #         writer,
            #         index=False,
            #         sheet_name="Delta Pixel Intensity",
            #         startrow=0,
            #         startcol=0,
            #     )

            # 動画全体のCSVの保存
            csv_path = os.path.join(directory, f"{file_name}_accumulated_diff.csv")
            np.savetxt(csv_path, accumulated_diff, delimiter=",", fmt="%f")

            # Setting information
            now = datetime.datetime.now()
            set_info = {
                "ROI": (rectangle1, rectangle2),
                "Threshold": (lower_th, upper_th),
                "Video path": video_path,
                "Analysis date": now.strftime("%Y-%m-%d %H:%M:%S"),
            }
            result_json_path = os.path.join(directory, f"{file_name}_setting_info.json")
            # system infoの保存
            with open(result_json_path, "w") as f:
                json.dump(set_info, f, indent=4)

            print("Complete!!!!")

    window.close()


if __name__ == "__main__":
    print("Start!!!")
    main()
