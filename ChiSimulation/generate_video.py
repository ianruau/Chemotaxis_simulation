import os
import cv2
import numpy as np
from tqdm import tqdm


def generate_chi_values(start, end):
    """Generate chi values from start to end (inclusive)"""
    return range(start, end + 1)


def create_video_from_images(image_folder, output_video, fps=30):
    # Generate all chi values from -10 to 164
    chi_values = generate_chi_values(-10, 164)
    total_frames = len(chi_values)

    # Get the first image to determine video dimensions
    first_chi = f"{chi_values[0]}-0"
    template = f"a=1-0_b=1-0_alpha=1-0_m=1-0_beta=1-0_chi={first_chi}_mu=1-0_nu=1_gamma=1-0_meshsize=50_time=5-0_epsilon=0-001_eigen_index=2.png"
    first_image_path = os.path.join(image_folder, template)

    if not os.path.exists(first_image_path):
        raise FileNotFoundError(f"First image not found: {first_image_path}")

    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'avc1' for H.264
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Process all images
    for chi in tqdm(chi_values, desc="Creating video"):
        chi_str = f"{chi}-0"
        image_name = f"a=1-0_b=1-0_alpha=1-0_m=1-0_beta=1-0_chi={chi_str}_mu=1-0_nu=1_gamma=1-0_meshsize=50_time=5-0_epsilon=0-001_eigen_index=2.png"
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image not found - {image_path}")
            continue

        img = cv2.imread(image_path)
        video.write(img)

    video.release()
    print(f"Video successfully created at: {output_video}")
    print(f"Total frames processed: {total_frames}")


# Parameters
image_folder = "."  # folder containing the PNG images
output_video = "output_video.mp4"
fps = 3  # frames per second

# Create the video
create_video_from_images(image_folder, output_video, fps)
