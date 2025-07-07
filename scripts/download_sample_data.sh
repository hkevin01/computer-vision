#!/bin/bash

echo "Downloading sample stereo images..."

# Create sample data directories
mkdir -p data/sample_images/left
mkdir -p data/sample_images/right
mkdir -p data/calibration

# Note: Add URLs to real stereo datasets here
# For now, create placeholder files
echo "Sample stereo image datasets:"
echo "1. Middlebury Stereo Dataset: https://vision.middlebury.edu/stereo/data/"
echo "2. KITTI Dataset: http://www.cvlibs.net/datasets/kitti/"
echo "3. ETH3D Dataset: https://www.eth3d.net/"

echo "Please download sample stereo images and place them in:"
echo "  data/sample_images/left/"
echo "  data/sample_images/right/"
echo "  data/calibration/"

# Create sample calibration checkerboard pattern
echo "Creating sample checkerboard pattern..."
python3 -c "
import cv2
import numpy as np

# Create 9x6 checkerboard pattern
board_w, board_h = 9, 6
square_size = 100  # pixels

img_w = board_w * square_size
img_h = board_h * square_size

# Create checkerboard
img = np.zeros((img_h, img_w), dtype=np.uint8)
for i in range(board_h):
    for j in range(board_w):
        if (i + j) % 2 == 0:
            y1, y2 = i * square_size, (i + 1) * square_size
            x1, x2 = j * square_size, (j + 1) * square_size
            img[y1:y2, x1:x2] = 255

cv2.imwrite('data/calibration/checkerboard_9x6.png', img)
print('Checkerboard pattern saved to data/calibration/checkerboard_9x6.png')
" 2>/dev/null || echo "Python3/OpenCV not available for checkerboard generation"

echo "Sample data setup completed!"
