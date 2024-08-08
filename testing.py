import cv2
import numpy as np
image_path = "C:\\Users\\PC\\Desktop\\damage_assessement_data\\train\\masks\\guatemala-volcano_00000000_post_disaster.png"
img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
# Step 2: Print the unique pixel values in the mask
unique_values = np.unique(img)
print("Unique pixel values in the mask:", unique_values)

# Step 3: Check if specific values like 1, 2, 3, 4 are present
values_to_check = [1, 2, 3, 4]
for value in values_to_check:
    if value in unique_values:
        print(f"Value {value} is present in the mask.")
    else:
        print(f"Value {value} is NOT present in the mask.")