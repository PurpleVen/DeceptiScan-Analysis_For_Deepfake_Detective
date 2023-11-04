import cv2
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Input and output folder paths
input_folder = 'real_before_change'
output_folder = 'real_cropped'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Read the input image
        img = cv2.imread(os.path.join(input_folder, filename))

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Loop through detected faces and save them
        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y + h, x:x + w]

            # Resize the face to 256x256
            face = cv2.resize(face, (256, 256))

            # Convert pixel values to float and scale to [0, 1]
            face = face.astype(float) / 255.0

            # Save the cropped face to the output folder
            output_filename = f'{filename}_face_{i}.jpg'
            cv2.imwrite(os.path.join(output_folder, output_filename), face * 255)

print("Face detection and saving completed.")
