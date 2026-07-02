import cv2
import os
# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

import os

# Load your photo
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, 'WIN_20260701_20_02_56_Pro.jpg')
image = cv2.imread(filename)

if image is None:
    print(f"Could not open '{filename}' - check the filename and that it's in this folder")
    exit()

# Convert to grayscale (face detection works on grayscale)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
faces = sorted(faces, key=lambda f: f[0])  # left-to-right, so numbers match what you see
print(f"Found {len(faces)} face(s)")

if len(faces) == 0:
    print("No faces detected - try a clearer photo or different lighting")
    exit()

# Draw a numbered box on each face so you can see which number is which
preview = image.copy()
for i, (x, y, w, h) in enumerate(faces):
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(preview, str(i), (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

cv2.imwrite('faces_preview.jpg', preview)
print("Saved faces_preview.jpg - open it in VS Code to see which number is which face")

# Ask which face(s) to blur
choice = input("Which face(s) do you want to blur? Enter numbers separated by "
                "commas (e.g. 0,2), or 'all': ").strip()

if choice.lower() == 'all':
    selected = list(range(len(faces)))
else:
    selected = [int(i) for i in choice.split(',') if i.strip().isdigit()]

# Blur only the faces you chose
for i in selected:
    if 0 <= i < len(faces):
        x, y, w, h = faces[i]
        x, y, w, h = int(x), int(y), int(w), int(h)
        # Make the blur area a little larger
        padding_x = int(w * 0.15)
        padding_top = int(h * 0.35)
        padding_bottom = int(h * 0.15)

        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_top)
        x2 = min(image.shape[1], x + w + padding_x)
        y2 = min(image.shape[0], y + h + padding_bottom)

        face_region = image[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        image[y1:y2, x1:x2] = blurred_face
# Save the result
cv2.imwrite('blurred_output.jpg', image)
print(f"Blurred {len(selected)} face(s). Saved as blurred_output.jpg")