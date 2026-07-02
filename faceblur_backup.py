import cv2
import os

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load your photo
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, "WIN_20260701_20_02_56_Pro.jpg")
image = cv2.imread(filename)

if image is None:
    print(f"Could not open '{filename}'")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5
)

# Sort faces from left to right so numbering is consistent
faces = sorted(faces, key=lambda f: f[0])

print(f"Found {len(faces)} face(s)")

if len(faces) == 0:
    print("No faces detected.")
    exit()

# Create preview with numbered boxes
preview = image.copy()

for i, (x, y, w, h) in enumerate(faces):
    x, y, w, h = int(x), int(y), int(w), int(h)

    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        preview,
        str(i),
        (x, y - 10 if y > 20 else y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

cv2.imwrite("faces_preview.jpg", preview)
print("Saved faces_preview.jpg")

# Ask which faces to blur
choice = input(
    "Which face(s) do you want to blur? "
    "Enter numbers separated by commas (example: 0,2) or 'all': "
).strip()

if choice.lower() == "all":
    selected = list(range(len(faces)))
else:
    selected = []
    for item in choice.split(","):
        item = item.strip()
        if item.isdigit():
            num = int(item)
            if 0 <= num < len(faces):
                selected.append(num)

print("Selected faces:", selected)

# Blur only selected faces
for i in selected:

    x, y, w, h = faces[i]

    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    face_region = image[y:y+h, x:x+w]

    blurred = cv2.GaussianBlur(face_region, (99, 99), 30)

    image[y:y+h, x:x+w] = blurred

# Save result
cv2.imwrite("blurred_output.jpg", image)

print(f"Blurred {len(selected)} face(s).")
print("Saved as blurred_output.jpg")