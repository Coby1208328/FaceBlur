import cv2

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load your photo (change this filename to match your image)
image = cv2.imread('your_photo.jpg')

# Convert to grayscale (face detection works on grayscale)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

print(f"Found {len(faces)} face(s)")

# Blur each detected face
for (x, y, w, h) in faces:
    face_region = image[y:y+h, x:x+w]
    blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
    image[y:y+h, x:x+w] = blurred_face

# Save the result
cv2.imwrite('blurred_output.jpg', image)
print("Saved as blurred_output.jpg")
