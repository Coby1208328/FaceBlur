Face Blurring Security Algorithm

An algorithm that is able to detect faces in images and then blur the face(s) that you would like to blur.


The Algorithm
Setup & Usage
Before running the script, you must install the required dependency:
pip install opencv-python

Follow these steps to use the project:
Place your photo in the same directory as the faceblur.py script.
Open the script and update the filename variable with your image's name.
Run the script using python faceblur.py.
Open faces_preview.jpg to identify which face numbers you wish to blur.
Enter the numbers in the terminal to generate your final blurred_output.jpg.
Project Code
import cv2
import os

def detect_faces(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return sorted(faces, key=lambda f: f[0])

def create_preview(image, faces):
    preview = image.copy()
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(preview, str(i), (x, max(y - 10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imwrite('faces_preview.jpg', preview)

def apply_blur(image, faces, selected_indices):
    for i in selected_indices:
        if 0 <= i < len(faces):
            x, y, w, h = map(int, faces[i])
            padding_x, padding_top, padding_bottom = int(w * 0.15), int(h * 0.35), int(h * 0.15)
            y1, y2 = max(0, y - padding_top), min(image.shape[0], y + h + padding_bottom)
            x1, x2 = max(0, x - padding_x), min(image.shape[1], x + w + padding_x)
            
            face_region = image[y1:y2, x1:x2]
            image[y1:y2, x1:x2] = cv2.GaussianBlur(face_region, (99, 99), 30)
    return image

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    filename = os.path.join(os.path.dirname(__file__), 'WIN_20260701_20_02_56_Pro.jpg')
    image = cv2.imread(filename)
    
    if image is None:
        print(f"Error: Could not open {filename}"); return
        
    faces = detect_faces(image, face_cascade)
    print(f"Found {len(faces)} face(s)")
    if not faces: return

    create_preview(image, faces)
    choice = input("Enter face numbers to blur (e.g. 0,2) or 'all': ").strip()
    selected = list(range(len(faces))) if choice.lower() == 'all' else [int(i) for i in choice.split(',') if i.strip().isdigit()]

    result = apply_blur(image, faces, selected)
    cv2.imwrite('blurred_output.jpg', result)
    print(f"Saved as blurred_output.jpg")

if __name__ == "__main__":
    main()

https://drive.google.com/file/d/1GRc2yupkjSQ08MR1x20uyfXCx24cgoG-/view
