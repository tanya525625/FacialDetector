import cv2
import os


class FaceDetector:
    def __init__(self, image_path: str):
        self.target_image = cv2.imread(image_path)
        self.face_cascade = cv2.CascadeClassifier(os.path.join(cv2.haarcascades, "haarcascade_frontalface_default.xml"))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(cv2.haarcascades, "haarcascade_eye.xml"))
        self.gray_image = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)

    def find_faces(self):
        faces = self.face_cascade.detectMultiScale(self.gray_image,
                                                   scaleFactor=1.3,
                                                   minNeighbors=10)

        for (x, y, w, h) in faces:
            img = cv2.rectangle(self.target_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = self.gray_image[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=5, minNeighbors=5)
            eyes_count = 1
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 4)
                eyes_count += 1

        small = cv2.resize(self.target_image, (0,0), fx=0.4, fy=0.4)
        cv2.imshow('Photo', small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = os.path.join('target_images', 'image_6.jpg')

    detector = FaceDetector(image_path)
    detector.find_faces()
