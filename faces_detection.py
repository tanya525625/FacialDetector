import cv2
from matplotlib import pyplot as plt

import os


class FaceDetector:
    def __init__(self, template_path: str, target_image_path: str, methods: list):
        self.template = cv2.imread(template_path, 0)
        self.target_image = cv2.imread(target_image_path, 0)
        self.template_width, self.template_height = self.template.shape[::-1]
        self.methods = methods

    def find_faces(self):
        for method in self.methods:
            img = self.target_image.copy()
            curr_method = eval(method)
            res = cv2.matchTemplate(img, self.template, curr_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + self.template_width, top_left[1] + self.template_height)

            cv2.rectangle(img, top_left, bottom_right, 255, 2)
            _make_plot(img, curr_method)


def _make_plot(image, method_name):
    plt.imshow(image, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(method_name)
    plt.show()


if __name__ == "__main__":
    template_path = os.path.join('templates', 'template.jpg')
    image_path = os.path.join('target_images', 'image_2.jpg')
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    detector = FaceDetector(template_path, image_path, methods)
    detector.find_faces()
