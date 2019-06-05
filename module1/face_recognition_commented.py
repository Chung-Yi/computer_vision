import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
video_capture = cv2.VideoCapture(0)


def detect(gray_img, frame):
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_grayimg = gray_img[y:y + h, x:x + w]
        image = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_grayimg, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex + x, ey + y), (ex + x + ew, ey + y + eh),
                          (0, 255, 0), 2)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return frame


def main():

    while True:
        _, frame = video_capture.read()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect(gray_img, frame)
        cv2.imshow('Video', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
