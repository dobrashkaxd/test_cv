import numpy as np
import cv2

# Загрузка каскада для обнаружения людей
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# cap = cv2.VideoCapture(0) # Использование камеры
# cap = cv2.VideoCapture('video.mp4')

cap = cv2.VideoCapture('2024.mp4')

while True:
    success, img = cap.read()

    if success:
        img = cv2.resize(img, (600, 500))
        cv2.waitKey(30)

        # Обнаружение людей
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        humans = human_cascade.detectMultiScale(gray, 1.1, 4)

        # Отрисовка прямоугольников вокруг обнаруженных людей
        for (x, y, w, h) in humans:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, 'Human', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('frame', img)
    else:
        break

cap.release()
cv2.destroyAllWindows()
