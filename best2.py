import numpy as np
import cv2


def motion_detection(threshold=1200, video_name:str = 'video.mp4'):
    """
    Создание объекта фонового вычитания с помощью метода cv2.createBackgroundSubtractorMOG2().
        Этот метод позволяет выделять объекты, двигающиеся относительно статичного фона.
    Если кадр успешно считан, он масштабируется до размеров 600x500 пикселей 
        и применяется метод фонового вычитания fgbg.apply() к изображению для получения маски движения.
    Пороговая обработка маски для получения двоичного изображения с помощью cv2.threshold()
    Применение операций морфологического преобразования, таких как эрозия и расширение, 
        с целью удаления шумов и объединения разрывов в маске.
    Поиск контуров на обработанном изображении с помощью cv2.findContours(). 
        Контур считается обнаруженным, если его площадь превышает определенный порог threshold.
    """

    cap = cv2.VideoCapture(video_name + '.mp4')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    save_video_moving = cv2.VideoWriter(f'{video_name}_motion_detected_best2.avi',  
                        cv2.VideoWriter_fourcc('M','J','P','G'), 
                        10, size) 

    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

    while True:
        success, cur_frame = cap.read()

        # check if we get the frame
        if success == True:
            # img = cv2.resize(cur_frame, (600, 500))
            cv2.waitKey(30)
            fgmask = fgbg.apply(cur_frame)
            _, thresh = cv2.threshold(fgmask.copy(), 180, 255, cv2.THRESH_BINARY)
            # creating a kernel of 4*4
            kernel = np.ones((7, 7), np.uint8)
            # applying errosion to avoid any small motion in video
            thresh = cv2.erode(thresh, kernel)
            # dilating our image
            thresh = cv2.dilate(thresh, None, iterations=6)

            # finding the contours
            contours, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # finding area of contour
                area = cv2.contourArea(contour)
                print(area)
                # if area greater than the specified value the only then we will consider it
                if area > threshold:
                    # find the rectangle co-ordinates
                    x, y, w, h = cv2.boundingRect(contour)
                    # and then dra it to indicate the moving object
                    cv2.rectangle(cur_frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
                    cv2.putText(cur_frame, 'MOVING OBJECT DETECTED', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(cur_frame, 'MOVING OBJECT DETECTED', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    print('MOTION DETECTED')
                    save_video_moving.write(cur_frame)
            cv2.imshow('frame',cur_frame)
            cv2.imshow('frame2', thresh)
        else:
            break

    cap.release()
    save_video_moving.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    motion_detection(threshold=1200, video_name='video')
    # motion_detection(threshold=1200, video_name='panda')
    # motion_detection(threshold=1200, video_name='cars')
    # motion_detection(threshold=1200, video_name='2024')
