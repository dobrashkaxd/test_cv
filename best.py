import pandas as pd
import cv2
from datetime import datetime


def motionDetection(threshold=10000, video_name:str = 'video'):
    """
    На каждом кадре вычисляется разница между текущим кадром и начальным состоянием 
        (выполняется абсолютное значение разницы).
    Она преобразуется в двоичное изображение, 
        чтобы выделить области, в которых есть значительные изменения.
    Далее на двоичном изображении ищутся контуры 
        с помощью функции cv2.findContours()
    Происходит итерация по каждому контуру, и если площадь контура больше определенного порога, 
        то он рассматривается как контур движущегося объекта.
    """
    # Assigning our initial state in the form of variable initialState as None for initial frames
    initialState = None

    # List of all the tracks when there is any detected of motion in the frames
    motionTrackList = [None, None]

    # A new list 'time' for storing the time when movement detected
    motionTime = []

    # Initialising DataFrame variable 'dataFrame' using pds libraries pd with Initial and Final column
    dataFrame = pd.DataFrame(columns=['Initial', 'Final'])

    # starting the webCam to capture the video using cv2 module

    # video = cv2.VideoCapture(0)
    # video = cv2.VideoCapture('cars.mp4')
    # video = cv2.VideoCapture('video.mp4')
    # video = cv2.VideoCapture('2024.mp4')
    video = cv2.VideoCapture(video_name + '.mp4')



    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)

    save_video_moving = cv2.VideoWriter(f'{video_name}_motion_detected_best.avi',  
                        cv2.VideoWriter_fourcc('M','J','P','G'), 
                        10, size) 

    # using infinite loop to capture the frames from the video
    while True:

        # Reading each image or frame from the video using read function
        check, cur_frame = video.read()

        # Defining 'motion' variable equal to zero as initial frame
        var_motion = 0

        # From colour images creating a gray frame
        if cur_frame is not None:
            gray_image = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            # To find the changes creating a GaussianBlur from the gray scale image
            gray_frame = cv2.GaussianBlur(gray_image, (21, 21), 0)
            # For the first iteration checking the condition
            # we will assign grayFrame to initalState if is none
            if initialState is None:
                initialState = gray_frame
                continue

            # Calculation of difference between static or initial and gray frame we created
            differ_frame = cv2.absdiff(initialState, gray_frame)
            # the change between static or initial background and current gray frame are highlighted
            thresh_frame = cv2.threshold(differ_frame, 30, 255, cv2.THRESH_BINARY)[
                1
            ]
            thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
            # For the moving object in the frame finding the coutours
            cont, _ = cv2.findContours(
                thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cur in cont:
                if cv2.contourArea(cur) < threshold:
                    continue
                var_motion = 1
                (cur_x, cur_y, cur_w, cur_h) = cv2.boundingRect(cur)
                # To create a rectangle of green color around the moving object
                cv2.rectangle(
                    cur_frame,
                    (cur_x, cur_y),
                    (cur_x + cur_w, cur_y + cur_h),
                    (0, 255, 0),
                    3,
                )
                cv2.putText(cur_frame, 'MOVING OBJECT DETECTED', (cur_x, cur_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                save_video_moving.write(cur_frame)
            # from the frame adding the motion status
            motionTrackList.append(var_motion)
            motionTrackList = motionTrackList[-2:]
            # Adding the Start time of the motion
            if motionTrackList[-1] == 1 and motionTrackList[-2] == 0:
                motionTime.append(datetime.now())
                save_video_moving.write(cur_frame)
            # Adding the End time of the motion
            if motionTrackList[-1] == 0 and motionTrackList[-2] == 1:
                motionTime.append(datetime.now())
                save_video_moving.write(cur_frame)
            # In the gray scale displaying the captured image
            cv2.imshow(
                'The image captured in the Gray Frame is shown below: ', gray_frame
            )
            # To display the difference between inital static frame and the current frame
            cv2.imshow(
                'Difference between the  inital static frame and the current frame: ',
                differ_frame,
            )
            # To display on the frame screen the black and white images from the video
            cv2.imshow(
                'Threshold Frame created from the PC or Laptop Webcam is: ',
                thresh_frame,
            )
            # Through the colour frame displaying the contour of the object
            cv2.imshow(
                'From the PC or Laptop webcam, this is one example of the Colour Frame:',
                cur_frame,
            )
            # Creating a key to wait
            wait_key = cv2.waitKey(1)
            # With the help of the 'm' key ending the whole process of our system
            if wait_key == ord('m'):
                # adding the motion variable value to motiontime list when something is moving on the screen
                if var_motion == 1:
                    motionTime.append(datetime.now())
                    save_video_moving.write(cur_frame)
                break
        else:
            break

    # At last we are adding the time of motion or var_motion inside the data frame
    for a in range(0, len(motionTime), 2):
        try:
            new_row = {'Initial': motionTime[a], 'Final': motionTime[a + 1]}
            dataFrame = pd.concat([dataFrame, pd.DataFrame([new_row])], ignore_index=True)
        except IndexError:
            break
        # dataFrame = dataFrame.append(
        #     {'Initial': time[a], 'Final': motionTime[a + 1]}, ignore_index=True
        # )


    # To record all the movements, creating a CSV file
    print('dataFrame ', dataFrame)
    dataFrame.to_csv('EachMovement.csv')

    # Releasing the video
    video.release()
    save_video_moving.release()
    # Now, Closing or destroying all the open windows with the help of openCV
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # motionDetection(threshold=10000, video_name = 'video')
    motionDetection(threshold=10000, video_name = 'video2')
    # motionDetection(threshold=10000, video_name = 'panda')
    # motionDetection(threshold=10000, video_name = 'cars')
    # motionDetection(threshold=5000, video_name = '2024')
