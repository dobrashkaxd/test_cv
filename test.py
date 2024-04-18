# Importing the Pandas libraries
import pandas as pd

# Importing the OpenCV libraries
import cv2

# Importing the datetime function of the datetime module
from datetime import datetime, timedelta

# Assigning our initial state in the form of variable initialState as None for initial frames
initialState = None

# List of all the tracks when there is any detected of motion in the frames
motionTrackList = [None, None]

# A new list 'time' for storing the time when movement detected
motionTime = []

# Time interval threshold for stopping motion recording (20 minutes)
STOP_THRESHOLD = timedelta(minutes=20)

# Initialising DataFrame variable 'dataFrame' using pandas libraries panda with Initial and Final column
dataFrame = pd.DataFrame(columns=['Time'])

# starting the webCam to capture the video using cv2 module
# video = cv2.VideoCapture(0)
video = cv2.VideoCapture('video.mp4')

# Video writer object to save video with motion
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('motion_detected_video.avi', fourcc, 20.0, (640, 480))

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
        if initialState is None:
            initialState = gray_frame
            continue

        # Calculation of difference between static or initial and gray frame we created
        differ_frame = cv2.absdiff(initialState, gray_frame)

        # the change between static or initial background and current gray frame are highlighted
        thresh_frame = cv2.threshold(differ_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        # For the moving object in the frame finding the contours
        cont, _ = cv2.findContours(
            thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cur in cont:
            if cv2.contourArea(cur) < 10000:
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

        # from the frame adding the motion status
        motionTrackList.append(var_motion)
        motionTrackList = motionTrackList[-2:]

        # Adding the time when motion detected
        if motionTrackList[-1] == 1 and motionTrackList[-2] == 0:
            motionTime.append(datetime.now())

        # Stop recording if no motion detected for 20 minutes
        if motionTime and datetime.now() - motionTime[-1] > STOP_THRESHOLD:
            break

        # Writing to the video file when motion is detected
        out.write(cur_frame)

        # Creating a key to wait
        wait_key = cv2.waitKey(1)

        # With the help of the 'm' key ending the whole process of our system
        if wait_key == ord('m'):
            # adding the motion variable value to motiontime list when something is moving on the screen
            if var_motion == 1:
                motionTime.append(datetime.now())
            break
    else:
        break

# Convert motionTime list to DataFrame
motionTime_df = pd.DataFrame({'Time': motionTime})

# Concatenate motionTime_df with dataFrame
dataFrame = pd.concat([dataFrame, motionTime_df], ignore_index=True)

# To record all the movements, creating a CSV file
dataFrame.to_csv('EachMovement.csv', index=False)

# Release the video writer and capture objects
out.release()
video.release()

# Now, Closing or destroying all the open windows with the help of openCV
cv2.destroyAllWindows()
