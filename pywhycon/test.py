from build.Release.circle_detector_module import *
import cv2
import time
circle_detector = CircleDetectorClass(640, 480)

prevCircle = CircleClass()
print(prevCircle)





# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    

    if not ret:
        continue
    

    start = time.time()
    new_circle = circle_detector.detect_np(frame, prevCircle)
    end = time.time()

    print("Elapsed time: ", end - start)

    prevCircle = new_circle

    image = cv2.circle(frame, (int(new_circle.x), int(new_circle.y)), radius=10, color=(0, 255, 0), thickness=2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()