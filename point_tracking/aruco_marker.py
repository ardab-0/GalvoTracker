import cv2 
import numpy as np
import time
# Load the predefined dictionary
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


# # Generate the marker
# markerImage = np.zeros((200, 200), dtype=np.uint8)
# markerImage = dictionary.generateImageMarker(33, 200, markerImage, 1)
# cv.imwrite("marker33.png", markerImage)



parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
vid = cv2.VideoCapture(0)
  
while(True):
    start = time.time()
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
    print(markerCorners)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    print(time.time() - start)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break







# #Load the dictionary that was used to generate the markers.
# dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
 
# # Initialize the detector parameters using default values
# parameters =  cv.aruco.DetectorParameters_create()
 
# # Detect the markers in the image
# markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)
