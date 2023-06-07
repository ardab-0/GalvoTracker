#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include "circle_detector.h"
#include <chrono>

using namespace cv;
using namespace std;
int main(int, char**)
{



    Mat frame;
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    // open the default camera using default API
    // cap.open(0);
    // OR advance usage: select any API backend
    int deviceID = 1;             // 0 = open default camera
    int apiID = cv::CAP_ANY;      // 0 = autodetect default API
    // open selected camera using selected API
    cap.open(1);

    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    

    cv::CircleDetector::Context a(frame_size.width, frame_size.height);
    cv::CircleDetector detector(frame_size.width, frame_size.height, &a);
    

    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl;

    cv::CircleDetector::Circle circle;

    for (;;)
    {   
        
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        
        // check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        auto beg = std::chrono::high_resolution_clock::now();
        circle = detector.detect(frame, circle);
        auto end = std::chrono::high_resolution_clock::now();

        circle.draw(frame);
        // show live and wait for a key with timeout long enough to show images
        imshow("Live", frame);
        if (waitKey(1) >= 0)
            break;
        

        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);


        std::cout << "Elapsed time: " << duration.count() << std::endl;
            
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}