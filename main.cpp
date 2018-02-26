#include <Eigen/Dense>
#include <Eigen/Core>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/aruco.hpp"
#include "opencv2/core/eigen.hpp"
#include <OpenGL/gl.h>
#include <OpenGl/glu.h>
#include <GLUT/glut.h>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <cmath>
#include <math.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <ctime>

#define EIGEN_RUNTIME_NO_MALLOC

using namespace std;
using namespace cv;
using namespace Eigen;

// >>>> Kalman Filter
int stateSize = 6;
int measSize = 4;
int contrSize = 0;

unsigned int type = CV_32F; // float type
/*
    This method creates a custom dictionary of aruco markers.
    This dictionary is used later to compare the markers on the camera to the dictionary
    The smaller the dictionary, the less computation time for comparing
*/
void createArucoMakers()
{
    Mat outputMarker3;
    Ptr<aruco::Dictionary> markerDictionary3 = aruco::Dictionary::create(6, 4);
    for (int i = 0; i < 10; i++)
    {
        aruco::drawMarker(markerDictionary3, i, 500, outputMarker3, 1);
        ostringstream convert;
        string imageName = "CubeMarker_";
        convert << imageName << i << ".jpg";
        imwrite(convert.str(), outputMarker3);
    }
}

int kalmanFilter(Mat toShow)
{
    // Camera frame

    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);
    
    cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
    cv::Mat meas(measSize, 1, type);    // [z_x,z_y,z_w,z_h]
    //cv::Mat procNoise(stateSize, 1, type)
    // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]
    
    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    cv::setIdentity(kf.transitionMatrix);
    
    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;
    
    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;
    
    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
    // <<<< Kalman Filter
    
    // Camera Index
    int idx = 1;
    
    // Camera Capture
    cv::VideoCapture cap;
    
    // >>>>> Camera Settings
    if (!cap.open(idx))
    {
        cout << "Webcam not connected.\n" << "Please verify\n";
        return EXIT_FAILURE;
    }
    
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 768);
    // <<<<< Camera Settings
    
    cout << "\nHit 'q' to exit...\n";
    
    char ch = 0;
    
    double ticks = 0;
    bool found = false;
    
    int notFoundCount = 0;
    
    // >>>>> Main loop
    while (ch != 'q' && ch != 'Q')
    {
        double precTick = ticks;
        ticks = (double) cv::getTickCount();
        
        double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds
        
        // Frame acquisition
        cap >> toShow;
        
        cv::Mat res;
        toShow.copyTo( res );
        
        if (found)
        {
            // >>>> Matrix A
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;
            // <<<< Matrix A
            
            cout << "dT:" << endl << dT << endl;
            
            state = kf.predict();
            cout << "State post:" << endl << state << endl;
            
            cv::Rect predRect;
            predRect.width = state.at<float>(4);
            predRect.height = state.at<float>(5);
            predRect.x = state.at<float>(0) - predRect.width / 2;
            predRect.y = state.at<float>(1) - predRect.height / 2;
            
            cv::Point center;
            center.x = state.at<float>(0);
            center.y = state.at<float>(1);
            cv::circle(res, center, 2, CV_RGB(255,0,0), -1);
            
            cv::rectangle(res, predRect, CV_RGB(255,0,0), 2);
        }
        
        // >>>>> Noise smoothing
        cv::Mat blur;
        cv::GaussianBlur(toShow, blur, cv::Size(5, 5), 3.0, 3.0);
        // <<<<< Noise smoothing
        
        // >>>>> HSV conversion
        cv::Mat frmHsv;
        cv::cvtColor(blur, frmHsv, CV_BGR2HSV);
        // <<<<< HSV conversion
        
        // >>>>> Color Thresholding
        // Note: change parameters for different colors
        cv::Mat rangeRes = cv::Mat::zeros(toShow.size(), CV_8UC1);
        cv::inRange(frmHsv, cv::Scalar(0,0,0,0),
                    cv::Scalar(180, 255, 30, 0), rangeRes);
        // <<<<< Color Thresholding
        
        // >>>>> Improving the result
        cv::erode(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        // <<<<< Improving the result
        
        // Thresholding viewing
        cv::imshow("Threshold", rangeRes);
        
        // >>>>> Contours detection
        vector<vector<cv::Point> > contours;
        cv::findContours(rangeRes, contours, CV_RETR_EXTERNAL,
                         CV_CHAIN_APPROX_NONE);
        // <<<<< Contours detection
        
        // >>>>> Filtering
        vector<vector<cv::Point> > balls;
        vector<cv::Rect> ballsBox;
        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::Rect bBox;
            bBox = cv::boundingRect(contours[i]);
            
            float ratio = (float) bBox.width / (float) bBox.height;
            if (ratio > 1.0f)
                ratio = 1.0f / ratio;
            
            // Searching for a bBox almost square
            if (ratio > 0.6 && bBox.area() >= 400)
            {
                balls.push_back(contours[i]);
                ballsBox.push_back(bBox);
            }
        }
        // <<<<< Filtering
        
        cout << "Balls found:" << ballsBox.size() << endl;
        
        // >>>>> Detection result
        for (size_t i = 0; i < balls.size(); i++)
        {
            cv::drawContours(res, balls, i, CV_RGB(20,150,20), 1);
            cv::rectangle(res, ballsBox[i], CV_RGB(0,255,0), 2);
            
            cv::Point center;
            center.x = ballsBox[i].x + ballsBox[i].width / 2;
            center.y = ballsBox[i].y + ballsBox[i].height / 2;
            cv::circle(res, center, 2, CV_RGB(20,150,20), -1);
            
            stringstream sstr;
            sstr << "(" << center.x << "," << center.y << ")";
            cv::putText(res, sstr.str(),
                        cv::Point(center.x + 3, center.y - 3),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20), 2);
        }
        // <<<<< Detection result
        
        // >>>>> Kalman Update
        if (balls.size() == 0)
        {
            notFoundCount++;
            cout << "notFoundCount:" << notFoundCount << endl;
            if( notFoundCount >= 100 )
            {
                found = false;
            }
            /*else
             kf.statePost = state;*/
        }
        else
        {
            notFoundCount = 0;
            
            meas.at<float>(0) = ballsBox[0].x + ballsBox[0].width / 2;
            meas.at<float>(1) = ballsBox[0].y + ballsBox[0].height / 2;
            meas.at<float>(2) = (float)ballsBox[0].width;
            meas.at<float>(3) = (float)ballsBox[0].height;
            
            if (!found) // First detection!
            {
                // >>>> Initialization
                kf.errorCovPre.at<float>(0) = 1; // px
                kf.errorCovPre.at<float>(7) = 1; // px
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;
                kf.errorCovPre.at<float>(28) = 1; // px
                kf.errorCovPre.at<float>(35) = 1; // px
                
                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
                state.at<float>(2) = 0;
                state.at<float>(3) = 0;
                state.at<float>(4) = meas.at<float>(2);
                state.at<float>(5) = meas.at<float>(3);
                // <<<< Initialization
                
                kf.statePost = state;
                
                found = true;
            }
            else
                kf.correct(meas); // Kalman Correction
            
            cout << "Measure matrix:" << endl << meas << endl;
        }
        // <<<<< Kalman Update
        
        // Final result
        cv::imshow("Tracking", res);
        
        // User key
        ch = cv::waitKey(1);
    }
    // <<<<< Main loop
    
    return EXIT_SUCCESS;
    
}

/*
    This method loads the camera matrix and distance coefficients from the 'name' file
    The file was created using a calibration process on the camera and then saved
*/
bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients)
{
    ifstream inStream(name);
    if (inStream)
    {
        uint16_t rows;
        uint16_t columns;
        
        inStream >> rows;
        inStream >> columns;
        
        cameraMatrix = Mat(Size(columns, rows), CV_64F);
        
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                double read = 0.0f;
                inStream >> read;
                cameraMatrix.at<double>(r,c) = read;
                cout << cameraMatrix.at<double>(r,c) << "\n";
            }
        }
        inStream >> rows;
        inStream >> columns;
        
        distanceCoefficients = Mat::zeros(rows, columns, CV_64F);
        
        
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                double read = 0.0f;
                inStream >> read;
                distanceCoefficients.at<double>(r,c) = read;
                cout << distanceCoefficients.at<double>(r,c) << "\n";
            }
        }
        inStream.close();
        return true;
    
    }
    return false;

}

/*
    This method converts euler angles to quaternion
*/
Quaterniond toQuaternion(double pitch, double roll, double yaw)
{
    Quaterniond q;
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    
    q.w() = cy * cr * cp + sy * sr * sp;
    q.x() = cy * sr * cp - sy * cr * sp;
    q.y() = cy * cr * sp + sy * sr * cp;
    q.z() = sy * cr * cp - cy * sr * sp;
    return q;
}

/*
    This method converts a matrix M of type Mat to quaternion.
    It is used here to convert the Rodrigues rotation vector to quaternion
*/
Eigen::Quaterniond toQuaternionOg(const cv::Mat &M)
{
    Eigen::Matrix<double, 3, 3> eigMat;
    cv2eigen(M, eigMat);
    Eigen::Quaterniond q(eigMat);
    
    return q;
}

/*
    This method gets the Euler roll angle from a quaternion
*/
static double getEulerX(const Quaterniond& q)
{
    // roll (x-axis rotation)
    double sinr = +2.0 * (q.w() * q.x() + q.y() * q.z());
    double cosr = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
    return atan2(sinr, cosr);
}

/*
    This method gets the Euler pitch angle from a quaternion
*/
static double getEulerY(const Quaterniond& q)
{
    // pitch (y-axis rotation)
    double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
    if (fabs(sinp) >= 1)
       return copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
       return asin(sinp);
}

/*
    This method gets the Euler yaw angle from a quaternion
*/
static double getEulerZ(const Quaterniond& q)
{
    // yaw (z-axis rotation)
    double siny = +2.0 * (q.w() * q.z() + q.x() * q.y());
    double cosy = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
    return atan2(siny, cosy);
}

/*
    This method computes the position of the corners of the markers in the camera world
    where the camera is at position (0,0,0). It then returns the position of the center of the cube
    by averaging the marker corners position translated to the center of the cube
*/
Point3f getCubePositionInGlobalFrame(double side, Vec3d rvec, Vec3d tvec){
    
    double half_side = side/2;
    
    
    // compute rot_mat
    Mat rot_mat;
    Rodrigues(rvec, rot_mat);
    
    // transpose of rot_mat for easy columns extraction
    Mat rot_mat_t = rot_mat.t();
    
    // the two E-O and F-O vectors
    double * tmp = rot_mat_t.ptr<double>(0);
    Point3f camWorldE(tmp[0]*half_side,
                      tmp[1]*half_side,
                      tmp[2]*half_side);
    
    tmp = rot_mat_t.ptr<double>(1);
    Point3f camWorldF(tmp[0]*half_side,
                      tmp[1]*half_side,
                      tmp[2]*half_side);
    
    // convert tvec to point
    Point3f tvec_3f(tvec[0], tvec[1], tvec[2]);
    
    // return vector:
    vector<Point3f> ret(5,tvec_3f);
    
    ret[0] +=  camWorldE + camWorldF; //top right corner
    ret[1] += -camWorldE + camWorldF; //top left corner
    ret[2] += -camWorldE - camWorldF; //bottom left corner
    ret[3] +=  camWorldE - camWorldF; //bottom right corner
    
    ret[0].x -= side/2;
    ret[0].y -= side/2;
    ret[0].z -= side/2;
    
    ret[1].x += side/2;
    ret[1].y -= side/2;
    ret[1].z -= side/2;
    
    ret[2].x += side/2;
    ret[2].y += side/2;
    ret[2].z -= side/2;
    
    ret[3].x -= side/2;
    ret[3].y += side/2;
    ret[3].z -= side;
    
    ret[4].x = (ret[0].x + ret[1].x + ret[2].x + ret[3].x)/4;
    ret[4].y = (ret[0].y + ret[1].y + ret[2].y + ret[3].y)/4;
    ret[4].z = (ret[0].z + ret[1].z + ret[2].z + ret[3].z)/4;
    return ret[4];
}

/*
    This is the main algorithm
    This method detects the markers, estimates their pose and returns the position of the cube
    as well as its orientation
    The orientation assumes that the marker number 4 starts on top of the cube
*/
int StartWebcamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDimensions)
{
    int counter = 0; // Help variable for the orientation used later
    Mat frame, rotMat;
    Mat kalmanCamera;
    vector<int> markerIds; // Vector that stores the ids of the markers detected
    vector<vector<Point2f>> markerCorners; // Corners of the markers
    aruco::DetectorParameters parameters;
    Ptr<aruco::Dictionary> markerDictionary3 = aruco::Dictionary::create(6, 4);
    VideoCapture vid(1); // Capturing the video, 1 is used for external camera
    vid.set(CV_CAP_PROP_FRAME_WIDTH,640); // We set the width to 640, to give 30 FPS
    vid.set(CV_CAP_PROP_FRAME_HEIGHT,480); // We set the height to 480
    double startRotationX = 0; // Start rotation X of marker 4 assumed to be on top
    double startRotationY = 0; // Start rotation Y of marker 4 assumed to be on top
    
    // All the following variables are for computing the orientation later
    double currentRotationX0 = 0;
    double currentRotationY0 = 0;
    double currentRotationX1 = 0;
    double currentRotationY1 = 0;
    double currentRotationX2 = 0;
    double currentRotationY2 = 0;
    double currentRotationX3 = 0;
    double currentRotationY3 = 0;
    double currentRotationX4 = 0;
    double currentRotationY4 = 0;
    double currentRotationX5 = 0;
    double currentRotationY5 = 0;
    double cubeRotationX = 0;
    double cubeRotationY = 0;
    double cubeRotationZ = 0;
    vector<float> orientation;
    
    if (!vid.isOpened())
        return -1;
    
    namedWindow("Camera", CV_WINDOW_AUTOSIZE);
    
    
    vector<Vec3d> rotationVectors, translationVectors;
    while (true)
    {
        
        // If the video is not properly working, we break
        if (!vid.read(frame))
            break;
        // Detect the markers, by comparing to the custom dictionary and store in markerIds
        aruco::detectMarkers(frame, markerDictionary3, markerCorners, markerIds);
        // We estimate their pose by outputing a rotation and translation vector
        aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimensions, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);
        
        
        for (int i = 0; i < markerIds.size(); i++)
        {
            // This is for marker 4, assumed to be on top of the cube at the beginning
            if (markerIds.at(i) == 4) {
                // First, transform the rotation vector into a rotation matrix
                Rodrigues(rotationVectors[i], rotMat);
                // Then transform this Rodrigues matrix into quaternion
                Eigen::Quaterniond q = toQuaternionOg(rotMat);
                // We store the orientation it started with so that it starts at 0 degrees in Euler
                if (counter == 0) {
                    startRotationX =  (getEulerX(q) * 180)/M_PI;
                    startRotationY =  (getEulerY(q) * 180)/M_PI;
                }
                counter = 1; // Now that counter is 1, we stored just the orientation it started with
                             // and do not update startRotation anymore
                // We get the current rotation in Euler angles
                currentRotationX4 = (getEulerX(q) * 180)/M_PI;
                currentRotationY4 = (getEulerY(q) * 180)/M_PI;
                // We compute the cube rotation by getting the current orientation minus how it started
                cubeRotationX = currentRotationX4 - startRotationX;
                cubeRotationY = currentRotationY4 - startRotationY;
                // The yaw rotation is just the yaw of the marker
                cubeRotationZ = (getEulerZ(q)*180/M_PI);
            }
            
            // If marker 0 is detected (which is on top of marker 4)
            // Assuming marker 4 is orientated in the right direction as in the dictionary
            if (markerIds.at(i) == 0) {
                Rodrigues(rotationVectors[i], rotMat);
                Eigen::Quaterniond q = toQuaternionOg(rotMat);
                currentRotationX0 = (getEulerX(q) * 180)/M_PI;
                currentRotationY0 = (getEulerY(q) * 180)/M_PI;
                // We add 90 to the orientation in X
                cubeRotationX = - 90 + currentRotationX0;
                cubeRotationY = currentRotationY0;
                cubeRotationZ = (getEulerZ(q)*180/M_PI);
            }
            
            // If marker 1 is detected (which is on the left of marker 4)
            // Assuming marker 4 is orientated in the right direction as in the dictionary
            if (markerIds.at(i) == 1) {
                Rodrigues(rotationVectors[i], rotMat);
                Eigen::Quaterniond q = toQuaternionOg(rotMat);
                currentRotationX1 = (getEulerX(q) * 180)/M_PI;
                currentRotationY1 = (getEulerY(q) * 180)/M_PI;
                cubeRotationX = currentRotationX1 - startRotationX;
                // Since going to the right is +90 degrees in Y, then we go 3 times to the right
                // which adds 270 degrees to the Y rotation
                cubeRotationY = 270 + currentRotationY1;
                cubeRotationZ = (getEulerZ(q)*180/M_PI);
            }
            
            // If marker 2 is detected (which is on the right of marker 4)
            // Assuming marker 4 is orientated in the right direction as in the dictionary
            if (markerIds.at(i) == 2) {
                Rodrigues(rotationVectors[i], rotMat);
                Eigen::Quaterniond q = toQuaternionOg(rotMat);
                currentRotationX2 = (getEulerX(q) * 180)/M_PI;
                currentRotationY2 = (getEulerY(q) * 180)/M_PI;
                cubeRotationX = currentRotationX2 - startRotationX;
                // We add 90 to the Y rotation since we go right by one marker and add its
                // local orientation
                cubeRotationY = currentRotationY2 + 90;
                cubeRotationZ = (getEulerZ(q)*180/M_PI);
            }
            
            // If marker 3 is detected (which is on the botton of marker 4)
            // Assuming marker 4 is orientated in the right direction as in the dictionary
            if (markerIds.at(i) == 3) {
                Rodrigues(rotationVectors[i], rotMat);
                Eigen::Quaterniond q = toQuaternionOg(rotMat);
                currentRotationX3 = (getEulerX(q) * 180)/M_PI;
                currentRotationY3 = (getEulerY(q) * 180)/M_PI;
                // We remove 90 from the rotation in X
                cubeRotationX = 90 + currentRotationX3;
                cubeRotationY = currentRotationY3;
                cubeRotationZ = (getEulerZ(q)*180/M_PI);
            }
            
            // If marker 5 is detected (which is 2 markers the right of marker 4 and 2 markers on top)
            // Assuming marker 4 is orientated in the right direction as in the dictionary
            if (markerIds.at(i) == 5) {
                Rodrigues(rotationVectors[i], rotMat);
                Eigen::Quaterniond q = toQuaternionOg(rotMat);
                currentRotationX5 = (getEulerX(q) * 180)/M_PI;
                currentRotationY5 = (getEulerY(q) * 180)/M_PI;
                cubeRotationX = - currentRotationX5 + 360;
                cubeRotationY = 180 + currentRotationY5;
                cubeRotationZ = (getEulerZ(q)*180/M_PI);
            }
            
            // These computations is to avoid having negative degree results, and just go from 0 to 360
            if (cubeRotationX < 0) {
                cubeRotationX = 360 + cubeRotationX;
            }

            if (cubeRotationX > 360) {
                cubeRotationX = cubeRotationX - 360;
            }

            if (cubeRotationY < 0) {
                cubeRotationY = 360 + cubeRotationY;
            }

            if (cubeRotationY > 360) {
                cubeRotationY = cubeRotationY - 360;
            }

            if (cubeRotationZ < 0) {
                cubeRotationZ = 360 + cubeRotationZ;
            }

            if (cubeRotationZ > 360) {
                cubeRotationZ = cubeRotationZ - 360;
            }
            
            // We output the results
//
            cout << "Orientation of cube in X in degrees: " << cubeRotationX << endl;
            cout << endl;
            cout << "Orientation of cube in Y in degrees: " << cubeRotationY << endl;
            cout << endl;
            cout << "Orientation of cube in Z in degrees: " << cubeRotationZ << endl;
            cout << endl;
            
//           Quaterniond cubeQuat = toQuaternion(cubeRotationY, cubeRotationX, cubeRotationZ);
//           orientation.push_back(cubeQuat.w());
//            cout << "Orientation in quaternion: [" << cubeQuat.w() << "," << cubeQuat.vec().x() << "," << cubeQuat.vec().y() << "," << cubeQuat.vec().z() << "]" << endl;
            
           cout << "Position of center of cube in world frame: " << getCubePositionInGlobalFrame(arucoSquareDimensions, rotationVectors[i], translationVectors[i]) << endl;
            
            // Draw the 3 axes when detecting a marker using their rotation and translation vecotrs
           aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.1f);
        }
    
        // Output the camera visualization
        imshow("Camera", frame);
        // If the ESC key is pressed stop
        if (waitKey(30) >= 0) break;
    }
    
    return 1;
}

/*
    The main function, which loads the camera calibration values and then runs the algorithm
*/
int main(int argv, char** argc)
{
    createArucoMakers();
    Mat frame;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distanceCoefficients;
    loadCameraCalibration("CameraCalibration", cameraMatrix, distanceCoefficients);
    StartWebcamMonitoring(cameraMatrix, distanceCoefficients, 0.026f);
    //kalmanFilter(frame);
    return 0;
}
