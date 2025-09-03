#include <iostream>
#include <opencv2/opencv.hpp>

int main(){
    std::cout<<"test_camera_list: starting"<<std::endl;
    for(int i=0;i<5;i++){
        cv::VideoCapture cap(i);
        if(cap.isOpened()){
            std::cout<<"Camera "<<i<<" available"<<std::endl;
            cap.release();
        }
    }
    return 0;
}
#include <iostream>
#include <opencv2/opencv.hpp>

int main(){
    std::cout<<"Enumerating cameras (0-9)"<<std::endl;
    for(int i=0;i<10;++i){
        cv::VideoCapture cap(i);
        if(cap.isOpened()){
            std::cout<<"Camera "<<i<<" opened. Resolutions: ";
            std::cout<<cap.get(cv::CAP_PROP_FRAME_WIDTH)<<"x"<<cap.get(cv::CAP_PROP_FRAME_HEIGHT)<<std::endl;
            cap.release();
        }
    }
    return 0;
}
