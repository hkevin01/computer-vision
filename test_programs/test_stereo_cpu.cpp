#include <iostream>
#include <opencv2/opencv.hpp>

int main(){
    std::cout<<"test_stereo_cpu: starting"<<std::endl;
    cv::Mat left = cv::imread("data/stereo_images/left.png", cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread("data/stereo_images/right.png", cv::IMREAD_GRAYSCALE);
    if(left.empty() || right.empty()){
        std::cout<<"Sample stereo images not found in data/stereo_images/ — skipping"<<std::endl;
        return 2;
    }
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0,16,3);
    cv::Mat disp;
    sgbm->compute(left,right,disp);
    cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite("reports/smoke/disparity.png", disp);
    std::cout<<"Wrote reports/smoke/disparity.png"<<std::endl;
    return 0;
}
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv){
    std::string left = "data/stereo_images/left.png";
    std::string right = "data/stereo_images/right.png";
    std::string out = "disparity.png";
    if(argc>1) left = argv[1];
    if(argc>2) right = argv[2];
    if(argc>3) out = argv[3];
    cv::Mat L = cv::imread(left, cv::IMREAD_GRAYSCALE);
    cv::Mat R = cv::imread(right, cv::IMREAD_GRAYSCALE);
    if(L.empty() || R.empty()){
        std::cerr << "Failed to load sample stereo images: "<<left<<" "<<right<<"\n";
        return 2;
    }
    int minDisparity = 0;
    int numDisparities = 16*5;
    int blockSize = 5;
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(minDisparity,numDisparities,blockSize);
    cv::Mat disp;
    sgbm->compute(L,R,disp);
    double minVal, maxVal;
    cv::minMaxLoc(disp,&minVal,&maxVal);
    cv::Mat disp8;
    disp.convertTo(disp8, CV_8U, 255/(maxVal - minVal));
    cv::imwrite(out, disp8);
    std::cout<<"Wrote disparity to "<<out<<std::endl;
    return 0;
}
