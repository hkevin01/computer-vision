#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv){
    std::string disp_path = "disparity.png";
    std::string out = "cloud.ply";
    if(argc>1) disp_path = argv[1];
    if(argc>2) out = argv[2];
    cv::Mat disp = cv::imread(disp_path, cv::IMREAD_GRAYSCALE);
    if(disp.empty()){
        std::cerr<<"Failed to load disparity image: "<<disp_path<<"\n";
        return 2;
    }
    // synthetic intrinsics
    float fx=700, fy=700, cx=disp.cols/2.0f, cy=disp.rows/2.0f, baseline=0.1f;
    std::ofstream ofs(out);
    ofs<<"ply\nformat ascii 1.0\n";
    ofs<<"element vertex "<< (disp.cols*disp.rows) <<"\n";
    ofs<<"property float x\nproperty float y\nproperty float z\n";
    ofs<<"end_header\n";
    for(int v=0; v<disp.rows; ++v){
        for(int u=0; u<disp.cols; ++u){
            float d = disp.at<unsigned char>(v,u);
            if(d<=0) { ofs<<"0 0 0\n"; continue; }
            float Z = fx * baseline / d;
            float X = (u - cx) * Z / fx;
            float Y = (v - cy) * Z / fy;
            ofs<<X<<" "<<Y<<" "<<Z<<"\n";
        }
    }
    ofs.close();
    std::cout<<"Wrote point cloud to "<<out<<std::endl;
    return 0;
}
