#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <utility>
#include <stdarg.h>

#include <sys/stat.h>

#include "../code/object_detector.h"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class Rtdetr
{
private:
    float detection_threshold; // 阈值
    float judge_threshold;
    PaddleDetection::ObjectDetector *det = nullptr;
    std::vector<std::string> labels;
    bool is_rbox = false;
    int batch_size;

public:
    // Rtdetr(float f) : thread(f) {}
    Rtdetr(float ts1, float ts2, float bs)
    {
        
        detection_threshold = ts1;
        judge_threshold=ts2;
        batch_size = bs;
        // spdlog::info("detection_threshold:{},judge_threshold:{},batch_size:{}",
        //     detection_threshold,
        //     judge_threshold,
        //     batch_size);

    }
    ~Rtdetr() { delete (det); }
    void initialize(std::string config_file_path_in, std::string engine_file_path_in);
    bool predict_video(cv::Mat flame,
                       // const int batch_size,
                       // const double threshold,
                       //  PaddleDetection::ObjectDetector *det,
                       std::vector<PaddleDetection::ObjectResult> &boxs,
                       std::vector<PaddleDetection::ObjectResult> &all_bboxs);

    bool predict_climb_or(std::vector<std::pair<PaddleDetection::ObjectResult,
                                                std::pair<PaddleDetection::ObjectResult,
                                                          PaddleDetection::ObjectResult>>>
                              F_p_F,
                          std::vector<PaddleDetection::ObjectResult> climb_result);

    void visualizeWarning(bool warn,
                          int frameCount,
                          int fps,
                          const cv::Mat &vis_img,
                          int frame_width,
                          int frame_height,
                          const std::string &output_dir,
                          cv::VideoWriter &outputVideo,
                          const std::vector<PaddleDetection::ObjectResult> &boxs); // 输出结果
    void visualizeWarning(bool warn, const cv::Mat &vis_img, const std::string &output_dir, const std::vector<PaddleDetection::ObjectResult> &boxs);
};
