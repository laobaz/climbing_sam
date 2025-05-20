//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <ctime>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "config_parser.h"
#include "preprocess_op.h"
#include "utils.h"


struct ImageProcessingStats {
  int totalImages = 0;
  int processedImages = 0;
  int failedImages = 0;
  double totalTime = 0.0;
};

namespace PaddleDetection {

// Generate visualization colormap for each class
std::vector<int> GenerateColorMap(int num_class);

// Visualiztion Detection Result
cv::Mat VisualizeResult(
    const cv::Mat& img,
    const std::vector<PaddleDetection::ObjectResult>& results,
    const std::vector<std::string>& lables,
    const std::vector<int>& colormap,
    const bool is_rbox);

class ObjectDetector {
 public:
  explicit ObjectDetector(const std::string& config_file,const std::string& model_filename) 
  {
    config_.load_config(config_file);
    threshold_ = config_.draw_threshold_;
    preprocessor_.Init(config_.preprocess_info_);
    LoadModel(model_filename);
  }

  // Load Paddle inference model
  void LoadModel(const std::string& model_filename);

  // Run predictor
  void Predict(const std::vector<cv::Mat> imgs,
               const int warmup = 0,
               const int repeats = 1,
               std::vector<PaddleDetection::ObjectResult>* result = nullptr,
               std::vector<int>* bbox_num = nullptr,
               std::vector<double>* times = nullptr);

  // Get Model Label list
  const std::vector<std::string>& GetLabelList() const {
    return config_.label_list_;
  }

 private:
  // Preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat& image_mat);
  // Postprocess result
  // void Postprocess(const std::vector<cv::Mat> mats,
  //                  std::vector<PaddleDetection::ObjectResult>* result,
  //                  std::vector<int> bbox_num,
  //                  std::vector<float> output_data_,
  //                  std::vector<int> output_mask_data_,
  //                  bool is_rbox);


  Preprocessor preprocessor_;
  ImageBlob inputs_;
  float threshold_;
  ConfigPaser config_;
};

}  // namespace PaddleDetection
