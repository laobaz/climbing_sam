

#include <iomanip>

#include <memory>

#include <dirent.h>
#include <cstddef>
#include <vector>

#include <xtensor.hpp>
#include <xtensor/xnpy.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>

#include <dirent.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "logging.h"
#include <uuid/uuid.h>

#include "object_detector.h"

#define DEVICE 0

#define CHECK(status)                                    \
  do                                                     \
  {                                                      \
    auto ret = (status);                                 \
    if (ret != 0)                                        \
    {                                                    \
      std::cerr << "Cuda failure: " << ret << std::endl; \
      abort();                                           \
    }                                                    \
  } while (0)

using namespace nvinfer1;

#define NMS_THRESH 0.45

#define BBOX_CONF_THRESH 0.12
// #define BBOX_CONF_THRESH  0.25
// #define BBOX_CONF_THRESH  0.45
// #define BBOX_CONF_THRESH  0.7
// #define BBOX_CONF_THRESH  0.5

namespace PaddleDetection
{

  static const int INPUT_W = 640;
  static const int INPUT_H = 640;

  // static const int NUM_CLASSES = 11;//电子围栏
  // static const int NUM_CLASSES = 1;//chepai
  static const int NUM_CLASSES = 3; // smoke phone

  static const int CHANNELS = 3;

  std::vector<int> input_image_shape = {640, 640};

  using namespace nvinfer1;

  IRuntime *runtime{nullptr};
  ICudaEngine *engine{nullptr};
  IExecutionContext *context{nullptr};

  NVLogger gLogger_test;

  //---------------------------------------------------------------------
  struct Object
  {
    cv::Rect_<float> rect;
    int label;
    float prob;
  };

  static inline float intersection_area(const Object &a, const Object &b)
  {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
  }

  static inline void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right)
  {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
      while (faceobjects[i].prob > p)
        i++;

      while (faceobjects[j].prob < p)
        j--;

      if (i <= j)
      {
        // swap
        std::swap(faceobjects[i], faceobjects[j]);

        i++;
        j--;
      }
    }

#pragma omp parallel sections
    {
#pragma omp section
      {
        if (left < j)
          qsort_descent_inplace(faceobjects, left, j);
      }
#pragma omp section
      {
        if (i < right)
          qsort_descent_inplace(faceobjects, i, right);
      }
    }
  }

  static inline void qsort_descent_inplace(std::vector<Object> &objects)
  {
    if (objects.empty())
      return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
  }

  static inline void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold)
  {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
      areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
      const Object &a = faceobjects[i];

      int keep = 1;
      for (int j = 0; j < (int)picked.size(); j++)
      {
        const Object &b = faceobjects[picked[j]];

        // intersection over union
        float inter_area = intersection_area(a, b);
        float union_area = areas[i] + areas[picked[j]] - inter_area;
        // float IoU = inter_area / union_area
        if (inter_area / union_area > nms_threshold)
          keep = 0;
      }

      if (keep)
        picked.push_back(i);
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  static const char *INPUT_BLOB_NAME1 = "im_shape";     // float32[p2o.DynamicDimension.1,3,640,640]
  static const char *INPUT_BLOB_NAME2 = "image";        // float32[p2o.DynamicDimension.0,2]
  static const char *INPUT_BLOB_NAME3 = "scale_factor"; // float32[p2o.DynamicDimension.2,2]

  // static const char* OUTPUT_BLOB_NAME1 = "reshape2_83.tmp_0";            //float32[N,6]
  // static const char* OUTPUT_BLOB_NAME2 = "tile_3.tmp_0";   //int32[-1]  //若干N

  static const char *OUTPUT_BLOB_NAME1 = "save_infer_model/scale_0.tmp_0"; // float32[N,6]
  static const char *OUTPUT_BLOB_NAME2 = "save_infer_model/scale_1.tmp_0"; // int32[-1]  //若干N

  void doInference(IExecutionContext &context, float *input1, float *input2, float *input3, std::vector<float> &output_bsc, int &output2)
  {

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine->getNbBindings() == 5);
    void *buffers[5];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex1 = engine->getBindingIndex(INPUT_BLOB_NAME1);
    const int inputIndex2 = engine->getBindingIndex(INPUT_BLOB_NAME2);
    const int inputIndex3 = engine->getBindingIndex(INPUT_BLOB_NAME3);

    assert(engine->getBindingDataType(inputIndex1) == nvinfer1::DataType::kFLOAT);
    assert(engine->getBindingDataType(inputIndex2) == nvinfer1::DataType::kFLOAT);
    assert(engine->getBindingDataType(inputIndex3) == nvinfer1::DataType::kFLOAT);

    const int outputIndex1 = engine->getBindingIndex(OUTPUT_BLOB_NAME1); //
    const int outputIndex2 = engine->getBindingIndex(OUTPUT_BLOB_NAME2); //
    assert(engine->getBindingDataType(outputIndex1) == nvinfer1::DataType::kFLOAT);
    assert(engine->getBindingDataType(outputIndex2) == nvinfer1::DataType::kINT32);

    // Create GPU buffers on device

    //"im_shape";       //float32[-1,2]       im_shape (Tensor): The shape of the input image without padding.
    CHECK(cudaMalloc(&buffers[inputIndex1], 2 * sizeof(float)));

    //  "image";       //float32[-1,3,640,640]
    CHECK(cudaMalloc(&buffers[inputIndex2], 3 * 640 * 640 * sizeof(float)));

    // "scale_factor";//float32[-1,2]         scale_factor (Tensor): The scale factor of the input image.
    CHECK(cudaMalloc(&buffers[inputIndex3], 2 * sizeof(float)));

    // "reshape2_83.tmp_0";      //float32[N,6] bbox_pred (Tensor): The output prediction with shape [N, 6], including labels, scores and bboxes.
    // 先分配最大个数100个
    CHECK(cudaMalloc(&buffers[outputIndex1], 500 * 6 * sizeof(float)));
    output_bsc.assign(500 * 6, 0.0f);

    // " tile_3.tmp_0";   //int32[-1]  bbox_num; shape [bs], and is N.
    CHECK(cudaMalloc(&buffers[outputIndex2], sizeof(int)));

    output2 = 0;

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host

    // shape
    CHECK(cudaMemcpyAsync(buffers[inputIndex1], input1, 2 * sizeof(float), cudaMemcpyHostToDevice, stream));
    // img data
    CHECK(cudaMemcpyAsync(buffers[inputIndex2], input2, 3 * 640 * 640 * sizeof(float), cudaMemcpyHostToDevice, stream));
    // scale
    CHECK(cudaMemcpyAsync(buffers[inputIndex3], input3, 2 * sizeof(float), cudaMemcpyHostToDevice, stream));

    //std::cout << "enqueueV2 start" << std::endl;
    context.enqueueV2(buffers, stream, nullptr);
    //std::cout << "enqueueV2 end" << std::endl;

    CHECK(cudaMemcpyAsync(&output2, buffers[outputIndex2], sizeof(int), cudaMemcpyDeviceToHost, stream));

    CHECK(cudaMemcpyAsync(output_bsc.data(), buffers[outputIndex1], 500 * 6 * sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex1]));
    CHECK(cudaFree(buffers[inputIndex2]));
    CHECK(cudaFree(buffers[inputIndex3]));

    CHECK(cudaFree(buffers[outputIndex1]));
    CHECK(cudaFree(buffers[outputIndex2]));
  }

  //////////////////////////////////////////////////////////////////////////

  // Load Model and creatstd::cout<<"have a complete person"<<std::endl;e model predictor
  void ObjectDetector::LoadModel(const std::string &model_filename)
  {

    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(model_filename, std::ios::binary);
    if (file.good())
    {
      file.seekg(0, file.end);
      size = file.tellg();
      file.seekg(0, file.beg);
      trtModelStream = new char[size];
      assert(trtModelStream);
      file.read(trtModelStream, size);
      file.close();
    }
    else
    {
      std::cout << "ppyoloe_test : Bad engine file!" << std::endl;
      exit(-1);
    }

    std::cout << "ppyoloe_test : Success read engine file!" << model_filename << std::endl;

    runtime = createInferRuntime(gLogger_test.getTRTLogger());
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
  }

  // Visualiztion MaskDetector results
  cv::Mat VisualizeResult(
      const cv::Mat &img,
      const std::vector<PaddleDetection::ObjectResult> &results,
      const std::vector<std::string> &lables,
      const std::vector<int> &colormap,
      const bool is_rbox = false)
  {
    cv::Mat vis_img = img.clone();
    int img_h = vis_img.rows;
    int img_w = vis_img.cols;
    for (int i = 0; i < results.size(); ++i)
    {
      // Configure color and text size
      std::ostringstream oss;
      oss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
      oss << lables[results[i].class_id] << " ";
      oss << results[i].confidence;
      std::string text = oss.str();
      int c1 = colormap[3 * results[i].class_id + 0];
      int c2 = colormap[3 * results[i].class_id + 1];
      int c3 = colormap[3 * results[i].class_id + 2];
      cv::Scalar roi_color = cv::Scalar(c1, c2, c3);
      int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
      double font_scale = 0.5f;
      float thickness = 0.5;
      cv::Size text_size =
          cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
      cv::Point origin;

      if (is_rbox)
      {
        // Draw object, text, and background
        for (int k = 0; k < 4; k++)
        {
          cv::Point pt1 = cv::Point(results[i].rect[(k * 2) % 8],
                                    results[i].rect[(k * 2 + 1) % 8]);
          cv::Point pt2 = cv::Point(results[i].rect[(k * 2 + 2) % 8],
                                    results[i].rect[(k * 2 + 3) % 8]);
          cv::line(vis_img, pt1, pt2, roi_color, 2);
        }
      }
      else
      {
        int w = results[i].rect[2] - results[i].rect[0];
        int h = results[i].rect[3] - results[i].rect[1];
        cv::Rect roi = cv::Rect(results[i].rect[0], results[i].rect[1], w, h);
        // Draw roi object, text, and background
        cv::rectangle(vis_img, roi, roi_color, 2);

        // Draw mask
        std::vector<int> mask_v = results[i].mask;
        if (mask_v.size() > 0)
        {
          cv::Mat mask = cv::Mat(img_h, img_w, CV_32S);
          std::memcpy(mask.data, mask_v.data(), mask_v.size() * sizeof(int));

          cv::Mat colored_img = vis_img.clone();

          std::vector<cv::Mat> contours;
          cv::Mat hierarchy;
          mask.convertTo(mask, CV_8U);
          cv::findContours(
              mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
          cv::drawContours(colored_img,
                           contours,
                           -1,
                           roi_color,
                           -1,
                           cv::LINE_8,
                           hierarchy,
                           100);

          cv::Mat debug_roi = vis_img;
          colored_img = 0.4 * colored_img + 0.6 * vis_img;
          colored_img.copyTo(vis_img, mask);
        }
      }

      origin.x = results[i].rect[0];
      origin.y = results[i].rect[1];

      // Configure text background
      cv::Rect text_back = cv::Rect(results[i].rect[0],
                                    results[i].rect[1] - text_size.height,
                                    text_size.width,
                                    text_size.height);
      // Draw text, and background
      cv::rectangle(vis_img, text_back, roi_color, -1);
      cv::putText(vis_img,
                  text,
                  origin,
                  font_face,
                  font_scale,
                  cv::Scalar(255, 255, 255),
                  thickness);
    }
    return vis_img;
  }

  void ObjectDetector::Preprocess(const cv::Mat &ori_im)
  {
    // Clone the image : keep the original mat for postprocess
    cv::Mat im = ori_im.clone();
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    preprocessor_.Run(&im, &inputs_);
  }

  // void ObjectDetector::Postprocess(
  //     const std::vector<cv::Mat> mats,
  //     std::vector<PaddleDetection::ObjectResult>* result,
  //     std::vector<int> bbox_num,
  //     std::vector<float> output_data_,
  //     std::vector<int> output_mask_data_,
  //     bool is_rbox = false) {
  //   result->clear();
  //   int start_idx = 0;
  //   int total_num = std::accumulate(bbox_num.begin(), bbox_num.end(), 0);
  //   int out_mask_dim = -1;
  //   if (config_.mask_) {
  //     out_mask_dim = output_mask_data_.size() / total_num;
  //   }

  //   for (int im_id = 0; im_id < mats.size(); im_id++) {
  //     cv::Mat raw_mat = mats[im_id];
  //     int rh = 1;
  //     int rw = 1;
  //     // if (config_.arch_ == "Face") {
  //     //   rh = raw_mat.rows;
  //     //   rw = raw_mat.cols;
  //     // }
  //     for (int j = start_idx; j < start_idx + bbox_num[im_id]; j++) {
  //       if (is_rbox) {
  //         // // Class id
  //         // int class_id = static_cast<int>(round(output_data_[0 + j * 10]));
  //         // // Confidence score
  //         // float score = output_data_[1 + j * 10];
  //         // int x1 = (output_data_[2 + j * 10] * rw);
  //         // int y1 = (output_data_[3 + j * 10] * rh);
  //         // int x2 = (output_data_[4 + j * 10] * rw);
  //         // int y2 = (output_data_[5 + j * 10] * rh);
  //         // int x3 = (output_data_[6 + j * 10] * rw);
  //         // int y3 = (output_data_[7 + j * 10] * rh);
  //         // int x4 = (output_data_[8 + j * 10] * rw);
  //         // int y4 = (output_data_[9 + j * 10] * rh);

  //         // PaddleDetection::ObjectResult result_item;
  //         // result_item.rect = {x1, y1, x2, y2, x3, y3, x4, y4};
  //         // result_item.class_id = class_id;
  //         // result_item.confidence = score;
  //         // result->push_back(result_item);
  //       } else {
  //         // Class id
  //         int class_id = static_cast<int>(round(output_data_[0 + j * 6]));
  //         // Confidence score
  //         float score = output_data_[1 + j * 6];
  //         int xmin = (output_data_[2 + j * 6] * rw);
  //         int ymin = (output_data_[3 + j * 6] * rh);
  //         int xmax = (output_data_[4 + j * 6] * rw);
  //         int ymax = (output_data_[5 + j * 6] * rh);
  //         int wd = xmax - xmin;
  //         int hd = ymax - ymin;

  //         PaddleDetection::ObjectResult result_item;
  //         result_item.rect = {xmin, ymin, xmax, ymax};
  //         result_item.class_id = class_id;
  //         result_item.confidence = score;

  //         if (config_.mask_) {
  //           std::vector<int> mask;
  //           for (int k = 0; k < out_mask_dim; ++k) {
  //             if (output_mask_data_[k + j * out_mask_dim] > -1) {
  //               mask.push_back(output_mask_data_[k + j * out_mask_dim]);
  //             }
  //           }
  //           result_item.mask = mask;
  //         }

  //         result->push_back(result_item);
  //       }
  //     }
  //     start_idx += bbox_num[im_id];
  //   }
  // }

  void ObjectDetector::Predict(const std::vector<cv::Mat> imgs,
                               const int warmup,
                               const int repeats,
                               std::vector<PaddleDetection::ObjectResult> *result,
                               std::vector<int> *bbox_num,
                               std::vector<double> *times)
  {
    //auto logger = spdlog::get("file_logger");
    auto preprocess_start = std::chrono::steady_clock::now();
    int batch_size = imgs.size();

    // in_data_batch
    std::vector<float> in_data_all;
    std::vector<float> im_shape_all(batch_size * 2);
    std::vector<float> scale_factor_all(batch_size * 2);
    std::vector<const float *> output_data_list_;
    // std::vector<int> out_bbox_num_data_;
    std::vector<int> out_mask_data_;

    // in_net img for each batch
    std::vector<cv::Mat> in_net_img_all(batch_size);

    // Preprocess image
    for (int bs_idx = 0; bs_idx < batch_size; bs_idx++)
    {
      cv::Mat im = imgs.at(bs_idx);
      Preprocess(im);
      im_shape_all[bs_idx * 2] = inputs_.im_shape_[0];
      im_shape_all[bs_idx * 2 + 1] = inputs_.im_shape_[1];

      scale_factor_all[bs_idx * 2] = inputs_.scale_factor_[0];
      scale_factor_all[bs_idx * 2 + 1] = inputs_.scale_factor_[1];

      // TODO: reduce cost time
      in_data_all.insert(
          in_data_all.end(), inputs_.im_data_.begin(), inputs_.im_data_.end());

      // collect in_net img
      in_net_img_all[bs_idx] = inputs_.in_net_im_;
    }

    // Pad Batch if batch size > 1 xb deleted!!!

    result->clear();
    bbox_num->clear();

    auto preprocess_end = std::chrono::steady_clock::now();

    // Run predictor
    // std::vector<std::vector<float>> out_tensor_list;
    // std::vector<std::vector<int>> output_shape_list;

    bool is_rbox = false;
    int reg_max = 7;
    int num_class = NUM_CLASSES;
    // warmup xb deleted

    std::vector<float> out_data_bsc;

    auto inference_start = std::chrono::steady_clock::now();
    int out_boxes_num = 0;

    // const ICudaEngine engine = context.getEngine();
    nvinfer1::IExecutionContext *context4thisCam(engine->createExecutionContext());
    
   // std::cout << "doInference  start" << std::endl;
    doInference(*context4thisCam, im_shape_all.data(), in_data_all.data(), scale_factor_all.data(), out_data_bsc, out_boxes_num);
    context4thisCam->destroy();
    //td::cout << "doInference  end" << std::endl;

    //std::cout << "out_data_bsc num is " << out_boxes_num;
    //td::cout << std::endl;

    bbox_num->clear();

    if (out_boxes_num == 0)
    {
      return;
    }

    /////////////////////////////////////////////////////////////////////////
    // std::cout<<" out_data_bsc:";
    // int printCnt=0;
    // for(auto item:out_data_boxes)
    // {
    //     printCnt++;
    //     std::cout<<item<<",";
    //     if(printCnt > 24)
    //     {
    //         break;
    //     }

    // }

    // std::cout<<std::endl;

    // std::cout<< "output_size_scores num is " << out_data_scores.size();

    // std::cout<<" out_data_scores:";
    // printCnt=0;
    // for(auto item:out_data_scores)
    // {
    //     printCnt++;
    //     std::cout<<item<<",";
    //     if(printCnt > 24)
    //     {
    //         break;
    //     }

    // }
    // std::cout<<std::endl;

    /////////////////////////////////////////
    std::vector<std::size_t> tmpshape = {out_boxes_num, 6}; // label,score,x1,y1,x2,y2
    auto boxes = xt::adapt(out_data_bsc, tmpshape);

    //     std::vector<std::size_t> tmpshape2 = { NUM_CLASSES,8400 };
    //     auto scores_probs = xt::adapt(out_data_scores, tmpshape2);

    //   //python: classes = np.argmax(box_scores, axis=-1)
    // //   xt::xtensor<int,8400> classes=xt::argmax(scores_probs, 0);
    //   auto classes=xt::argmax(scores_probs, 0);
    //   std::cout << "classes shape:" <<xt::adapt(classes.shape()) <<std::endl;

    //   //python: scores = np.max(box_scores, axis=-1)
    // //   xt::xtensor<float,8400> scores = xt::amax(scores_probs,0);
    //   auto scores = xt::amax(scores_probs,0);
    //   std::cout << "scores shape:" <<xt::adapt(scores.shape()) <<std::endl;
    //   // std::cout << "scores:" <<scores <<std::endl;

    std::vector<Object> proposals;
    for (size_t i = 0; i < out_boxes_num; i++)
    {
      Object obj;
      obj.rect.x = boxes(i, 2);
      obj.rect.y = boxes(i, 3);
      obj.rect.width = boxes(i, 4) - obj.rect.x;  // x2-x1
      obj.rect.height = boxes(i, 5) - obj.rect.y; // y2-y1
      obj.label = boxes(i, 0);
      obj.prob = boxes(i, 1);

      // if(i < 50)
      // {
          // std::cout << " score:"  << obj.prob
          //         <<" label:"  << obj.label
          //         <<" x:"  << obj.rect.x
          //         <<" y:"  << obj.rect.y
          //         <<" width:"  << obj.rect.width
          //         <<" height:"  << obj.rect.height
          //         <<std::endl;
      // }
 

      if (!isfinite(obj.prob))
      {
        // ANNIWOLOG(INFO) <<logstr<<":get unexpected infinite value!!" ;
        //std::cout << ":get unexpected infinite value,ignored!!" << std::endl;
        continue;
      }

      if (obj.rect.width < 1 || obj.rect.height < 1)
      {
        // std::cout << "this box ignored for h,w < 1"<< std::endl;
        continue;
      }
      if (obj.rect.x < 0 || obj.rect.y < 0)
      {
        // std::cout << "this box ignored for - x,y"<< std::endl;
        continue;
      }
      if (obj.prob < BBOX_CONF_THRESH)
      {
        // std::cout << "this box ignored for score:"<<obj.prob << std::endl;
        continue;
      }

      proposals.push_back(obj);
    }

    //std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);

    int count = picked.size();

    //std::cout << "num of boxes after nms: " << count << std::endl;

    // picked convert to -1,6 -> plain vector
    //  std::vector<float> out_data(6*count);

    // for(size_t i=0;i<count;i++)
    for (int i = 0; i < count; i++)
    {

      // Class id
      int class_id = static_cast<int>(round(proposals[picked[i]].label));
      // Confidence score
      float score = proposals[picked[i]].prob;
      int xmin = (proposals[picked[i]].rect.x);
      int ymin = (proposals[picked[i]].rect.y);
      int xmax = (proposals[picked[i]].rect.x + proposals[picked[i]].rect.width);
      int ymax = (proposals[picked[i]].rect.y + proposals[picked[i]].rect.height);
      // int wd = xmax - xmin;
      // int hd = ymax - ymin;
      // spdlog::info("class={} confidence={:.4f} rect=[{} {} {} {}]", 
      //   class_id,     
      //   score, 
      //   xmin, ymin, xmax, ymax);

      PaddleDetection::ObjectResult result_item;
      result_item.rect = {xmin, ymin, xmax, ymax};
      result_item.class_id = class_id;
      result_item.confidence = score;

      // printf("class=%d confidence=%.4f rect=[%d %d %d %d ]\n",
      //        result_item.class_id,
      //        result_item.confidence,
      //        result_item.rect[0],
      //        result_item.rect[1],
      //        result_item.rect[2],
      //        result_item.rect[3]);
      spdlog::info("class={} confidence={:.4f} rect=[{} {} {} {}]", 
        result_item.class_id,
        result_item.confidence,
        result_item.rect[0],
        result_item.rect[1],
        result_item.rect[2],
        result_item.rect[3]);

      result->push_back(result_item);

      // out_data[i*6+0]= proposals[picked[i]].label; //Class id
      // out_data[i*6+1]= proposals[picked[i]].prob; //score
      // out_data[i*6+2]= proposals[picked[i]].rect.x; //x1
      // out_data[i*6+3]= proposals[picked[i]].rect.y; //y1
      // out_data[i*6+4]= proposals[picked[i]].rect.x + proposals[picked[i]].rect.width; //x2
      // out_data[i*6+5]= proposals[picked[i]].rect.y + proposals[picked[i]].rect.height; //y2
    }
    bbox_num->push_back(count);
    


  }

  std::vector<int> GenerateColorMap(int num_class)
  {
    auto colormap = std::vector<int>(3 * num_class, 0);
    for (int i = 0; i < num_class; ++i)
    {
      int j = 0;
      int lab = i;
      while (lab)
      {
        colormap[i * 3] |= (((lab >> 0) & 1) << (7 - j));
        colormap[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j));
        colormap[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j));
        ++j;
        lab >>= 3;
      }
    }
    return colormap;
  }

} // namespace PaddleDetection
