#include "rtdetr.h"
#include <spdlog/spdlog.h>

//static float thread = 0.7;

static inline float intersection_area(const PaddleDetection::ObjectResult &a, const PaddleDetection::ObjectResult &b)
{
  int left = std::max(a.rect[0], b.rect[0]);
  int right = std::min(a.rect[2], b.rect[2]);
  int top = std::max(a.rect[1], b.rect[1]);
  int bottom = std::min(a.rect[3], a.rect[3]);

  // 计算宽度和高度
  int width = right - left;
  int height = bottom - top;

  if (width > 0 && height > 0)
  {
    return (float)(width * height);
  }
  else
  {
    return 0; // 没有交集
  }
}
static inline float chepaiobj_areas(const PaddleDetection::ObjectResult &a)
{
  return (a.rect[3] - a.rect[1]) * (a.rect[2] - a.rect[0]);
}

void Rtdetr::initialize(std::string config_file_path_in, std::string engine_file_path_in)
{
  //auto logger = spdlog::get("file_logger");
  spdlog::info("staring init rtdetr‘s engine :{}", engine_file_path_in);
  det = new PaddleDetection::ObjectDetector(config_file_path_in, engine_file_path_in);
  std::cout << "success init rtdetr engine" << std::endl;

  labels=det->GetLabelList();   
}

bool Rtdetr::predict_climb_or(std::vector<std::pair<PaddleDetection::ObjectResult,
                                                    std::pair<PaddleDetection::ObjectResult, PaddleDetection::ObjectResult>>>
                                  F_p_F,
                              std::vector<PaddleDetection::ObjectResult> climb_result)
{

  //auto logger = spdlog::get("file_logger");
  for (int i = 0; i < F_p_F.size(); i++)
  {
    spdlog::info("now is person:{} handle", i);
    // 判断人是否在攀登物上里面
    for (int j = 0; j < climb_result.size(); j++)
    {
      spdlog::info("staring handle climbing {}", climb_result[j].class_id);
      float inter_area = intersection_area(F_p_F[i].first, climb_result[j]);
      // std::cout<<"person's inter_area is "<<inter_area<<std::endl;
      float chepaiobj_area = chepaiobj_areas(F_p_F[i].first);
      if (chepaiobj_area == 0)
        continue;
      // std::cout<<"person's chepaiobj_area is "<<chepaiobj_area<<std::endl;
      double IOU = inter_area / chepaiobj_area;
      //std::cout<<"person's IOU is "<<IOU<<std::endl;

      if (IOU> (double)(judge_threshold))
      {
        //[0][1]为左上角x,y[2][3]为右下角x,y
        // int w = results[i].rect[2] - results[i].rect[0];
        //  int h = results[i].rect[3] - results[i].rect[1];
        //  int person_foot_x= F_p_F[i].second.second.rect[0];
        //  int person_foot_y=F_p_F[i].second.second.rect[1];
        //  int person_foot_width=F_p_F[i].second.second.rect[2]-F_p_F[i].second.second.rect[0];
        //  int person_foot_height=F_p_F[i].second.second.rect[3]-F_p_F[i].second.second.rect[1];

        // 高的攀登物情况 脚
        if (climb_result[j].rect[3] > F_p_F[i].second.second.rect[3]    // 脚的y轴在墙上，
            && climb_result[j].rect[0] < F_p_F[i].second.second.rect[0] // 脚在墙的范围内
            && climb_result[j].rect[2] > F_p_F[i].second.second.rect[2])
        {
          spdlog::info("Successfully detected person:{} climbing high", i);
          return true;
        }
        // 头
        else if (climb_result[j].rect[0] - 15 < F_p_F[i].second.first.rect[0] // 头超过墙上边，
                 && climb_result[j].rect[0] < F_p_F[i].second.first.rect[0]   // 头在墙的范围内
                 && climb_result[j].rect[2] > F_p_F[i].second.first.rect[2])
        {
          spdlog::info("Successfully detected person:{} climbing high", i);
          return true;
        }
        else
        {
          continue;
        }
      }
      else
      {
        continue;
      }
    }
  }
  spdlog::info("no detected person climbing high");
  return false;
}

bool Rtdetr::predict_video(cv::Mat flame,
                          // const int batch_size,
                           //const double threshold,
                           // PaddleDetection::ObjectDetector *det,
                           std::vector<PaddleDetection::ObjectResult> &boxs,
                           std::vector<PaddleDetection::ObjectResult> &all_bboxs)
{
  //auto logger = spdlog::get("file_logger");

  spdlog::info("detection_threshold:{},judge_threshold:{},batch_size:{}",
    detection_threshold,
    judge_threshold,
    batch_size);
  std::vector<PaddleDetection::ObjectResult> result;
  std::vector<int> bbox_num; // 每张图片的object数目
  std::vector<double> det_times;
  bool is_rbox = false;

  spdlog::info("Predicting start");
  std::vector<cv::Mat> imgs;
  imgs.push_back(flame);
  det->Predict(imgs, 0, 1, &result, &bbox_num, &det_times);

  spdlog::info("Predicting end");
  // get labels and colormap
  auto labels = det->GetLabelList();
  auto colormap = PaddleDetection::GenerateColorMap(labels.size());
  int item_start_idx = 0;

  cv::Mat &im = flame;
  //std::vector<PaddleDetection::ObjectResult> im_result;
  std::vector<PaddleDetection::ObjectResult> face_result;
  std::vector<PaddleDetection::ObjectResult> foot_result;
  std::vector<PaddleDetection::ObjectResult> person_result;
  std::vector<std::pair<PaddleDetection::ObjectResult,
                        std::pair<PaddleDetection::ObjectResult,
                                  PaddleDetection::ObjectResult>>>
      F_p_F;
  std::vector<PaddleDetection::ObjectResult> climb_result;
  //int detect_num = 0;
  // 过滤object部分
  for (int j = 0; j < bbox_num[0]; j++)
  {
    PaddleDetection::ObjectResult item = result[item_start_idx + j];
    // std::cout<<
    if (!isfinite(item.confidence))
    {
      // ANNIWOLOG(INFO) <<logstr<<":get unexpected infinite value!!" ;
      // std::cout << ":get unexpected infinite value!!" << std::endl;
    }

    if (item.confidence <  detection_threshold|| item.class_id == -1)
    {
      // spdlog::info("item.confidence :{}",item.confidence);
      // spdlog::info("跳过");
      continue;
    }
    //detect_num += 1;
    all_bboxs.push_back(item);
    if (item.class_id == 7)
    {
      foot_result.push_back(item);
    }
    else if (item.class_id == 6)
    {
      face_result.push_back(item);
    }
    else if (item.class_id == 0)
    {
      person_result.push_back(item);
      //boxs.push_back(item);
    }
    else if (item.class_id != 3)
    {
      climb_result.push_back(item);
      boxs.push_back(item);
    }
  }
  
  bool have_person = false;
  for (auto pobj : person_result)
  {
    
    PaddleDetection::ObjectResult face_and_p_obj;
    bool face_and_p = false;
    for (auto face_obj : face_result)
    {
      float inter_area = intersection_area(pobj, face_obj);
      
      float chepaiobj_area = chepaiobj_areas(face_obj);
      double IOU = inter_area / chepaiobj_area;
      spdlog::info("inter_area:{},chepaiobj_area:{},IOU:{}",inter_area,chepaiobj_area,IOU);
      if (IOU> judge_threshold)
      {
        face_and_p_obj = face_obj;
        face_and_p = true;
        break;
      }
    }
    PaddleDetection::ObjectResult foot_and_p_obj;
    bool foot_and_p = false;
    for (auto foot_obj : foot_result)
    {
      float inter_area = intersection_area(pobj, foot_obj);
      float chepaiobj_area = chepaiobj_areas(foot_obj);
      double IOU = inter_area / chepaiobj_area;
      spdlog::info("inter_area:{},chepaiobj_area:{},IOU:{}",inter_area,chepaiobj_area,IOU);
      if (IOU> judge_threshold)
      {
        foot_and_p_obj = foot_obj;
        foot_and_p = true;
        break;
      }
    }
    if (foot_and_p && face_and_p)
    {
      F_p_F.push_back(std::make_pair(pobj, std::make_pair(face_and_p_obj, foot_and_p_obj)));
      have_person = true;
      boxs.push_back(pobj);
    }
  }
  int i = 0;
  for (auto obj : F_p_F)
  {
    spdlog::info("having  person{} is:rect[{} {} {} {}],it's head is :rect[{} {} {} {}] ，it's foot is :rect[{} {} {} {}]",
                 i++,
                 obj.first.rect[0], obj.first.rect[1], obj.first.rect[2], obj.first.rect[3],
                 obj.second.first.rect[0], obj.second.first.rect[1], obj.second.first.rect[2], obj.second.first.rect[3],
                 obj.second.second.rect[0], obj.second.second.rect[1], obj.second.second.rect[2], obj.second.second.rect[3]);
  }

  // 如果有完整的人物检测信息，执行相关逻辑（此部分可以根据需求扩展）
  if (have_person)
  {
    // 判断是否有登高的危险
    spdlog::info("staring predict warn?");
    // std::cout << "staring predict warn?" << std::endl;
    return this->predict_climb_or(F_p_F, climb_result);
  }
  else
  {
    spdlog::info("no predict warn!");
    return false;
  }     
}

void Rtdetr::visualizeWarning(bool warn,

                              int frameCount,
                              int fps,
                              const cv::Mat &sam_img,
                              int frame_width,
                              int frame_height,
                              const std::string &output_dir,
                              cv::VideoWriter &outputVideo,
                              const std::vector<PaddleDetection::ObjectResult> &boxs)
{

  auto colormap = PaddleDetection::GenerateColorMap(labels.size());
  cv::Mat vis_img = PaddleDetection::VisualizeResult(
    sam_img, boxs, labels, colormap, is_rbox);
  //auto logger = spdlog::get("file_logger");
  if (warn)
  {
    // 创建红色矩形框
    cv::Scalar red_color(0, 0, 255);            // 红色
    int thickness = 2;                          // 框的厚度
    cv::Mat img_with_warning = vis_img.clone(); // 克隆图像以便处理

    // 绘制矩形
    cv::rectangle(img_with_warning, cv::Point(2, 2), cv::Point(frame_width - 5, frame_height - 5), red_color, thickness = 1);
    spdlog::info("成功绘制矩阵框");
    // 标签文本和字体设置
    std::string label = "Warning";
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 1.0;
    int font_thickness = 1;
    cv::Scalar font_color(0, 0, 255); // 红色字体
    int baseline = 0;

    // 计算标签文字大小，以便确定文字放置位置
    cv::Size text_size = cv::getTextSize(label, font_face, font_scale, font_thickness, &baseline);
    // 文字位置：放置在矩形的中心
    cv::Point text_org((frame_width - text_size.width) / 2, text_size.height + 2);

    // 在图像上绘制标签
    cv::putText(img_with_warning,
                label,
                text_org,
                font_face,
                font_scale,
                font_color,
                font_thickness);

    // 每秒保存图像
    if (frameCount % (fps) == 0)
    {
      spdlog::info("warn!!! maybe having person is climbing");
      //std::cout << "warn!!! maybe having person is climbing" << std::endl;
      std::string img_path = std::to_string(frameCount / fps);

      // 构造输出路径
      std::string output_path(output_dir);
      if (output_dir.rfind(OS_PATH_SEP) != output_dir.size() - 1)
      {
        output_path += OS_PATH_SEP;
      }
      output_path += (std::string("warn_") + img_path + ".jpg");

      // 保存图像
      cv::imwrite(output_path, img_with_warning);
      spdlog::info("Visualized output saved as {}", output_path.c_str());

      // printf("Visualized output saved as %s\n", output_path.c_str());
    }
    // 输出视频
    outputVideo.write(img_with_warning);
    spdlog::info("成功Sam输出一帧视频");
  }
  else
  {
    spdlog::info("成功无Sam输出一帧视频");
    // 没有检测到需要警告的情况时，不做任何处理
    outputVideo.write(vis_img);
  }
}

void Rtdetr::visualizeWarning(bool warn, const cv::Mat &sam_img, const std::string &output_dir, const std::vector<PaddleDetection::ObjectResult> &boxs)   
{
  auto colormap = PaddleDetection::GenerateColorMap(labels.size());
  cv::Mat vis_img = PaddleDetection::VisualizeResult(
    sam_img, boxs, labels, colormap, is_rbox);
  //auto logger = spdlog::get("file_logger");
  // 构造输出路径
  if (warn)
  {
    // 创建红色矩形框
    cv::Scalar red_color(0, 0, 255);            // 红色
    int thickness = 2;                          // 框的厚度
    cv::Mat img_with_warning = vis_img.clone(); // 克隆图像以便处理

    // 绘制矩形
    cv::rectangle(img_with_warning, cv::Point(2, 2), cv::Point(sam_img.cols - 5, sam_img.rows - 5), red_color, thickness = 1);

    // 标签文本和字体设置
    std::string label = "Warning";
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5;
    int font_thickness = 1;
    cv::Scalar font_color(0, 0, 255); // 红色字体
    int baseline = 0;

    // 计算标签文字大小，以便确定文字放置位置
    cv::Size text_size = cv::getTextSize(label, font_face, font_scale, font_thickness, &baseline);
    // 文字位置：放置在矩形的中心
    cv::Point text_org((sam_img.cols - text_size.width) / 2, text_size.height + 2);

    // 在图像上绘制标签
    cv::putText(img_with_warning,
                label,
                text_org,
                font_face,
                font_scale,
                font_color,
                font_thickness);

    // 每秒保存图像

    spdlog::info("warn!!! maybe having person is climbing");

    if (!cv::imwrite(output_dir, img_with_warning))
    {
      spdlog::error("Unable to save processing results");
    }

    // 保存图像
    // cv::imwrite(output_path, img_with_warning);

    spdlog::info("Visualized output saved as {}", output_dir.c_str());
  }
  else
  {
    // 没有检测到需要警告的情况时，不做任何处理
    cv::imwrite(output_dir, vis_img);
  }
}