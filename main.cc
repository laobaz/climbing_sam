

#include <filesystem>

#include <chrono>
#include <iomanip>

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
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <sys/stat.h>

#include "code/object_detector.h"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "rtdetr/rtdetr.h"
#include "Sam/SAM.h"

struct IniSection
{
  std::string name;
  std::map<std::string, std::string> keyValuePairs;
};

// 读取INI文件的主函数
std::vector<IniSection> readIniFile(const std::string &filename)
{
  std::vector<IniSection> sections;
  std::ifstream file(filename);

  if (!file.is_open())
  {
    spdlog::info("Error: Unable to open file {}", filename);
    return sections;
  }

  std::string line;
  IniSection *currentSection = nullptr;

  while (std::getline(file, line))
  {
    // 去除行首尾的空白字符
    line.erase(line.find_last_not_of(" \t") + 1);
    line.erase(0, line.find_first_not_of(" \t"));

    // 跳过空行和注释行（以;或#开头)
    if (line.empty() || line[0] == ';' || line[0] == '#')
    {
      continue;
    }

    // 检查是否是节头 [section]
    if (line[0] == '[' && line.back() == ']')
    {
      std::string sectionName = line.substr(1, line.size() - 2);
      sections.push_back({sectionName, {}});
      currentSection = &sections.back();
    }
    // 处理键值对
    else if (currentSection != nullptr)
    {
      size_t equalsPos = line.find('=');
      if (equalsPos != std::string::npos)
      {
        std::string key = line.substr(0, equalsPos);
        std::string value = line.substr(equalsPos + 1);

        // 去除键和值的首尾空白字符
        key.erase(key.find_last_not_of(" \t") + 1);
        key.erase(0, key.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));

        // 去除值两端的引号（如果有）
        if (!value.empty() && ((value.front() == '"' && value.back() == '"') ||
                               (value.front() == '\'' && value.back() == '\'')))
        {
          value = value.substr(1, value.size() - 2);
        }

        if (!key.empty())
        {
          currentSection->keyValuePairs[key] = value;
        }
      }
    }
  }

  file.close();
  return sections;
}

std::string getIniValue(const std::vector<IniSection> &sections,
                        const std::string &sectionName,
                        const std::string &key,
                        const std::string &defaultValue = "")
{
  for (const auto &section : sections)
  {
    if (section.name == sectionName)
    {
      auto it = section.keyValuePairs.find(key);
      if (it != section.keyValuePairs.end())
      {
        return it->second;
      }
      break;
    }
  }
  return defaultValue;
}

namespace fs = std::filesystem;
using namespace spdlog;
static std::string DirName(const std::string &filepath)
{
  auto pos = filepath.rfind(OS_PATH_SEP);
  if (pos == std::string::npos)
  {
    return "";
  }
  return filepath.substr(0, pos);
}

static bool PathExists(const std::string &path)
{
#ifdef _WIN32
  struct _stat buffer;
  return (_stat(path.c_str(), &buffer) == 0);
#else
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
#endif // !_WIN32
}

static void MkDir(const std::string &path)
{
  if (PathExists(path))
    return;
  int ret = 0;
#ifdef _WIN32
  ret = _mkdir(path.c_str());
#else
  ret = mkdir(path.c_str(), 0755);
#endif // !_WIN32
  if (ret != 0)
  {
    std::string path_error(path);
    path_error += " mkdir failed!";
    throw std::runtime_error(path_error);
  }
}

static void MkDirs(const std::string &path)
{
  if (path.empty())
    return;
  if (PathExists(path))
    return;

  MkDirs(DirName(path));
  MkDir(path);
}

int main(int argc, char **argv)
{



  auto logger = spdlog::basic_logger_mt("RT_SAM", "logs/app.log", true);
  // auto logger = spdlog::stdout_color_mt("console");
  set_default_logger(logger);
  set_level(spdlog::level::debug);
  std::string filename = "config.ini";
  auto iniData = readIniFile(filename);
  if (iniData.empty())
  {
    spdlog::info("No configuration data loaded or file not found.");
    return 1;
  }
  else
  {
    spdlog::info("successfully loaded data to config");
  }

  // std::string IOU=getIniValue(iniData,"default_ini","rt_iou","0.5");

  std::string engine_file_path_in = getIniValue(iniData, "address", "engine_file_path_in", "");
  std::string config_file_path_in = getIniValue(iniData, "address", "config_file_path_in", "");
  std::string default_outimgsPath=getIniValue(iniData, "address", "outimgsPath", "");
  std::string output_dir=getIniValue(iniData, "address", "output_dir", "./");
  std::string onnx_paths_encoder=getIniValue(iniData, "address", "onnx_paths_encoder", "");
  std::string onnx_paths_decoder=getIniValue(iniData, "address", "onnx_paths_decoder", "");
  std::string sam_score=getIniValue(iniData, "default_ini", "sam_score", "0.5");
  std::string sam_nms=getIniValue(iniData, "default_ini", "sam_nms", "0.5");
  std::string encoder_thread=(iniData, "default_ini", "encoder_thread", "2");
  std::string decoder_thread=(iniData, "default_ini", "decoder_thread", "2");
  std::string rt_threshold1 = getIniValue(iniData, "default_ini", "rt_detection_threshold", "0.5");
  std::string rt_threshold2 = getIniValue(iniData, "default_ini", "rt_judge_threshold", "0.7");
  std::string rt_batch_size = getIniValue(iniData, "default_ini", "rt_batch_size", "1");
  spdlog::info("sam_score:{} ,sam_nms:{},encoder_thread:{} ,decoder_thread:{},rt_detection_threshold:{},rt_judge_threshold:{},rt_batch_size:{}",
    sam_score,sam_nms,encoder_thread,decoder_thread,rt_threshold1,rt_threshold2,rt_batch_size);

  // std::cout << "staring" << std::endl;
  //const char *engine_file_path_in = "trt_model_xz/rtdetr_r50vd_6x_coco_xz_new.trt";
  // const char* engine_file_path_in="./trt_model_helmet/rtdetr_r50vd_6x_coco.trt";
  //const char *config_file_path_in = "trt_model_xz/infer_cfg.yml";

  //const char *output_dir = "./";
  // std::vector<std::string> onnx_paths{
  //     "trt_model_xz/sam/ESAM_encoder.onnx",
  //     "trt_model_xz/sam/ESAM_decoder.onnx"};


  std::vector<std::string> onnx_paths{onnx_paths_encoder,onnx_paths_decoder};

  // 建立日志系统
  //  创建文件日志器（自动线程安全）


  // spdlog::debug("This is a debug message");
  // spdlog::info("Welcome to spdlog!");
  // spdlog::error("Some error occurred: {}", 42);

  std::string video_or_filePath = argv[1];
  std::string outimgsPath = argv[2]==""?default_outimgsPath:argv[2];
  std::string models = argv[3];

  cv::VideoCapture cap(video_or_filePath);

  Rtdetr Rt(std::stof(rt_threshold1), std::stof(rt_threshold2), std::stof(rt_batch_size));
  // std::cout<<"tip"<<std::endl;
  Rt.initialize(config_file_path_in, engine_file_path_in);
  SAM Sam(std::stof(sam_score),std::stof(sam_nms),std::stof(encoder_thread),std::stof(decoder_thread));
  Sam.initialize(onnx_paths, true);
  if (models == "video")
  {
    if (!cap.isOpened())
    {
      logger->error("Error opening video stream or file");
      return -1;
    }
    else
    {
      logger->info("vdeo exists: {}", video_or_filePath);
    }
    struct stat info;

    if (stat(argv[2], &info) != 0)
    {
      logger->error("Cannot access ");
      return 1;
    }
    else if (info.st_mode & S_IFDIR)
    {
      logger->info("Directory exists: {}", argv[2]);
    }
    else
    {
      logger->error(" {}is not a directory.", argv[2]);
      return 1;
    }
    // 获取视频帧率
    static double fps = static_cast<double>(cap.get(cv::CAP_PROP_FPS));
    static int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    static int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frameInterval = static_cast<int>(fps); //
    spdlog::info("fps is:{},frame_width is: {}, frame_height is: {}", fps, frame_width, frame_height);
    cv::Mat frame;
    int frameCount = 0;
    // std::cout<<"tip"<<std::endl;

    cv::VideoWriter outputVideo("output_with_detections.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), frameInterval, cv::Size(frame_width, frame_height));
    std::vector<Ort::Value> output_tensors;
    std::vector<PaddleDetection::ObjectResult> boxs;
    std::vector<PaddleDetection::ObjectResult> all_bboxs;
    while (cap.read(frame))
    {
      // spdlog::info("当前 logger 数量: {}", spdlog::details::registry::instance().);

      // std::cout<<"staring video"<<std::endl;
      //  检查是否是该处理的帧
      //  在这里处理每秒的帧
      //  std::cout << "Processing frame at " << frameCount / fps << " seconds." << std::endl;
      // rtdetr判断成功
      boxs.clear();
      output_tensors.clear();
      all_bboxs.clear();
      spdlog::info("staring handle NO: {} ", frameCount);
      bool having_warn = Rt.predict_video(frame, boxs, all_bboxs);

      // logger->info("Successfully detected the person climbing high");

      // 进行SAM的判断
      if (having_warn)
      {
        spdlog::info("Successfully detected the person climbing high");
        for (int i = 0; i < boxs.size(); i++)
        {
          spdlog::info("having boxs:{} ", boxs[i].class_id);
        }
        if (Sam.inference(frame, boxs, output_tensors))
        {
          // Rt.visualizeWarning(true, frameCount, fps, frame, frame_width, frame_height, output_dir, outputVideo);

          try
          {
            Sam.postprocess(output_tensors, frame);
          }
          catch (const std::exception &e)
          {
            spdlog::info("tensor postprocess failed!!");
          }

          Rt.visualizeWarning(having_warn, frameCount, fps, frame, frame_width, frame_height, outimgsPath, outputVideo, all_bboxs);
          // outputVideo.write(frame);
        }
        else
        {
          spdlog::info("not detected the person's Sam_object");
          Rt.visualizeWarning(having_warn, frameCount, fps, frame, frame_width, frame_height, outimgsPath, outputVideo, all_bboxs);
        }
      }
      else
      {
        spdlog::info("not detected the person climbing high");
        Rt.visualizeWarning(having_warn, frameCount, fps, frame, frame_width, frame_height, outimgsPath, outputVideo, all_bboxs);
      }
      spdlog::default_logger()->flush(); // 强制刷新缓冲区
      frameCount++;
    }
    spdlog::info("视频处理完成");
    spdlog::default_logger()->flush(); // 强制刷新缓冲区
    sleep(3);
    cap.release();
    outputVideo.release();
    Sam.~SAM();
  }

  else if (models == "photo")
  {
    ImageProcessingStats stats;

    // 检查目录是否存在
    if (!fs::exists(video_or_filePath) || !fs::is_directory(video_or_filePath))
    {
      logger->info("The specified path is not a valid directory");
      return 1;
    }
    else
    {
      info("photo_path exists: {}", video_or_filePath);
    }

    // 支持的图片格式
    const std::vector<std::string> supportedFormats = {".jpg", ".jpeg", ".png", ".bmp"};

    // 遍历目录
    for (const auto &entry : fs::directory_iterator(video_or_filePath))
    {
      if (entry.is_regular_file())
      {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (std::find(supportedFormats.begin(), supportedFormats.end(), ext) != supportedFormats.end())
        {
          stats.totalImages++;
          logger->info("处理中 {}", entry.path().string());
          ;
          // processImage(entry.path().string(), stats);
          cv::Mat frame = cv::imread(entry.path().string());
          std::vector<Ort::Value> output_tensors;
          std::vector<PaddleDetection::ObjectResult> boxs;
          std::vector<PaddleDetection::ObjectResult> all_bboxs;
          bool having_warn = Rt.predict_video(frame, boxs, all_bboxs);

          // logger->info("Successfully detected the person climbing high");
          //  进行SAM的判断
          fs::path p(entry.path().string());
          std::string outputPath = outimgsPath + "/processed_" + p.filename().string();
          if (having_warn)
          {
            for (int i = 0; i < boxs.size(); i++)
            {
              spdlog::info("having boxs:{} ", boxs[i].class_id);
            }
            if (Sam.inference(frame, boxs, output_tensors))
            {
              spdlog::info("successfully Sam");
              stats.processedImages++;
              Sam.postprocess(output_tensors, frame);
              Rt.visualizeWarning(having_warn, frame, outputPath, all_bboxs);
            }
            else
            {
              stats.failedImages++;
              Rt.visualizeWarning(having_warn, frame, outputPath, all_bboxs);
            }
          }
          else
          {
            stats.failedImages++;
            Rt.visualizeWarning(having_warn, frame, outputPath, all_bboxs);
          }
        }
      }
      // 输出统计信息
    }
    std::cout << "\n处理完成:\n"
              << "总图片数: " << stats.totalImages << "\n"
              << "成功处理: " << stats.processedImages << "\n"
              << "失败处理: " << stats.failedImages << "\n";
  }
  else
  {
    logger->error("{} is no a models", models);
  }
  spdlog::shutdown();

  // if (cv::waitKey(30) >= 0)
  // break; // 按下任意键退出
}
