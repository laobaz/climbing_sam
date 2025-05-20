#pragma once
#include <tbb/tbb.h>
#include <fstream>
#include <memory>
#include <vector>
#include <iostream>
#include <functional>
#include <tbb/tbb.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "../code/object_detector.h"
#include <bits/algorithmfwd.h>
#include <c++/9/bits/algorithmfwd.h>
#include <random>
#include<algorithm>


class SAM{

struct ParamsSam{
    float score = 0.5f;
    float nms = 0.5f;
};
struct Node{
    std::vector<int64_t> dim; // batch,channel,height,width
    char* name = nullptr;
};

private:
    bool is_inited = false;
    int encoder_thread=2;
    int decoder_thread=2;
    cv::Mat* ori_img = nullptr;
    ParamsSam parms;

    //Env
	Ort::Env encoder_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"sam_encoder");
	Ort::Env decoder_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"sam_decoder");
	
	//onnx会话配置相关
	Ort::Session* encoder_session = nullptr;
	Ort::Session* decoder_session = nullptr;
	
	//输入相关
	std::vector<Node> encoder_input_nodes;
	std::vector<Node> decoder_input_nodes;
	//输出相关
	std::vector<Node> encoder_output_nodes;
	std::vector<Node> decoder_output_nodes;

	std::vector<cv::Mat> input_images;

    //options
	Ort::SessionOptions encoder_options = Ort::SessionOptions();
	Ort::SessionOptions decoder_options = Ort::SessionOptions();

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
protected:
    void preprocess(cv::Mat &image);
    
	std::vector<Ort::Value> encoder_infer(std::vector<Ort::Value>&);
	std::vector<Ort::Value> decoder_infer(std::vector<Ort::Value>&);
public:
    SAM(float score,float nms,float encoder,float decoder){
        parms.nms=nms;
        parms.score=score;
        encoder_thread=encoder;
        decoder_thread=decoder;
    };
	SAM(const SAM&) = delete;// 删除拷贝构造函数
    SAM& operator=(const SAM&) = delete;// 删除赋值运算符
    ~SAM(){
		if (encoder_session != nullptr) delete encoder_session;
		if (decoder_session != nullptr) delete decoder_session;
    };
    int setparms(ParamsSam parms);
    bool initialize(std::vector<std::string>& onnx_paths, bool is_cuda);
    bool inference(cv::Mat image,std::vector<PaddleDetection::ObjectResult> &boxes,std::vector<Ort::Value> &output_tensors);
    void postprocess(std::vector<Ort::Value> &output_tensors,cv::Mat &ori_img);
    
};