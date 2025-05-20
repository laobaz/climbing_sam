#include "SAM.h"
#include <algorithm>
#include <string>
#include <spdlog/spdlog.h>

void SAM::postprocess(std::vector<Ort::Value> &output_tensors, cv::Mat &ori_img)
{
    // std::mutex mutex; //确保多线程中，vector的正确性
    // std::vector<cv::Mat> out_imgs; // 二值图像，大小是原始图像大小
    // auto logger = spdlog::get("file_logger");
    auto rand = [](uint _min, uint _max)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(_min, _max);
        return dis(gen);
    };
    cv::Mat output_mask(ori_img.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    //并行求解所有通道的图像，保留合乎形状的通道
     tbb::parallel_for(tbb::blocked_range<int>(0, output_tensors.size(), 1),[&](const tbb::blocked_range<int>& r){
         auto index = r.begin();
         float* output = output_tensors[index].GetTensorMutableData<float>();
         cv::Mat outimg(ori_img.size(),CV_32FC1,output); // 这是padding图像上的输出mask
         cv::Mat dst;
         outimg.convertTo(dst, CV_8UC1, 255);
         cv::threshold(dst,dst,0,255,cv::THRESH_BINARY);
         //开运算
         cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
         cv::morphologyEx(dst, dst, cv::MORPH_OPEN, element);
         // 查找轮廓
         std::vector<std::vector<cv::Point>> contours;
         cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
         cv::drawContours(ori_img, contours, -1, cv::Scalar(255,20,20),2,cv::LINE_AA);
         //*****************************************************
         std::vector<cv::Mat> subimg(3);
         //int blue = std::min(30*index,255),green = std::min(50+30*index,255),red = std::min(100+30*index,255);
         int blue = rand(0,255),green = rand(0,255),red = rand(0,255);
         cv::threshold(dst,subimg[0],0,std::clamp(blue,0,70),cv::THRESH_BINARY);
         cv::threshold(dst,subimg[1],0,green,cv::THRESH_BINARY);
         cv::threshold(dst,subimg[2],0,red,cv::THRESH_BINARY);
         // 合并
         cv::Mat merge;
         cv::merge(subimg,merge);
         output_mask +=merge;
         // std::lock_guard<std::mutex> lock(mutex);
         // out_imgs.emplace_back(dst);
     });

    // for (int index = 0; index < output_tensors.size(); ++index)
    // {
    //     float *output = output_tensors[index].GetTensorMutableData<float>();
    //     cv::Mat outimg(ori_img.size(), CV_32FC1, output); // 这是padding图像上的输出mask
    //     cv::Mat dst;
    //     outimg.convertTo(dst, CV_8UC1, 255);
    //     cv::threshold(dst, dst, 0, 255, cv::THRESH_BINARY);

    //     // 开运算
    //     cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    //     cv::morphologyEx(dst, dst, cv::MORPH_OPEN, element);

    //     // 查找轮廓
    //     std::vector<std::vector<cv::Point>> contours;
    //     cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //     cv::drawContours(ori_img, contours, -1, cv::Scalar(255, 20, 20), 2, cv::LINE_AA);

    //     //*****************************************************
    //     std::vector<cv::Mat> subimg(3);
    //     int blue = rand(0,255),green = rand(0,255),red = rand(0,255);
    //     cv::threshold(dst, subimg[0], 0, std::clamp(blue, 0, 70), cv::THRESH_BINARY);
    //     cv::threshold(dst, subimg[1], 0, green, cv::THRESH_BINARY);
    //     cv::threshold(dst, subimg[2], 0, red, cv::THRESH_BINARY);

    //     // 合并
    //     cv::Mat merge;
    //     cv::merge(subimg, merge);
    //     output_mask += merge;
    // }


    // output_mask结果
    cv::cvtColor(output_mask, output_mask, cv::COLOR_BGR2BGRA);
    cv::cvtColor(ori_img, ori_img, cv::COLOR_BGR2BGRA);
    cv::addWeighted(output_mask, 0.65, ori_img, 1, 0, ori_img);
    cv::cvtColor(ori_img,ori_img,cv::COLOR_BGRA2BGR);
    spdlog::info("successfully output sam_img");
}

int SAM::setparms(ParamsSam parms)
{
    this->parms = std::move(parms);
    return 1;
}

bool SAM::initialize(std::vector<std::string> &onnx_paths, bool is_cuda)
{
    // 约定顺序是,encoder.onnx,decoder.onnx
    // auto logger = spdlog::get("file_logger");
    assert(onnx_paths.size() == 2);
    auto is_file = [](const std::string &filename)
    {
        std::ifstream file(filename.c_str());
        return file.good();
    };
    for (const auto &path : onnx_paths)
    {
        if (!is_file(path))
        {
            spdlog::error("Model file dose not exist.file:", path);
            // std::cout<<"Model file dose not exist.file:"<<path<<std::endl;
            return 1;
        }   
    }
    this->encoder_options.SetIntraOpNumThreads(encoder_thread); // 设置线程数量
    this->decoder_options.SetIntraOpNumThreads(decoder_thread); // 设置线程数量
    //***********************************************************           
    if (is_cuda)
    {
        try
        {
            OrtCUDAProviderOptions options;
            options.device_id = 0;
            options.arena_extend_strategy = 0;
            options.gpu_mem_limit = SIZE_MAX;
            options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
            options.do_copy_in_default_stream = 1;
            this->encoder_options.AppendExecutionProvider_CUDA(options);
            this->decoder_options.AppendExecutionProvider_CUDA(options);
            std::cout << "Using CUDA..." << std::endl;
        }
        catch (const std::exception &e)
        {
            std::string error(e.what());
            return false;
        }
    }
    else
    {
        spdlog::info("using cpu....");
    }
    //**************************************************************
    try
    {
#ifdef _WIN32

#else
        encoder_session = new Ort::Session(encoder_env, (const char *)onnx_paths[0].c_str(), this->encoder_options);
        decoder_session = new Ort::Session(decoder_env, (const char *)onnx_paths[1].c_str(), this->decoder_options);
#endif
    }
    catch (const std::exception &e)
    {
        spdlog::info("Failed to load model. Please check your onnx file!");
        return 0;
        // return new std::string("Failed to load model. Please check your onnx file!");
    }
    //**************************************************************
    Ort::AllocatorWithDefaultOptions allocator;
    size_t const encoder_input_num = this->encoder_session->GetInputCount(); // 1
    size_t const decoder_input_num = this->decoder_session->GetInputCount(); // 4

    //-------------------------------获取Encoder输入节点的信息------------------
    for (size_t index = 0; index < encoder_input_num; index++)
    {
        Ort::AllocatedStringPtr input_name_Ptr = this->encoder_session->GetInputNameAllocated(index, allocator);
        Node node;
        node.dim = {1, 3, -1, -1};
        char *name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->encoder_input_nodes.push_back(node);
    }
    //-------------------------------获取Decoder输入节点的信息------------------
    for (size_t index = 0; index < decoder_input_num; index++)
    {
        Ort::AllocatedStringPtr input_name_Ptr = this->decoder_session->GetInputNameAllocated(index, allocator);
        Node node;
        if (index == 0)
            node.dim = {1, 256, 64, 64};
        if (index == 1)
            node.dim = {1, 1, 2, 2};
        if (index == 2)
            node.dim = {1, 1, 2};
        if (index == 3)
            node.dim = {2};
        char *name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->decoder_input_nodes.push_back(node);
    }
    //*******************************************************************
    size_t const encoder_output_num = this->encoder_session->GetOutputCount(); // 1
    size_t const decoder_output_num = this->decoder_session->GetOutputCount(); // 1
    //-------------------------------获取Encoder输出节点的信息------------------
    for (size_t index = 0; index < encoder_output_num; index++)
    {
        Ort::AllocatedStringPtr output_name_Ptr = this->encoder_session->GetOutputNameAllocated(index, allocator);
        Node node;
        node.dim = {1, 256, 64, 64};
        char *name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        //*************************
        this->encoder_output_nodes.push_back(node);
    }
    //-------------------------------获取Decoder输出节点的信息------------------
    for (size_t index = 0; index < decoder_output_num; index++)
    {
        Ort::AllocatedStringPtr output_name_Ptr = this->decoder_session->GetOutputNameAllocated(index, allocator);
        Node node;
        if (index == 0)
            node.dim = {1, 1, 3, -1, -1};
        if (index == 1)
            node.dim = {1, 1, 3};
        if (index == 2)
            node.dim = {1, 3, 256, 256};
        char *name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->decoder_output_nodes.push_back(node);
    }
    //****************************打印模型信息*******************************

    spdlog::info("*************************Encoder**********************");
    for (const auto &inputs : encoder_input_nodes)
    {
        spdlog::info("input name:");
        for (size_t i = 0; i < inputs.dim.size() - 1; i++)
        {
            // std::print("{}*",inputs.dim[i]);
            spdlog::info("{}*", inputs.dim[i]);
            // std::cout << inputs.dim[i]<< "*";
        }
        spdlog::info(inputs.dim[inputs.dim.size() - 1]);
        // std::cout<<inputs.dim[inputs.dim.size()-1];
    }
    // std::cout<<"--------------------------------";
    for (const auto &outputs : encoder_output_nodes)
    {
        spdlog::info("{}=", outputs.name);
        for (size_t i = 0; i < outputs.dim.size() - 1; i++)
        {
            // std::print("{}*",outputs.dim[i]);
            spdlog::info("{}", outputs.dim[i]);
        }
        // std::println("{}",outputs.dim[outputs.dim.size()-1]);
    }
    spdlog::info("*************************Decoder**********************");
    for (const auto &inputs : decoder_input_nodes)
    {
        // std::print("{}=",inputs.name);
        spdlog::info("inputs.name");
        for (size_t i = 0; i < inputs.dim.size() - 1; i++)
        {
            // std::print("{}*",inputs.dim[i]);
        }
        // std::println("{}",inputs.dim[inputs.dim.size()-1]);
    }
    // std::println("--------------------------------");
    for (const auto &outputs : decoder_output_nodes)
    {
        // std::print("{}=",outputs.name);
        for (size_t i = 0; i < outputs.dim.size() - 1; i++)
        {
            //  std::print("{}*",outputs.dim[i]);
        }
        // std::println("{}",outputs.dim[outputs.dim.size()-1]);
    }
    // std::println("******************************************************");
    //*********************************************************************
    this->is_inited = true;
    spdlog::info("initialize ok!!");
    return true;
}

void SAM::preprocess(cv::Mat &image)
{
    input_images.clear();
    std::vector<cv::Mat> mats{image};
    cv::Mat blob_encoder = cv::dnn::blobFromImages(mats, 1 / 255.0, image.size(), cv::Scalar(0, 0, 0), true, false);
    input_images.push_back(std::move(blob_encoder));
    //spdlog::info("preprocess 中 input_images 的数量是 {}",input_images.size());
}

bool SAM::inference(cv::Mat image, std::vector<PaddleDetection::ObjectResult> &boxes, std::vector<Ort::Value> &output_tensors)
{
    if (image.empty() || !is_inited)
        return false;
    //cv::Mat using_image = &image;
    // auto logger = spdlog::get("file_logger");
    //  图片预处理
    try
    {
        this->preprocess(image);
    }
    catch (const std::exception &e)
    {
        return false;
    }
    // // ******************************yolo推理**************************************
    // std::vector<Ort::Value> yolo_input_tensor;
    // yolo_input_tensor.push_back(Ort::Value::CreateTensor<float>(
    //                     memory_info,
    //                     this->input_images[0].ptr<float>(),
    //                     this->input_images[0].total(),
    //                     this->yolo_input_nodes[0].dim.data(),
    //                     this->yolo_input_nodes[0].dim.size())
    // );
    // std::vector<cv::Rect> boxes = this->yolo_infer(yolo_input_tensor);
    // if(boxes.empty()) return "yolo can not detect any bbox!";
    // // return 1;

    //*******************************encoder推理***********************************

    if(boxes.empty())
    {
        spdlog::info("boxes is empty");
        return false;
    }
    spdlog::info("staring encoder reasoning");
    this->encoder_input_nodes[0].dim = {1, 3, image.rows, image.cols};
    std::vector<Ort::Value> encoder_input_tensor;
    encoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        this->input_images[0].ptr<float>(),
        this->input_images[0].total(),
        this->encoder_input_nodes[0].dim.data(), // 3*1024*1024
        this->encoder_input_nodes[0].dim.size()));
    Ort::Value image_embeddings = std::move(this->encoder_infer(encoder_input_tensor).at(0));

    // Ort::Value image_embeddings=Ort::Value::CreateTensor<float>(
    //         memory_info,
    //         this->input_images[0].ptr<float>(),
    //         this->input_images[0].total(),
    //         this->encoder_input_nodes[0].dim.data(), // 3*1024*1024
    //         this->encoder_input_nodes[0].dim.size());

    spdlog::info("staring decoder reasoning");
    //*******************************decoder推理***********************************
    std::vector<std::vector<float>> points_cord;
    // std::vector<cv::Rect> boxes = this->yolo_infer(yolo_input_tensor);
    for (auto &box : boxes)
    {
        // Rectangle coordinates of detected object: left, right, top, down
        //[0][1]为左上角x,y[2][3]为右下角x,y
        std::vector<float> point_val{(float)box.rect[0], (float)box.rect[1], (float)box.rect[2], (float)box.rect[3]}; // xyxy
        points_cord.emplace_back(point_val);
    }
    // 创建一个占位batched_point_coords,实际并不起作用
    ////[0][1]为左上角x,y[2][3]为右下角x,y
    std::vector<float> point_val{(float)boxes[0].rect[0], (float)boxes[0].rect[1], (float)boxes[0].rect[2], (float)boxes[0].rect[3]}; // xyxy
    auto batched_point_coords = Ort::Value::CreateTensor<float>(
        memory_info,
        point_val.data(),
        point_val.size(),
        this->decoder_input_nodes[1].dim.data(),
        this->decoder_input_nodes[1].dim.size());
    // 添加batched_point_labels
    std::vector<float> point_labels = {2, 3};
    auto batched_point_labels = Ort::Value::CreateTensor<float>(
        memory_info,
        point_labels.data(),
        point_labels.size(),
        this->decoder_input_nodes[2].dim.data(),
        this->decoder_input_nodes[2].dim.size());
    // 添加orig_im_size
    std::vector<int64> img_size{image.rows, image.cols}; // h,w
    auto orig_im_size = Ort::Value::CreateTensor<int64>(
        memory_info,
        img_size.data(),
        img_size.size(),
        this->decoder_input_nodes[3].dim.data(),
        this->decoder_input_nodes[3].dim.size());

    // std::vector<Ort::Value> output_tensors;
    output_tensors.clear();
    std::vector<Ort::Value> input_tensors;
    // 移交所有权
    input_tensors.emplace_back(std::move(image_embeddings));
    input_tensors.emplace_back(std::move(batched_point_coords));
    input_tensors.emplace_back(std::move(batched_point_labels));
    input_tensors.emplace_back(std::move(orig_im_size));
    try
    {
        for (auto &points : points_cord)
        {
            input_tensors[1] = Ort::Value::CreateTensor<float>(
                memory_info,
                points.data(),
                points.size(),
                this->decoder_input_nodes[1].dim.data(),
                this->decoder_input_nodes[1].dim.size());
            output_tensors.emplace_back(std::move(this->decoder_infer(input_tensors).at(0)));
        }
    }
    catch (const std::exception &e)
    {
        spdlog::info("decoder_infer failed!!");
        return false;
    }
    //***********************输出后处理**************************************
    // spdlog::info("")
    return true;
}

std::vector<Ort::Value> SAM::encoder_infer(std::vector<Ort::Value> &input_tensor)
{
    std::vector<const char *> input_names, output_names;
    for (auto &node : this->encoder_input_nodes)
        input_names.push_back(node.name);
    for (auto &node : this->encoder_output_nodes)
        output_names.push_back(node.name);
    //*******************************推理******************************
    std::vector<Ort::Value> output_tensors;
    try
    {
        output_tensors = this->encoder_session->Run(
            Ort::RunOptions{nullptr}, // 默认
            input_names.data(),       // 输入节点的所有名字
            input_tensor.data(),      // 输入tensor
            input_tensor.size(),      // 输入tensor的数量
            output_names.data(),      // 输出节点的所有名字
            output_names.size()       // 输出节点名字的数量
        );
    }
    catch (const std::exception &e)
    {
        spdlog::info("forward encoder model failed!!");
    }
    return std::move(output_tensors);
}
std::vector<Ort::Value> SAM::decoder_infer(std::vector<Ort::Value> &input_tensor)
{
    std::vector<const char *> input_names, output_names;
    for (auto &node : this->decoder_input_nodes)
        input_names.push_back(node.name);
    for (auto &node : this->decoder_output_nodes)
        output_names.push_back(node.name);
    //*******************************推理******************************
    std::vector<Ort::Value> output_tensors;
    try
    {
        output_tensors = this->decoder_session->Run(
            Ort::RunOptions{nullptr}, // 默认
            input_names.data(),       // 输入节点的所有名字
            input_tensor.data(),      // 输入tensor
            input_tensor.size(),      // 输入tensor的数量
            output_names.data(),      // 输出节点的所有名字
            output_names.size()       // 输出节点名字的数量
        );
    }
    catch (const std::exception &e)
    {
        spdlog::info("forward decoder model failed!!");
    }
    return std::move(output_tensors);
}
