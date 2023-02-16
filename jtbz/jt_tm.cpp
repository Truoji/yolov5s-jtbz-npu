/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: xwwang@openailab.com
 * Author: stevenwudi@fiture.com
 * 
 * original model: https://github.com/ultralytics/yolov5
 */

#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>


#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

using namespace cv;
using namespace std;


struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};
//初始化一个sigmoid函数
static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}
//定义一个交叉区域
static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
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
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
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
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

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

static void generate_proposals(int stride, const float* feat, float prob_threshold, std::vector<Object>& objects,
                               int letterbox_cols, int letterbox_rows)
{
    static float anchors[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};

    int anchor_num = 3;
    int feat_w = letterbox_cols / stride;
    int feat_h = letterbox_rows / stride;
    int cls_num = 45;
    int anchor_group;
    if (stride == 8)
        anchor_group = 1;
    if (stride == 16)
        anchor_group = 2;
    if (stride == 32)
        anchor_group = 3;
    for (int h = 0; h <= feat_h - 1; h++)
    {
        for (int w = 0; w <= feat_w - 1; w++)
        {
            for (int a = 0; a <= anchor_num - 1; a++)
            {
                //process cls score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int s = 0; s <= cls_num - 1; s++)
                {
                    float score = feat[a * feat_w * feat_h * (cls_num + 5) + h * feat_w * (cls_num + 5) + w * (cls_num + 5) + s + 5];
                    if (score > class_score)
                    {
                        class_index = s;
                        class_score = score;
                    }
                }
                //process box score
                float box_score = feat[a * feat_w * feat_h * (cls_num + 5) + (h * feat_w) * (cls_num + 5) + w * (cls_num + 5) + 4];
                float final_score = sigmoid(box_score) * sigmoid(class_score);
                if (final_score >= prob_threshold)
                {
                    int loc_idx = a * feat_h * feat_w * (cls_num + 5) + h * feat_w * (cls_num + 5) + w * (cls_num + 5);
                    float dx = sigmoid(feat[loc_idx + 0]);
                    float dy = sigmoid(feat[loc_idx + 1]);
                    float dw = sigmoid(feat[loc_idx + 2]);
                    float dh = sigmoid(feat[loc_idx + 3]);
                    float pred_cx = (dx * 2.0f - 0.5f + w) * stride;
                    float pred_cy = (dy * 2.0f - 0.5f + h) * stride;
                    float anchor_w = anchors[(anchor_group - 1) * 6 + a * 2 + 0];
                    float anchor_h = anchors[(anchor_group - 1) * 6 + a * 2 + 1];
                    float pred_w = dw * dw * 4.0f * anchor_w;
                    float pred_h = dh * dh * 4.0f * anchor_h;
                    float x0 = pred_cx - pred_w * 0.5f;
                    float y0 = pred_cy - pred_h * 0.5f;
                    float x1 = pred_cx + pred_w * 0.5f;
                    float y1 = pred_cy + pred_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = final_score;
                    objects.push_back(obj);
                }
            }
        }
    }
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "pl80", "p6", "p5", "pm55", "pl60", "ip", "p11", "i2r", "p23", "pg",
        "il80", "ph4", "i4", "pl70", "pne", "ph4.5", "p12", "p3", "pl5", "w13",
        "i4l", "pl30", "p10", "pn", "w55", "p26", "p13", "pr40", "pl20", "pm30",
        "pl40", "i2", "pl120", "w32", "ph5", "il60", "w57", "pl100",
        "w59", "il100", "p19", "pm20", "i5", "p27", "pl50"};
    
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
                obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));

        /*cout << class_names[obj.label] << endl;
        result_file << class_names[obj.label] << "\n";*/
    }

    //cv::imwrite("yolov5_timvx_letterbox_out.jpg", image);
}

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

void get_input_data_focus_uint8(Mat img, uint8_t* input_data, int letterbox_rows, int letterbox_cols, const float* mean,
                                const float* scale, float input_scale, int zero_point)
{
    //cv::Mat sample = cv::imread(image_file, 1);
    //cv::Mat img;

    //if (sample.channels() == 1)
    //    cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    //else
    //    cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);

    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((letterbox_rows * 1.0 / img.rows) < (letterbox_cols * 1.0 / img.cols))
    {
        scale_letterbox = letterbox_rows * 1.0 / img.rows;
    }
    else
    {
        scale_letterbox = letterbox_cols * 1.0 / img.cols;
    }
    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    cv::resize(img, img, cv::Size(resize_cols, resize_rows));
    img.convertTo(img, CV_32FC3);
    // Generate a gray image for letterbox using opencv
    cv::Mat img_new(letterbox_cols, letterbox_rows, CV_32FC3, cv::Scalar(0.5 / scale[0] + mean[0], 0.5 / scale[1] + mean[1], 0.5 / scale[2] + mean[2]));
    int top = (letterbox_rows - resize_rows) / 2;
    int bot = (letterbox_rows - resize_rows + 1) / 2;
    int left = (letterbox_cols - resize_cols) / 2;
    int right = (letterbox_cols - resize_cols + 1) / 2;
    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    img_new.convertTo(img_new, CV_32FC3);
    float* img_data = (float*)img_new.data;
    std::vector<float> input_temp(3 * letterbox_cols * letterbox_rows);

    /* nhwc to nchw */
    for (int h = 0; h < letterbox_rows; h++)
    {
        for (int w = 0; w < letterbox_cols; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index = h * letterbox_cols * 3 + w * 3 + c;
                int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                input_temp[out_index] = (img_data[in_index] - mean[c]) * scale[c];
            }
        }
    }

    /* focus process */
    for (int i = 0; i < 2; i++) // corresponding to rows
    {
        for (int g = 0; g < 2; g++) // corresponding to cols
        {
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < letterbox_rows / 2; h++)
                {
                    for (int w = 0; w < letterbox_cols / 2; w++)
                    {
                        int in_index = i + g * letterbox_cols + c * letterbox_cols * letterbox_rows + h * 2 * letterbox_cols + w * 2;
                        int out_index = i * 2 * 3 * (letterbox_cols / 2) * (letterbox_rows / 2) + g * 3 * (letterbox_cols / 2) * (letterbox_rows / 2) + c * (letterbox_cols / 2) * (letterbox_rows / 2) + h * (letterbox_cols / 2) + w;

                        /* quant to uint8 */
                        int udata = (round)(input_temp[in_index] / input_scale + (float)zero_point);
                        if (udata > 255)
                            udata = 255;
                        else if (udata < 0)
                            udata = 0;

                        input_data[out_index] = udata;
                    }
                }
            }
        }
    }
}


//void get_input_data_yolo(Mat img, uint8_t* input_data, int img_h, int img_w)
//{
//    // Resize the size so that it can suitable for model's input size
//    resize(img, img, cv::Size(img_h, img_w));
//    // Convert image data from uint8 to float
//    img.convertTo(img, CV_32FC3);
//    float* img_data = (float*)img.data;
//    int hw = img_h * img_w;
//    // Pre processing for mobilenet_ssd input data
//    //float mean[3] = {127.5, 127.5, 127.5};
//    float mean[3] = { 0.f, 0.f, 0.f };
//    for (int h = 0; h < img_h; h++)
//    {
//        for (int w = 0; w < img_w; w++)
//        {
//            for (int c = 0; c < 3; c++)
//            {
//                int index = c * hw + h * img_w + w;
//                //input_data[index] = 1.0f / 255 * (img_data[index] - mean[c]);
//                input_data[index] = 1.0f / 255 * (*img_data - mean[c]);
//                //input.data[index] = (resImg.data[index] - means[c]) * scale[c];
//                img_data++;
//            }
//        }
//    }
//    image swaprgb_img = { 0 };
//    swaprgb_img.c = 3;
//    swaprgb_img.w = img_w;
//    swaprgb_img.h = img_h;
//    swaprgb_img.data = input_data;
//    rgb2bgr_permute(swaprgb_img);
//}



int main(int argc, char* argv[])
{
    const char* model_file = nullptr;
    //const char* image_file = nullptr;
    int img_c = 3;
    const float mean[3] = {0, 0, 0};
    const float scale[3] = {0.003921, 0.003921, 0.003921};

    // set default letterbox size
    int letterbox_rows = 640;
    int letterbox_cols = 640;

    int repeat_count = 1;
    int num_thread = 1;

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'i':
            //image_file = optarg;
            break;
        case 'r':
            repeat_count = atoi(optarg);
            break;
        case 't':
            num_thread = atoi(optarg);
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }
    /* check files */
   // if (nullptr == model_file)
   // {
   //     fprintf(stderr, "Error: Tengine model file not specified!\n");
   //     show_usage();
   //     return -1;
   // }

   // /*if (nullptr == image_file)
   // {
   //     fprintf(stderr, "Error: Image file not specified!\n");
   //     show_usage();
   //     return -1;
   // }*/

   //// if (!check_file_exist(model_file) || !check_file_exist(image_file))
   // if (!check_file_exist(model_file))
   //     return -1;


    //cv::Mat img = cv::imread(image_file, 1);
    //if (img.empty())
    //{
    //    fprintf(stderr, "cv::imread %s failed\n", image_file);
    //    return -1;
    //}

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_UINT8;
    opt.affinity = 0;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create VeriSilicon TIM-VX backend */
    context_t timvx_context = create_context("timvx", 1);
    int rtt = set_context_device(timvx_context, "TIMVX", nullptr, 0);
    if (0 > rtt)
    {
        fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
        return -1;
    }

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(timvx_context, "tengine", model_file);
    if (graph == nullptr)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    int img_size = letterbox_rows * letterbox_cols * img_c;
    int dims[] = {1, 12, int(letterbox_rows / 2), int(letterbox_cols / 2)};
    std::vector<uint8_t> input_data(img_size);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data.data(), img_size) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    


    Mat frame;
    VideoCapture capture(1);
   // capture.set(3,1024);
   // capture.set(4,768);

    while (true)
    {
        capture >> frame;
        double start_time = get_current_time();
        
        /* prepare process input data, set the data mem to input tensor */
        float input_scale = 0.f;
        int input_zero_point = 0;
        get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);
        //get_input_data_yolo(frame, input_data.data(), letterbox_rows, letterbox_cols);
        get_input_data_focus_uint8(frame, input_data.data(), letterbox_rows, letterbox_cols, mean, scale, input_scale, input_zero_point);


        //将识别结果输入到/tmp/output文件中
        ofstream result_file;
        if (result_file.bad())
        {
            cout << "cannot open file" << endl;;
        }

        result_file.open("/tmp/output", ios::app);

        /* run graph */

         
        for (int i = 0; i < repeat_count; i++)
        {
            double start = get_current_time();
            if (run_graph(graph, 1) < 0)
            {
                fprintf(stderr, "Run graph failed\n");
                return -1;
            }
        }
        /*fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count, num_thread,
            total_time / repeat_count, max_time, min_time);
        fprintf(stderr, "--------------------------------------\n");*/

        /* yolov5 postprocess */
        // 0: 1, 3, 20, 20, 85
        // 1: 1, 3, 40, 40, 85
        // 2: 1, 3, 80, 80, 85
        tensor_t p8_output = get_graph_output_tensor(graph, 0, 0);
        tensor_t p16_output = get_graph_output_tensor(graph, 1, 0);
        tensor_t p32_output = get_graph_output_tensor(graph, 2, 0);

        /* dequant output data */
        float p8_scale = 0.f;
        float p16_scale = 0.f;
        float p32_scale = 0.f;
        int p8_zero_point = 0;
        int p16_zero_point = 0;
        int p32_zero_point = 0;

        get_tensor_quant_param(p8_output, &p8_scale, &p8_zero_point, 1);
        get_tensor_quant_param(p16_output, &p16_scale, &p16_zero_point, 1);
        get_tensor_quant_param(p32_output, &p32_scale, &p32_zero_point, 1);

        int p8_count = get_tensor_buffer_size(p8_output) / sizeof(uint8_t);
        int p16_count = get_tensor_buffer_size(p16_output) / sizeof(uint8_t);
        int p32_count = get_tensor_buffer_size(p32_output) / sizeof(uint8_t);

        std::vector<float> p8_data(p8_count);
        std::vector<float> p16_data(p16_count);
        std::vector<float> p32_data(p32_count);

        uint8_t* p8_data_u8 = (uint8_t*)get_tensor_buffer(p8_output);
        uint8_t* p16_data_u8 = (uint8_t*)get_tensor_buffer(p16_output);
        uint8_t* p32_data_u8 = (uint8_t*)get_tensor_buffer(p32_output);

        for (int c = 0; c < p8_count; c++)
        {
            p8_data[c] = ((float)p8_data_u8[c] - (float)p8_zero_point) * p8_scale;
        }

        for (int c = 0; c < p16_count; c++)
        {
            p16_data[c] = ((float)p16_data_u8[c] - (float)p16_zero_point) * p16_scale;
        }

        for (int c = 0; c < p32_count; c++)
        {
            p32_data[c] = ((float)p32_data_u8[c] - (float)p32_zero_point) * p32_scale;
        }

        /* postprocess */
        const float prob_threshold = 0.27f;
        const float nms_threshold = 0.45f;

        std::vector<Object> proposals;
        std::vector<Object> objects8;
        std::vector<Object> objects16;
        std::vector<Object> objects32;
        std::vector<Object> objects;

        generate_proposals(32, p32_data.data(), prob_threshold, objects32, letterbox_cols, letterbox_rows);
        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        generate_proposals(16, p16_data.data(), prob_threshold, objects16, letterbox_cols, letterbox_rows);
        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        generate_proposals(8, p8_data.data(), prob_threshold, objects8, letterbox_cols, letterbox_rows);
        proposals.insert(proposals.end(), objects8.begin(), objects8.end());

        qsort_descent_inplace(proposals);
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);

        /* yolov5 draw the result */

        float scale_letterbox;
        int resize_rows;
        int resize_cols;
        if ((letterbox_rows * 1.0 / frame.rows) < (letterbox_cols * 1.0 / frame.cols))
        {
            scale_letterbox = letterbox_rows * 1.0 / frame.rows;
        }
        else
        {
            scale_letterbox = letterbox_cols * 1.0 / frame.cols;
        }
        resize_cols = int(scale_letterbox * frame.cols);
        resize_rows = int(scale_letterbox * frame.rows);

        int tmp_h = (letterbox_rows - resize_rows) / 2;
        int tmp_w = (letterbox_cols - resize_cols) / 2;

        float ratio_x = (float)frame.rows / resize_rows;
        float ratio_y = (float)frame.cols / resize_cols;

        int count = picked.size();
        //fprintf(stderr, "detection num: %d\n", count);

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];
            float x0 = (objects[i].rect.x);
            float y0 = (objects[i].rect.y);
            float x1 = (objects[i].rect.x + objects[i].rect.width);
            float y1 = (objects[i].rect.y + objects[i].rect.height);

            x0 = (x0 - tmp_w) * ratio_x;
            y0 = (y0 - tmp_h) * ratio_y;
            x1 = (x1 - tmp_w) * ratio_x;
            y1 = (y1 - tmp_h) * ratio_y;

            x0 = std::max(std::min(x0, (float)(frame.cols - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(frame.rows - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(frame.cols - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(frame.rows - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;

            //draw_objects(frame, objects);
        /*    cout << objects << endl;*/
            
        }
        draw_objects(frame, objects);

        imshow("yolo", frame);
        result_file.close();
        if (char(waitKey(1)) == 'q')
            break;
        double end_time = get_current_time();
        double total_time = end_time - start_time;
        //cout << total_time << endl;
        
    }


    

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
}
