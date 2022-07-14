//#pragma once
//#include "Blurer.h"
//
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/dnn.hpp>
//
//#include <tesseract/baseapi.h>
//#include <leptonica/allheaders.h>
//
//
//#include <string>
//#include <vector>
//#include <memory>
//#include <iostream>
//#include <thread>
//#include <chrono>
//#include <algorithm>
//
//#include <cstdint>
//
//using namespace core_api;
//
//class FrameBlurer
//{
//public:
//    ~FrameBlurer() { m_ocr->End(); }
//
//    void init(const char* text_detector, const char* text_reader);
//
//    cv::Mat forward(cv::Mat frame);
//private:
//    static constexpr float confThreshold = 0.975f;
//    static constexpr float nmsThreshold = 0.6f;
//    static constexpr int inpWidth = 480;
//    static constexpr int inpHeight = 480;
//
//    std::unique_ptr<cv::dnn::Net> m_text_finder;
//    std::unique_ptr< tesseract::TessBaseAPI> m_ocr;
//
//    static void decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh,
//        std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences);
//};
//
//
//void FrameBlurer::init(const char* text_detector, const char* text_reader)
//{
//    cv::String model = text_detector;
//    try
//    {
//        m_text_finder = std::make_unique<cv::dnn::Net>(cv::dnn::readNet(model));
//    }
//    catch (const cv::Exception& e)
//    {
//        std::cerr << e.what() << std::endl;
//    }
//
//    m_ocr = std::make_unique<tesseract::TessBaseAPI>(tesseract::TessBaseAPI());
//    if (m_ocr->Init(text_reader ? text_reader : NULL, "eng", tesseract::OEM_LSTM_ONLY))
//    {
//        std::cerr << "Could not initialize tesseract.\n";
//    }
//}
//
//
//cv::Mat FrameBlurer::forward(cv::Mat frame)
//{
//    cv::Mat frame_copy = frame;
//
//    std::vector<cv::Mat> output;
//    std::vector<cv::String> outputLayers(2);
//    outputLayers[0] = "feature_fusion/Conv_7/Sigmoid";
//    outputLayers[1] = "feature_fusion/concat_3";
//
//    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(inpWidth, inpHeight));
//    m_text_finder->setInput(blob);
//    m_text_finder->forward(output, outputLayers);
//
//    cv::Mat scores = output[0];
//    cv::Mat geometry = output[1];
//
//    std::vector<cv::RotatedRect> boxes;
//    std::vector<float> confidences;
//    decode(scores, geometry, confThreshold, boxes, confidences);
//
//    std::vector<int> indices;
//    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
//
//    cv::Point2f ratio((float)frame.cols / inpWidth, (float)frame.rows / inpHeight);
//    std::vector<DetectedRect> detected;
//    for (auto index : indices)
//    {
//        cv::Rect bbox = boxes[index].boundingRect();
//        cv::Rect normalized_bbox = cv::Rect{ int((float)bbox.x * ratio.x), int((float)bbox.y * ratio.y), int((float)bbox.width * ratio.x), int((float)bbox.height * ratio.y) };
//        std::string out_text;
//
//        detected.push_back({ normalized_bbox,  out_text });
//    }
//
//    m_ocr->SetImage(frame.data, frame.cols, frame.rows, 3, frame.step);
//    for (auto& rect : detected)
//    {
//        m_ocr->SetRectangle(rect.bbox.x, rect.bbox.y, rect.bbox.width, rect.bbox.height);
//        m_ocr->SetSourceResolution(2000);
//
//        rect.text = m_ocr->GetUTF8Text();
//    }
//   
//    cv::Mat blurred = frame_copy;
//    for (auto& [region, text] : detected)
//    {
//        cv::Mat blured_region;
//        try
//        {
//            cv::GaussianBlur(frame_copy(region), blured_region, cv::Size(0, 0), 4);
//
//            blured_region.copyTo(blurred(region));
//        }
//        catch (std::exception& e)
//        {
//            std::cerr << e.what() << std::endl;
//        }
//    }
//
//    return blurred;
//}
//
//
//
//void FrameBlurer::decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh, std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences)
//{
//    detections.clear();
//    CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
//    CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
//    CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);
//
//    const int height = scores.size[2];
//    const int width = scores.size[3];
//    for (int y = 0; y < height; ++y)
//    {
//        const float* scoresData = scores.ptr<float>(0, 0, y);
//        const float* x0_data = geometry.ptr<float>(0, 0, y);
//        const float* x1_data = geometry.ptr<float>(0, 1, y);
//        const float* x2_data = geometry.ptr<float>(0, 2, y);
//        const float* x3_data = geometry.ptr<float>(0, 3, y);
//        const float* anglesData = geometry.ptr<float>(0, 4, y);
//        for (int x = 0; x < width; ++x)
//        {
//            float score = scoresData[x];
//            if (score < scoreThresh)
//                continue;
//
//            float offsetX = x * 4.0f, offsetY = y * 4.0f;
//            float angle = anglesData[x];
//            float cosA = std::cos(angle);
//            float sinA = std::sin(angle);
//            float h = x0_data[x] + x2_data[x];
//            float w = x1_data[x] + x3_data[x];
//
//            cv::Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
//                offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
//            cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
//            cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
//            cv::RotatedRect r(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
//            detections.push_back(r);
//            confidences.push_back(score);
//        }
//    }
//}