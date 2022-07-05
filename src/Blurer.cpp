#include "Blurer.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <iostream>

using namespace core_api;


struct core_api::DetectedRect
{
    cv::Rect bbox;
    std::string text;
};


class core_api::Blurer::BlurerImpl
{
public:
    void init();

    void load(std::string_view filepath);

    const std::vector<DetectedRect>& detect(Blurer::detection_mode mode = Blurer::detection_mode::all);
    const std::vector<DetectedRect>& currently_detected() const;

    void add_exceptions(const std::vector<DetectedRect>& exceptions);

    void load_blurred_to_buffer(size_t frame_index = 0);

    //inline const cv::Mat& matrix_buffer() const { return m_buffer; }
    image_data buffer() const;


private:
    static constexpr float confThreshold = 0.7;
    static constexpr float nmsThreshold = 0.4;
    static constexpr int inpWidth = 1280;
    static constexpr int inpHeight = 1280;


    cv::VideoCapture m_capture;
    std::unique_ptr<cv::dnn::Net> m_text_finder;
    std::unique_ptr< tesseract::TessBaseAPI> m_ocr;//TODO: custom deleter that calls TessBaseAPI::End()

    cv::Mat m_current_frame;
    cv::Mat m_buffer;

    std::vector<DetectedRect> m_currently_detected;



    static void decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh,
        std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences);
};


void core_api::Blurer::BlurerImpl::init()
{
    cv::String model = "frozen_east_text_detection.pb";
    m_text_finder = std::make_unique<cv::dnn::Net>(cv::dnn::readNet(model));

    m_ocr = std::make_unique<tesseract::TessBaseAPI>(tesseract::TessBaseAPI());
    m_ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
}

void core_api::Blurer::BlurerImpl::load(std::string_view filepath)
{
    m_capture.open(filepath.data());
    m_capture >> m_current_frame;

    m_ocr->SetImage(m_current_frame.data, m_current_frame.cols, m_current_frame.rows, 3, m_current_frame.step);
}

const std::vector<DetectedRect>& core_api::Blurer::BlurerImpl::detect(detection_mode mode)
{
    std::vector<cv::Mat> output;
    std::vector<cv::String> outputLayers(2);
    outputLayers[0] = "feature_fusion/Conv_7/Sigmoid";
    outputLayers[1] = "feature_fusion/concat_3";

    cv::Mat blob = cv::dnn::blobFromImage(m_current_frame, 1.0, cv::Size(inpWidth, inpHeight), cv::Scalar(123.68, 116.78, 103.94), true, false);
    m_text_finder->setInput(blob);
    m_text_finder->forward(output, outputLayers);

    cv::Mat scores = output[0];
    cv::Mat geometry = output[1];

    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;
    decode(scores, geometry, confThreshold, boxes, confidences);

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	
    cv::Point2f ratio((float)m_current_frame.cols / inpWidth, (float)m_current_frame.rows / inpHeight);
    std::vector<DetectedRect> detected;
    for (auto index : indices)
    {
        cv::Rect bbox = boxes[index].boundingRect();
        cv::Rect normalized_bbox = cv::Rect{ int((float)bbox.x * ratio.x), int((float)bbox.y * ratio.y), int((float)bbox.width * ratio.x), int((float)bbox.height * ratio.y) };


        m_ocr->SetRectangle(normalized_bbox.x, normalized_bbox.y, normalized_bbox.width, normalized_bbox.height);
        m_ocr->SetSourceResolution(2000);

        //std::string outText = m_ocr->GetUTF8Text();
        detected.push_back({ normalized_bbox, m_ocr->GetUTF8Text() });
    }

    m_currently_detected = std::move(detected);
    return m_currently_detected;
}

void core_api::Blurer::BlurerImpl::add_exceptions(const std::vector<DetectedRect>& exceptions)
{
    //TODO)
}

void core_api::Blurer::BlurerImpl::load_blurred_to_buffer(size_t frame_index)
{
    m_buffer = m_current_frame;
    for (auto&[region, text] : m_currently_detected)
    {
        cv::Mat blured_region;
        cv::GaussianBlur(m_current_frame(region), blured_region, cv::Size(0, 0), 4);

        
        blured_region.copyTo(m_buffer(region));
    }
}


image_data core_api::Blurer::BlurerImpl::buffer() const
{
    std::vector<uint8_t> res;
    res.assign(m_buffer.data, m_buffer.data + m_buffer.total() * m_buffer.channels());
    return { res, m_buffer.cols, m_buffer.rows };
}



void core_api::Blurer::BlurerImpl::decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh, std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences)
{
    detections.clear();
    CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
    CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

    const int height = scores.size[2];
    const int width = scores.size[3];
    for (int y = 0; y < height; ++y)
    {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x)
        {
            float score = scoresData[x];
            if (score < scoreThresh)
                continue;

            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            cv::Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
            cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
            cv::RotatedRect r(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}




core_api::Blurer::Blurer() : m_impl(new BlurerImpl()) {}

core_api::Blurer::~Blurer()
{
    delete m_impl;
}

void core_api::Blurer::init()
{
    m_impl->init();
}

void core_api::Blurer::load(const char* filepath)
{
    m_impl->load(filepath);
}

void core_api::Blurer::detect(detection_mode mode)
{
    m_impl->detect(mode);
}

//const std::vector<DetectedRect>& core_api::Blurer::currently_detected() const
//{
//    return m_impl->currently_detected();
//}

void core_api::Blurer::add_exceptions(const std::vector<DetectedRect>& exceptions)
{
    m_impl->add_exceptions(exceptions);
}

void core_api::Blurer::load_blurred_to_buffer(size_t frame_index)
{
    m_impl->load_blurred_to_buffer(frame_index);
}

image_data core_api::Blurer::buffer() const
{
    return m_impl->buffer();
}
