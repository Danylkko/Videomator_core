#include "Blurer.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>


#include <string>
#include <string_view>
#include <array>
#include <vector>
#include <memory>
#include <iostream>
#include <thread>
#include <chrono>

#include <cstdint>


using namespace core_api;


struct core_api::DetectedRect
{
    cv::Rect bbox;
    std::string text;
};


class FrameBlurer
{
public:
    ~FrameBlurer() { m_ocr->End(); }

    void init(const char* text_detector, const char* text_reader);


    cv::Mat forward(cv::Mat frame, Blurer::detection_mode mode);
private:
    static constexpr float confThreshold = 0.1;
    static constexpr float nmsThreshold = 0.0;
    static constexpr int inpWidth = 320;
    static constexpr int inpHeight = 320;

    std::unique_ptr<cv::dnn::Net> m_text_finder;
    std::unique_ptr< tesseract::TessBaseAPI> m_ocr;

    static void decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh,
        std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences);
};

void FrameBlurer::init(const char* text_detector, const char* text_reader)
{
    cv::String model = text_detector;
    try
    {
        m_text_finder = std::make_unique<cv::dnn::Net>(cv::dnn::readNet(model));
    }
    catch (const cv::Exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    m_ocr = std::make_unique<tesseract::TessBaseAPI>(tesseract::TessBaseAPI());
    if (m_ocr->Init(text_reader? text_reader : NULL, "eng", tesseract::OEM_LSTM_ONLY))
    {
        std::cerr << "Could not initialize tesseract.\n";
        exit(1);
    }
}


cv::Mat FrameBlurer::forward(cv::Mat frame, Blurer::detection_mode mode)
{
    cv::Mat frame_copy = frame;


    std::vector<cv::Mat> output;
    std::vector<cv::String> outputLayers(2);
    outputLayers[0] = "feature_fusion/Conv_7/Sigmoid";
    outputLayers[1] = "feature_fusion/concat_3";

    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(inpWidth, inpHeight));
    m_text_finder->setInput(blob);
    m_text_finder->forward(output, outputLayers);

    cv::Mat scores = output[0];
    cv::Mat geometry = output[1];

    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;
    decode(scores, geometry, confThreshold, boxes, confidences);

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    
    cv::Point2f ratio((float)frame.cols / inpWidth, (float)frame.rows / inpHeight);
    std::vector<DetectedRect> detected;
    for (auto index : indices)
    {
        cv::Rect bbox = boxes[index].boundingRect();
        cv::Rect normalized_bbox = cv::Rect{ int((float)bbox.x * ratio.x), int((float)bbox.y * ratio.y), int((float)bbox.width * ratio.x), int((float)bbox.height * ratio.y) };
        std::string out_text;

        

        detected.push_back({ normalized_bbox,  out_text });
    }

    if (mode != Blurer::detection_mode::all)
    {
       /* m_ocr->SetImage(frame.data, frame.cols, frame.rows, 3, frame.step);
        for (auto& rect : detected)
        {
            m_ocr->SetRectangle(rect.bbox.x, rect.bbox.y, rect.bbox.width, rect.bbox.height);
            m_ocr->SetSourceResolution(2000);

            rect.text = m_ocr->GetUTF8Text();
        }*/
    }



    cv::Mat blurred = frame_copy;
    for (auto& [region, text] : detected)
    {
        cv::Mat blured_region;
        try
        {
            cv::GaussianBlur(frame_copy(region), blured_region, cv::Size(0, 0), 4);

            blured_region.copyTo(blurred(region));
        }
        catch (std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    return blurred;
}

class VideoStream
{
public:

private:

    std::vector<cv::Mat>::const_iterator m_begin;
    std::vector<cv::Mat>::const_iterator m_end;

};

class VideoRenderer
{
public:
    inline void init_blurer(const char* text_detector, const char* text_reader = nullptr) { m_blurer.init(text_detector, text_reader); }
    inline void set_source(cv::VideoCapture& capture) { m_video = &capture; }

    void render_async(Blurer::detection_mode mode);
    
    cv::Mat get_frame(int32_t index = -1);

    void save(const char* path);
private:
    static constexpr uint32_t num_render_threads = 4;

    std::mutex m_frames_lock;
    std::vector<cv::Mat> m_proccesed_frames;
    std::vector<cv::Mat>::const_iterator m_selected_frame;

    FrameBlurer m_blurer;

    cv::VideoCapture* m_video = nullptr;

    cv::VideoWriter m_writer;

    std::vector<std::thread> m_rendering_threads;

    void render_impl(Blurer::detection_mode mode, uint32_t frame_count, uint32_t offset);

    void wait_render_finish();
};

void VideoRenderer::render_impl(Blurer::detection_mode mode, uint32_t frame_count, uint32_t offset)
{
    for (int i = 0; i < frame_count; i++)
    {
        cv::Mat frame;
        *m_video >> frame;
        std::lock_guard lock{ m_frames_lock };
        m_proccesed_frames[i + offset] = (m_blurer.forward(frame, mode));
    }
}

void VideoRenderer::render_async(Blurer::detection_mode mode)
{
    int frame_count = m_video->get(cv::CAP_PROP_FRAME_COUNT);
    m_proccesed_frames = std::vector<cv::Mat>(frame_count);

 /*   int frames_per_thread = frame_count / num_render_threads;
    int remainder = frame_count % num_render_threads;

    for (int i = 0; i < num_render_threads; i++)
    {
        if (i == num_render_threads - 1)
            frames_per_thread += remainder;

        std::vector<cv::Mat> thread_frames;
        thread_frames.reserve(frames_per_thread);
        for (int j = 0; i < frames_per_thread; j++)
        {
            thread_frames.push_back(

        )
        }

        m_rendering_threads.emplace_back(&VideoRenderer::render_impl, this, mode, frames_per_thread, i * frames_per_thread);
    }*/

    m_rendering_threads.emplace_back(&VideoRenderer::render_impl, this, mode, frame_count, 0);
}

cv::Mat VideoRenderer::get_frame(int32_t index)
{
    try
    {
        cv::Mat frame;
        if ((frame = m_proccesed_frames.at(index) ).empty())
        {
            wait_render_finish();
        }
    }
    catch (std::out_of_range& e)
    {
        std::cerr << std::string("ERROR getting frame") + std::to_string(index) + std::string("rendering hasn't started yet!");
    }
    return m_proccesed_frames.at(index);
}


void VideoRenderer::save(const char* path)
{   
    wait_render_finish();
    using namespace std::chrono_literals;
    cv::VideoWriter output(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), m_video->get(cv::CAP_PROP_FPS), cv::Size(m_video->get(cv::CAP_PROP_FRAME_WIDTH), m_video->get(cv::CAP_PROP_FRAME_HEIGHT)));

    for (auto& frame : m_proccesed_frames)
    {
        
        while (frame.empty())
        {

            std::this_thread::sleep_for(10ms);

        }

        output.write(frame);
    }
    //for (auto& thread : m_rendering_threads)
    //    thread.join();

    m_rendering_threads.clear();
}

void VideoRenderer::wait_render_finish()
{
    for (auto& thread : m_rendering_threads)
        thread.join();
}

class core_api::Blurer::BlurerImpl
{
public:
    void init(const char* east_path, const char* tesseract_data_path);

    void load(const char* filepath);

    void start_render(detection_mode mode);

    const std::vector<DetectedRect>& detect(Blurer::detection_mode mode = Blurer::detection_mode::all);
    const std::vector<DetectedRect>& currently_detected() const;

    void add_exceptions(const std::vector<DetectedRect>& exceptions);

    void load_blurred_to_buffer(size_t frame_index = 0);

    //inline const cv::Mat& matrix_buffer() const { return m_buffer; }
    core_api::image_data buffer() const;

    void save_rendered_impl(const char* filepath);

private:
    //static constexpr float confThreshold = 0.7;
    //static constexpr float nmsThreshold = 0.4;
    //static constexpr int inpWidth = 1280;
    //static constexpr int inpHeight = 1280;


    cv::VideoCapture m_capture;

    VideoRenderer m_renderer;
    //std::unique_ptr<cv::dnn::Net> m_text_finder;
    //std::unique_ptr< tesseract::TessBaseAPI> m_ocr;//TODO: custom deleter that calls TessBaseAPI::End()

    //cv::Mat m_current_frame;
    



    cv::Mat m_buffer;

    //std::vector<DetectedRect> m_currently_detected;
};


void core_api::Blurer::BlurerImpl::init(const char* east_path, const char* tesseract_data_path)
{
    //cv::String model = "frozen_east_text_detection.pb";
    //try
    //{
    //    m_text_finder = std::make_unique<cv::dnn::Net>(cv::dnn::readNet(model));
    //}
    //catch (const cv::Exception& e)
    //{
    //    std::cerr << e.what() << std::endl;
    //}
    //
    // 

    //m_ocr = std::make_unique<tesseract::TessBaseAPI>(tesseract::TessBaseAPI());
    ////m_ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);

    //if (m_ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY)) 
    //{
    //    std::cerr << "Could not initialize tesseract.\n";
    //    exit(1);
    //}
    m_renderer.init_blurer(east_path, tesseract_data_path);
}

void core_api::Blurer::BlurerImpl::start_render(detection_mode mode)
{
    m_renderer.render_async(mode);
}

void core_api::Blurer::BlurerImpl::load(const char* filepath)
{
    m_capture.open(filepath);
    if (!m_capture.isOpened())
    {
        throw std::runtime_error("FAILED loading file at" + std::string(filepath));
    }

    m_renderer.set_source(m_capture);
    //m_ocr->SetImage(m_current_frame.data, m_current_frame.cols, m_current_frame.rows, 3, m_current_frame.step);
}

const std::vector<DetectedRect>& core_api::Blurer::BlurerImpl::detect(detection_mode mode)
{
    //std::vector<cv::Mat> output;
    //std::vector<cv::String> outputLayers(2);
    //outputLayers[0] = "feature_fusion/Conv_7/Sigmoid";
    //outputLayers[1] = "feature_fusion/concat_3";

    //cv::Mat blob = cv::dnn::blobFromImage(m_current_frame, 1.0, cv::Size(inpWidth, inpHeight), cv::Scalar(123.68, 116.78, 103.94), true, false);
    //m_text_finder->setInput(blob);
    //m_text_finder->forward(output, outputLayers);

    //cv::Mat scores = output[0];
    //cv::Mat geometry = output[1];

    //std::vector<cv::RotatedRect> boxes;
    //std::vector<float> confidences;
    //decode(scores, geometry, confThreshold, boxes, confidences);

    //std::vector<int> indices;
    //cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	
    //cv::Point2f ratio((float)m_current_frame.cols / inpWidth, (float)m_current_frame.rows / inpHeight);
    //std::vector<DetectedRect> detected;
    //for (auto index : indices)
    //{
    //    cv::Rect bbox = boxes[index].boundingRect();
    //    cv::Rect normalized_bbox = cv::Rect{ int((float)bbox.x * ratio.x), int((float)bbox.y * ratio.y), int((float)bbox.width * ratio.x), int((float)bbox.height * ratio.y) };


    //    m_ocr->SetRectangle(normalized_bbox.x, normalized_bbox.y, normalized_bbox.width, normalized_bbox.height);
    //    m_ocr->SetSourceResolution(2000);

    //    std::string outText = m_ocr->GetUTF8Text();
    //    detected.push_back({ normalized_bbox,  outText });
    //}

    //m_currently_detected = std::move(detected);
    //return m_currently_detected;
    return std::vector<DetectedRect>();
}

void core_api::Blurer::BlurerImpl::add_exceptions(const std::vector<DetectedRect>& exceptions)
{
    //TODO)
}

void core_api::Blurer::BlurerImpl::load_blurred_to_buffer(size_t frame_index)
{
    m_buffer = m_renderer.get_frame(0);
    //for (auto&[region, text] : m_currently_detected)
    //{
    //    cv::Mat blured_region;
    //    try
    //    {
    //        cv::GaussianBlur(m_current_frame(region), blured_region, cv::Size(0, 0), 4);


    //        blured_region.copyTo(m_buffer(region));
    //    }
    //    catch (std::exception& e)
    //    {
    //        std::cout << "\n\n\n\n\n\n\n";
    //        std::cout << e.what() << std::endl;
    //        std::cout << "\n\n\n\n\n\n\n";
    //    }
    //    
    //}
}


core_api::image_data core_api::Blurer::BlurerImpl::buffer() const
{
    return { m_buffer.data, m_buffer.cols, m_buffer.rows };
}



void FrameBlurer::decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh, std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences)
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


void core_api::Blurer::BlurerImpl::save_rendered_impl(const char* filepath)
{
    m_renderer.save(filepath);
}



core_api::Blurer::Blurer() : m_impl(new BlurerImpl()) {}

core_api::Blurer::~Blurer()
{
    delete m_impl;
}

void core_api::Blurer::init(const char* east_path, const char* tesseract_data_path)
{
    m_impl->init(east_path, tesseract_data_path);
}

void core_api::Blurer::load(const char* filepath)
{
    m_impl->load(filepath);
}

void core_api::Blurer::start_render(detection_mode mode)
{
    m_impl->start_render(mode);
}

//void core_api::Blurer::detect(detection_mode mode)
//{
//    m_impl->detect(mode);
//}
//
////const std::vector<DetectedRect>& core_api::Blurer::currently_detected() const
////{
////    return m_impl->currently_detected();
////}
//
//void core_api::Blurer::add_exceptions(const DetectedRect* exceptions, unsigned int size)
//{
//    std::vector<DetectedRect> excepts(size);
//    for (int i = 0; i < size; i++)
//        excepts.push_back(exceptions[i]);
//
//    m_impl->add_exceptions(excepts);
//}
//
void core_api::Blurer::create_stream(unsigned int frame_index)
{

    m_impl->load_blurred_to_buffer(frame_index);
}



image_data core_api::Blurer::stream_buffer() const
{
    return m_impl->buffer();
}

void core_api::Blurer::save_rendered(const char* filepath)
{
    m_impl->save_rendered_impl(filepath);
}