#include "Blurer.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

//#include <openvino/openvino.hpp>
//
//#include <models/detection_model.h>
//#include <models/detection_model_ssd.h>
//#include <models/detection_model_yolo.h>
//#include <pipelines/async_pipeline.h>
//#include <pipelines/metadata.h>
//#include <utils/args_helper.hpp>
//#include <utils/common.hpp>
//#include <utils/config_factory.h>
//#include <utils/default_flags.hpp>
//#include <utils/images_capture.h>
//#include <utils/ocv_common.hpp>
//#include <utils/performance_metrics.hpp>
//#include <utils/slog.hpp>


#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>

#include <cstdint>


using namespace core_api;


struct core_api::DetectedRect
{
    cv::Rect bbox;
    std::string text;

    bool empty();
};

bool core_api::DetectedRect::empty()
{
    return text == "empty";
}

class Processor
{
public:
    static Processor& instance();

    cv::Mat blur(const std::vector<DetectedRect>& detections, cv::Mat frame);

    void add_exeption(const char* text);
    void remove_exeption(const char* text);
private:
    std::vector<std::string> m_exceptions;
};

Processor& Processor::instance()
{
    static Processor instance;
    return instance;
}

void Processor::add_exeption(const char* text)
{
    std::string text_s = text;
    if (std::find(m_exceptions.cbegin(), m_exceptions.cend(), text_s) != m_exceptions.end())
    {
        m_exceptions.emplace_back(std::move(text_s));
    }
}

void Processor::remove_exeption(const char* text)
{
    std::string text_s = text;
    auto iter = std::find(m_exceptions.cbegin(), m_exceptions.cend(), text_s);
    if (iter != m_exceptions.end())
    {
        m_exceptions.erase(iter);
    }
}

cv::Mat Processor::blur(const std::vector<DetectedRect>& detections, cv::Mat frame)
{
    cv::Mat blurred = frame;
    for (auto& [region, text] : detections)
    {
        if (std::find(m_exceptions.begin(), m_exceptions.end(), text) != m_exceptions.end())
            continue;
        cv::Mat blured_region;
        if (region.area() != 0 && region.tl().y > 0 && region.tl().x > 0 && region.tl().y < frame.rows && region.tl().x < frame.cols)
        {
            try
            {
                //std::cout << region << std::endl;
                cv::GaussianBlur(frame(region), blured_region, cv::Size(0, 0), region.width / 10, region.height / 10);

                blured_region.copyTo(blurred(region));
            }
            catch (std::exception& e)
            {
                std::cerr << e.what() << std::endl;
            }
        }

    }
    return blurred;
}




class FrameBlurer
{
public:
    static FrameBlurer& render_instance();
    static FrameBlurer& preview_instance();

    ~FrameBlurer() { m_ocr->End(); }

    void init(const char* model_data, const char* model_format, const char* tesseract_data_path);

    std::vector<DetectedRect> forward(cv::Mat frame, Blurer::detection_mode mode, std::vector<DetectedRect> prev_frame_data = std::vector<DetectedRect>());
private:
    FrameBlurer() = default;

    static constexpr float confThreshold = 0.85f;
    static constexpr float nmsThreshold = 0.6f;
    static constexpr int inpWidth = 1200;
    static constexpr int inpHeight = 1200;
    static constexpr int outLength = 100;
    static constexpr int outWidth = 7;

    std::unique_ptr<cv::dnn::Net> m_text_finder;
    std::unique_ptr<tesseract::TessBaseAPI> m_ocr;

    std::mutex m_lock;

    static void decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh,
        std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences);
};

FrameBlurer& FrameBlurer::render_instance()
{
    static FrameBlurer inst;
    return inst;
}

FrameBlurer& FrameBlurer::preview_instance()
{
    static FrameBlurer inst;
    return inst;
}

class VideoRenderer
{
public:
    inline void init_blurer(const char* model_data, const char* model_format, const char* tesseract_data_path) {
        FrameBlurer::render_instance().init(model_data, model_format, tesseract_data_path);
        FrameBlurer::preview_instance().init(model_data, model_format, tesseract_data_path);
    };

    void set_source(cv::VideoCapture& capture, std::string capture_source);
    void reset();

    inline const std::string& capture_source()const { return m_source_path; }

    double get_source_fps() const { return m_video->get(cv::CAP_PROP_FPS); }
    int get_source_frames() const { return static_cast<int>(m_video->get(cv::CAP_PROP_FRAME_COUNT)); }

    void render_async(Blurer::detection_mode mode);

    inline const std::vector<std::vector<DetectedRect>>& frames() const { return m_proccesed_frames; }

    void save(const char* path);

    void wait_render_finish();

    static std::mutex frames_lock;
private:
    static constexpr uint32_t num_render_threads = 4;

    
    std::vector<std::vector<DetectedRect>> m_proccesed_frames;
    //std::vector<cv::Mat>::const_iterator m_selected_frame;
    std::atomic_bool m_render_active = false;

    cv::VideoCapture* m_video = nullptr;
    std::string m_source_path;

    std::vector<std::thread> m_rendering_threads;

    void render_impl(Blurer::detection_mode mode, uint32_t frame_count, uint32_t offset);
};
std::mutex VideoRenderer::frames_lock = std::mutex();

class VideoStream
{
public:
    VideoStream(VideoRenderer& rend, int index);

   // inline VideoStream(std::vector<cv::Mat>::const_iterator start_frame, std::vector<cv::Mat>::const_iterator end) :m_iter(start_frame), m_end(end) { m_buffer = *m_iter; }
    inline ~VideoStream() { pause(); }

    image_data buffer();
    image_data buffer_preview();
    
    void set_callback(OnFrameCallback callback);
    inline void remove_callback() { m_callback_fn = nullptr; }

    void load_next_frame();
    void load_next_frame_wait();

    void play(uint32_t fps);

    void pause();

private:
    std::vector<DetectedRect> m_buffer;
    cv::Mat m_img_buffer;

    cv::VideoCapture m_capture;

    //size_t index;
    std::vector<std::vector<DetectedRect>>::const_iterator m_iter;
    std::vector<std::vector<DetectedRect>>::const_iterator m_end;

    OnFrameCallback m_callback_fn = nullptr;

    std::mutex m_buffer_lock;

    std::unique_ptr<std::thread> m_running_thread = nullptr;

    bool m_play = false;

    void increment_iterator(std::chrono::milliseconds wait_time);
};

VideoStream::VideoStream(VideoRenderer& rend, int index) :m_iter(rend.frames().cbegin() + index), m_end(rend.frames().cend()) 
{ 
    std::scoped_lock lokc(VideoRenderer::frames_lock);
    m_buffer = *m_iter;
    m_capture.open(rend.capture_source());

    m_capture.set(cv::CAP_PROP_POS_FRAMES, index);
    m_capture >> m_img_buffer;
    m_capture.set(cv::CAP_PROP_POS_FRAMES, index);
}


void VideoStream::load_next_frame()
{
    using namespace std::chrono_literals;
    std::scoped_lock lock1(VideoRenderer::frames_lock);
    std::scoped_lock lock2(m_buffer_lock);


    if (m_iter == m_end)
        return;
    if (m_iter->size() != 0)
    {
        m_buffer = *m_iter;
        m_capture >> m_img_buffer;
        //index++;
        m_iter++;
    }
}

void VideoStream::load_next_frame_wait()
{
    using namespace std::chrono_literals;
    while (m_iter->size() == 0)
        std::this_thread::sleep_for(20ms);

    load_next_frame();
}

void VideoStream::set_callback(OnFrameCallback callback)
{
    m_callback_fn = callback;
}

void VideoStream::play(uint32_t fps)
{
    m_play = true;
    auto increment_rate = std::chrono::milliseconds(1000 / fps);
    m_running_thread = std::make_unique<std::thread>(&VideoStream::increment_iterator, this, increment_rate);
}

void VideoStream::pause()
{
    m_play = false;
    if(m_running_thread)
        m_running_thread->join();
}

void VideoStream::increment_iterator(std::chrono::milliseconds wait_time)
{
    while (m_iter != m_end && m_play)
    {
        load_next_frame_wait();
        if (m_callback_fn)
        {
            cv::Mat blurred = Processor::instance().blur(m_buffer, m_img_buffer);
            m_callback_fn(image_data{ blurred.data, blurred.cols, blurred.rows });
        }
    }
}






void VideoRenderer::set_source(cv::VideoCapture& capture, std::string capture_source)
{
    m_render_active = false;
    for (auto& thread : m_rendering_threads)
        thread.join();
    m_rendering_threads.clear();

    m_proccesed_frames.clear();

    m_video = &capture;
    m_source_path = capture_source;
}

void VideoRenderer::reset()
{

}


class core_api::Blurer::BlurerImpl
{
public:
    void init(const char* model_data, const char* model_format, const char* tesseract_data_path);

    void load(const char* filepath);
    inline int get_fps()const { return m_renderer.get_source_fps(); }
    inline int get_frame_count()const { return m_renderer.get_source_frames(); }

    float rendering_progres() const;
    bool done_rendering() const;

    void start_render(detection_mode mode);

    void start_stream(unsigned int frame_index);
    void play_stream(int fps);
    void pause_stream();

    inline void set_on_update_callback(OnFrameCallback callback) { m_stream->set_callback(callback); }
    inline void reset__on_update_callback() { m_stream->remove_callback(); }

    inline void stream_load_next() { m_stream->load_next_frame(); }
    inline void stream_load_next_wait() { m_stream->load_next_frame_wait(); }

    core_api::image_data buffer() const;
    core_api::image_data buffer_preview() const;

    void save_rendered_impl(const char* filepath);

private:
    cv::VideoCapture m_capture;

    VideoRenderer m_renderer;

    std::unique_ptr<VideoStream> m_stream = nullptr;
};

bool core_api::Blurer::BlurerImpl::done_rendering() const
{
    int frames_done = std::count_if(m_renderer.frames().begin(), m_renderer.frames().end(), [](const std::vector<DetectedRect>& frame_data) { return frame_data.size()>0; });
    return frames_done == m_renderer.frames().size();
}

float core_api::Blurer::BlurerImpl::rendering_progres() const
{
    int frames_done = std::count_if(m_renderer.frames().begin(), m_renderer.frames().end(), [](const std::vector<DetectedRect>& frame_data) { return frame_data.size() > 0; });
    return (float)frames_done / m_renderer.frames().size();
}


void core_api::Blurer::BlurerImpl::init(const char* model_data, const char* model_format, const char* tesseract_data_path)
{
    m_renderer.init_blurer(model_data, model_format, tesseract_data_path);
}

void core_api::Blurer::BlurerImpl::start_render(detection_mode mode)
{
    m_renderer.render_async(mode);
}

void core_api::Blurer::BlurerImpl::load(const char* filepath)
{
    m_stream.reset();

    m_capture.open(filepath);
    if (!m_capture.isOpened())
    {
        throw std::runtime_error("FAILED loading file at" + std::string(filepath));
    }
    m_renderer.set_source(m_capture, filepath);
}


void core_api::Blurer::BlurerImpl::start_stream(unsigned int frame_index)
{
    m_stream = std::make_unique<VideoStream>(m_renderer, frame_index);
}

void core_api::Blurer::BlurerImpl::play_stream(int fps)
{
    //m_renderer.wait_render_finish();
    m_stream->play(fps);
}

void core_api::Blurer::BlurerImpl::pause_stream()
{
    m_stream->pause();
}


core_api::image_data core_api::Blurer::BlurerImpl::buffer() const
{
    return m_stream->buffer();
}

core_api::image_data core_api::Blurer::BlurerImpl::buffer_preview() const
{
    return m_stream->buffer_preview();
}






void core_api::Blurer::BlurerImpl::save_rendered_impl(const char* filepath)
{
    m_renderer.wait_render_finish();
    m_renderer.save(filepath);
}



core_api::Blurer::Blurer() : m_impl(new BlurerImpl()) {}

core_api::Blurer::~Blurer()
{
    delete m_impl;
}

void core_api::Blurer::init(const char* model_data, const char* model_format, const char* tesseract_data_path)
{
    m_impl->init(model_data, model_format, tesseract_data_path);
}

void core_api::Blurer::load(const char* filepath)
{
    m_impl->load(filepath);
}

int core_api::Blurer::get_fps()
{
    return m_impl->get_fps();
}

int core_api::Blurer::get_frame_count()
{
    return m_impl->get_frame_count();
}

bool core_api::Blurer::done_rendering()
{
    return m_impl->done_rendering();
}

float core_api::Blurer::rendering_progress()
{
    return m_impl->rendering_progres();
}

void core_api::Blurer::start_render(detection_mode mode)
{
    m_impl->start_render(mode);
}

void core_api::Blurer::add_exeption(const char* text)
{
    Processor::instance().add_exeption(text);
}

void core_api::Blurer::remove_exeption(const char* text)
{
    Processor::instance().remove_exeption(text);
}

void core_api::Blurer::create_stream(unsigned int frame_index)
{
    m_impl->start_stream(frame_index);
}




void core_api::Blurer::play_stream(int fps)
{
    m_impl->play_stream(fps);
}

void core_api::Blurer::pause_stream()
{
    m_impl->pause_stream();
}


image_data core_api::Blurer::stream_buffer() const
{
    return m_impl->buffer();
}

image_data core_api::Blurer::stream_buffer_preview() const
{
    return m_impl->buffer_preview();
}


void core_api::Blurer::set_on_update_callback(OnFrameCallback callback)
{
    m_impl->set_on_update_callback(callback);
}

void core_api::Blurer::reset_on_update_callback()
{
    m_impl->reset__on_update_callback();
}

void core_api::Blurer::stream_load_next()
{
    m_impl->stream_load_next();
}

void core_api::Blurer::stream_load_next_wait()
{
    m_impl->stream_load_next_wait();
}


void core_api::Blurer::save_rendered(const char* filepath)
{
    m_impl->save_rendered_impl(filepath);
}


void FrameBlurer::init(const char* model_data, const char* model_format, const char* tesseract_data_path)
{
    try
    {
        m_text_finder = std::make_unique<cv::dnn::Net>(cv::dnn::readNet(model_data, model_format));
        //m_text_finder->setPreferableBackend(cv::dnn::DNN_TARGET_GPU);
    }
    catch (const cv::Exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    m_ocr = std::make_unique<tesseract::TessBaseAPI>(tesseract::TessBaseAPI());
    if (m_ocr->Init(tesseract_data_path ? tesseract_data_path : NULL, "eng", tesseract::OEM_LSTM_ONLY))
    {
        std::cerr << "Could not initialize tesseract.\n";
    }
}


std::vector<DetectedRect> FrameBlurer::forward(cv::Mat frame, Blurer::detection_mode mode, std::vector<DetectedRect> prev_frame_data)
{
    cv::Mat frame_copy = frame;
    std::vector<cv::Mat> output;

    //std::lock_guard<std::mutex> lock(m_lock);
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true);
    m_text_finder->setInput(blob);
    try
    {
        m_text_finder->forward(output);

    }
    catch (cv::Exception e)
    {
        std::cout << e.what() << std::endl;
    }
   
    float* out = output[0].ptr<float>(0,0);
    
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    for (int i = 0; i < outLength; i++)
    {
        int scale = outWidth;
        float score = out[scale * i + 2];

        float left = out[scale * i + 3] * frame.cols;
        float top = out[scale * i + 4] * frame.rows;
        float right = out[scale * i + 5] * frame.cols;
        float bottom = out[scale * i + 6] * frame.rows;
        
        float width = std::abs(left - right);
        float height = std::abs(top - bottom);
        cv::Rect box{ (int)left, (int)top, (int)width, (int)height };

        for (auto& [rect, text] : prev_frame_data)
        {
            int dist = cv::norm(rect.tl() - box.tl());
            float area_ratio = width * height / rect.area();

            if (dist < 25)
            {
                score += 0.5;
            }
        }
        score = score > 1 ? 1 : score;
        scores.push_back(score);
        boxes.emplace_back(box);
    }

    //decode(scores, geometry, confThreshold, boxes, confidences);

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confThreshold, nmsThreshold, indices);

    cv::Point2f ratio((float)frame.cols / inpWidth, (float)frame.rows / inpHeight);
    std::vector<DetectedRect> detected;
    for (auto index : indices)
    {
        
        cv::Rect bbox = boxes[index];
        //cv::Rect normalized_bbox = cv::Rect{ int((float)bbox.x * ratio.x), int((float)bbox.y * ratio.y), int((float)bbox.width * ratio.x), int((float)bbox.height * ratio.y) };
        std::string out_text;

        //std::cout << bbox << std::endl;

        detected.push_back({ bbox,  out_text });
    }


    m_ocr->SetImage(frame.data, frame.cols, frame.rows, 3, frame.step);
    for (auto& rect : detected)
    {
        if (rect.bbox.x >= 0 && rect.bbox.y >= 0 && rect.bbox.x + rect.bbox.width <= frame.cols && rect.bbox.y + rect.bbox.height <= frame.rows && rect.bbox.width > 0 && rect.bbox.height > 0)
        {
            m_ocr->SetRectangle(rect.bbox.x, rect.bbox.y, rect.bbox.width, rect.bbox.height);
            m_ocr->SetSourceResolution(2000);


            rect.text = m_ocr->GetUTF8Text();
        }
    }
    if (detected.size() == 0)
        detected.push_back(DetectedRect{ cv::Rect(0,0,0,0), "empty" });
    return detected;
}

void FrameBlurer::decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh, std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences)
{
}



void VideoRenderer::render_impl(Blurer::detection_mode mode, uint32_t frame_count, uint32_t offset)
{
    for (unsigned int i = 0; i < frame_count && m_render_active; i++)
    {
        if (i == 132)
            std::cout << "???" << std::endl;
        cv::Mat frame;
        *m_video >> frame;
        std::lock_guard lock{ frames_lock };
        if (i == 0)
            m_proccesed_frames[i + offset] = (FrameBlurer::render_instance().forward(frame, mode));
        else
            m_proccesed_frames[i + offset] = (FrameBlurer::render_instance().forward(frame, mode, m_proccesed_frames[i - 1 + offset]));
        std::cout << i << std::endl;
    }
}

void VideoRenderer::render_async(Blurer::detection_mode mode)
{
    m_render_active = true;
    int frame_count = get_source_frames();
    m_proccesed_frames = std::vector<std::vector<DetectedRect>>(frame_count);

    m_rendering_threads.emplace_back(&VideoRenderer::render_impl, this, mode, frame_count, 0);
    //m_proccesed_frames = std::vector<cv::Mat>(frame_count);

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
}




void VideoRenderer::save(const char* path)
{
    wait_render_finish();
    using namespace std::chrono_literals;
    cv::VideoWriter output(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), get_source_fps(), cv::Size(m_video->get(cv::CAP_PROP_FRAME_WIDTH), m_video->get(cv::CAP_PROP_FRAME_HEIGHT)));
    m_video->set(cv::CAP_PROP_POS_FRAMES, 0);

    for (auto& frame_data : m_proccesed_frames)
    {
        cv::Mat frame;
        *m_video >> frame;
        frame = Processor::instance().blur(frame_data, frame);
        output.write(frame);
    }
    //for (auto& thread : m_rendering_threads)
    //    thread.join();

    m_rendering_threads.clear();
}

void VideoRenderer::wait_render_finish()
{
    for (auto& thread : m_rendering_threads)
    {
        if(thread.joinable())
            thread.join();
    }
        
}

image_data VideoStream::buffer()
{
    std::lock_guard<std::mutex> guard(m_buffer_lock);
    
    cv::Mat frame = Processor::instance().blur(m_buffer, m_img_buffer);
    #ifndef _WIN32
    if (!frame.empty())
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    #endif

    unsigned int size = m_buffer.size();
    core_api::Detective* dets = new Detective[size];
    for (int i = 0; i < size; i++)
    {
        char* text = new char[m_buffer[i].text.size()];
        strcpy(text, m_buffer[i].text.c_str());
        dets[i] = Detective{ m_buffer[i].bbox.tl().y,  m_buffer[i].bbox.tl().x, m_buffer[i].bbox.width, m_buffer[i].bbox.height, text };
    }
    return { frame.data, frame.cols, frame.rows, dets, size };
}

image_data VideoStream::buffer_preview()
{
    std::lock_guard lock(VideoRenderer::frames_lock);

    std::vector<DetectedRect> detections = m_buffer.size()==0 ? FrameBlurer::preview_instance().forward(m_img_buffer, core_api::Blurer::detection_mode::all) : m_buffer;
    //auto detections = FrameBlurer::preview_instance().forward(m_img_buffer, core_api::Blurer::detection_mode::all);

    cv::Mat frame = Processor::instance().blur(detections, m_img_buffer);
#ifndef _WIN32
    if (!frame.empty())
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
#endif

    unsigned int size = detections.size();
    //core_api::Detective* dets = new Detective[size];
    //for (int i = 0; i < size; i++)
    //{
    //    char* text = new char[detections[i].text.size()];
    //    strcpy(text, detections[i].text.c_str());
    //    dets[i] = Detective{ detections[i].bbox.tl().y,  detections[i].bbox.tl().x, detections[i].bbox.width, detections[i].bbox.height, text };
    //}
    //return { frame.data, frame.cols, frame.rows, dets, size };
    return{ frame.data, frame.cols, frame.rows, nullptr, size };
}
