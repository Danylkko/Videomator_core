#include "Blurer.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>


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
};


class FrameBlurer
{
public:
    ~FrameBlurer() { m_ocr->End(); }

    void init(const char* text_detector, const char* text_reader);

    cv::Mat forward(cv::Mat frame, Blurer::detection_mode mode);
private:
    static constexpr float confThreshold = 0.975f;
    static constexpr float nmsThreshold = 0.6f;
    static constexpr int inpWidth = 480;
    static constexpr int inpHeight = 480;

    std::unique_ptr<cv::dnn::Net> m_text_finder;
    std::unique_ptr< tesseract::TessBaseAPI> m_ocr;

    static void decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh,
        std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences);
};


class VideoStream
{
public:
    inline VideoStream(std::vector<cv::Mat>::const_iterator start_frame, std::vector<cv::Mat>::const_iterator end) :m_iter(start_frame), m_end(end) { m_buffer = *m_iter; }
    inline ~VideoStream() { pause(); }

    const cv::Mat& buffer();
    
    void set_callback(OnFrameCallback callback);
    inline void remove_callback() { m_callback_fn = nullptr; }

    void load_next_frame();
    void load_next_frame_wait();

    void play(uint32_t fps);

    void pause();

private:
    cv::Mat m_buffer;
    std::mutex buffer_lock;

    std::vector<cv::Mat>::const_iterator m_iter;
    std::vector<cv::Mat>::const_iterator m_end;

    OnFrameCallback m_callback_fn = nullptr;


    std::unique_ptr<std::thread> m_running_thread = nullptr;

    bool m_play = false;

    void increment_iterator(std::chrono::milliseconds wait_time);
};

void VideoStream::load_next_frame()
{
    if (!(m_iter + 1)->empty())
    {
        m_iter++;

        std::lock_guard guard(buffer_lock);
        m_buffer = *m_iter;
    }
}

void VideoStream::load_next_frame_wait()
{
    using namespace std::chrono_literals;
    if (m_iter == m_end - 1)
        return;
    while ((m_iter + 1)->empty())
        std::this_thread::sleep_for(20ms);

    m_iter++;
    std::lock_guard guard(buffer_lock);
    m_buffer = *m_iter;
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
    m_running_thread->join();
}

void VideoStream::increment_iterator(std::chrono::milliseconds wait_time)
{
    while (m_iter != m_end && m_play)
    {
        if (!m_iter->empty())
        {
            if (m_callback_fn)
            {
                m_callback_fn(image_data{ m_iter->data, m_iter->cols, m_iter->rows });
            }

            std::lock_guard guard(buffer_lock);
            m_buffer = *m_iter;
            m_iter++;
        }
        std::this_thread::sleep_for(wait_time);
    }
}



class VideoRenderer
{
public:
    inline void init_blurer(const char* text_detector, const char* text_reader = nullptr) { m_blurer.init(text_detector, text_reader); }
    void set_source(cv::VideoCapture& capture);
    void reset();

    double get_source_fps() const { return m_video->get(cv::CAP_PROP_FPS); }
    int get_source_frames() const { return static_cast<int>(m_video->get(cv::CAP_PROP_FRAME_COUNT)); }

    void render_async(Blurer::detection_mode mode);
    
    inline const std::vector<cv::Mat>& frames() const { return m_proccesed_frames; }

    void save(const char* path);

    void wait_render_finish();
private:
    static constexpr uint32_t num_render_threads = 4;

    std::mutex m_frames_lock;
    std::vector<cv::Mat> m_proccesed_frames;
    //std::vector<cv::Mat>::const_iterator m_selected_frame;
    bool m_render_active = false;

    FrameBlurer m_blurer;

    cv::VideoCapture* m_video = nullptr;

    cv::VideoWriter m_writer;

    std::vector<std::thread> m_rendering_threads;

    void render_impl(Blurer::detection_mode mode, uint32_t frame_count, uint32_t offset);


};


void VideoRenderer::set_source(cv::VideoCapture& video)
{
    m_render_active = false;
    for (auto& thread : m_rendering_threads)
        thread.join();

    m_proccesed_frames.clear();
    m_writer.release();

    m_video = &video;
}

void VideoRenderer::reset()
{

}


class core_api::Blurer::BlurerImpl
{
public:
    void init(const char* east_path, const char* tesseract_data_path);

    void load(const char* filepath);
    inline int get_fps()const { return m_renderer.get_source_fps(); }
    inline int get_frame_count()const { return m_renderer.get_source_frames(); }

    bool done_rendering() const;

    void start_render(detection_mode mode);

    void add_exceptions(const std::vector<DetectedRect>& exceptions);

    void start_stream(size_t frame_index = 0);
    void play_stream(int fps);
    void pause_stream();

    inline void set_on_update_callback(OnFrameCallback callback) { m_stream->set_callback(callback); }
    inline void reset__on_update_callback() { m_stream->remove_callback(); }

    inline void stream_load_next() { m_stream->load_next_frame(); }
    inline void stream_load_next_wait() { m_stream->load_next_frame_wait(); }

    core_api::image_data buffer() const;

    void save_rendered_impl(const char* filepath);

private:
    cv::VideoCapture m_capture;

    VideoRenderer m_renderer;

    std::unique_ptr<VideoStream> m_stream = nullptr;
};

bool core_api::Blurer::BlurerImpl::done_rendering() const
{
    int frames = std::count_if(m_renderer.frames().begin(), m_renderer.frames().end(), [](const cv::Mat& mat) { return !mat.empty(); });
    return frames == get_frame_count();
}


void core_api::Blurer::BlurerImpl::init(const char* east_path, const char* tesseract_data_path)
{
    m_renderer.init_blurer(east_path, tesseract_data_path);
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

    m_renderer.set_source(m_capture);
}


void core_api::Blurer::BlurerImpl::start_stream(size_t frame_index)
{
    m_stream = std::make_unique<VideoStream>(m_renderer.frames().cbegin() + frame_index, m_renderer.frames().cend());
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
    return { m_stream->buffer().data, m_stream->buffer().cols, m_stream->buffer().rows };
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

void core_api::Blurer::init(const char* east_path, const char* tesseract_data_path)
{
    m_impl->init(east_path, tesseract_data_path);
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

void core_api::Blurer::start_render(detection_mode mode)
{
    m_impl->start_render(mode);
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
    if (m_ocr->Init(text_reader ? text_reader : NULL, "eng", tesseract::OEM_LSTM_ONLY))
    {
        std::cerr << "Could not initialize tesseract.\n";
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
         m_ocr->SetImage(frame.data, frame.cols, frame.rows, 3, frame.step);
         for (auto& rect : detected)
         {
             m_ocr->SetRectangle(rect.bbox.x, rect.bbox.y, rect.bbox.width, rect.bbox.height);
             m_ocr->SetSourceResolution(2000);

             rect.text = m_ocr->GetUTF8Text();
         }
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





void VideoRenderer::render_impl(Blurer::detection_mode mode, uint32_t frame_count, uint32_t offset)
{
    for (unsigned int i = 0; i < frame_count && m_render_active; i++)
    {
        cv::Mat frame;
        *m_video >> frame;
        std::lock_guard lock{ m_frames_lock };
        m_proccesed_frames[i + offset] = (m_blurer.forward(frame, mode));
    }
}

void VideoRenderer::render_async(Blurer::detection_mode mode)
{

    m_render_active = true;
    int frame_count = get_source_frames();
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




void VideoRenderer::save(const char* path)
{
    
    using namespace std::chrono_literals;
    cv::VideoWriter output(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), get_source_fps(), cv::Size(m_video->get(cv::CAP_PROP_FRAME_WIDTH), m_video->get(cv::CAP_PROP_FRAME_HEIGHT)));

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

const cv::Mat& VideoStream::buffer()
{
    std::lock_guard<std::mutex> guard(buffer_lock); 
    #ifndef _WIN32
        if(!m_buffer.empty())
            cv::cvtColor(m_buffer, m_buffer, cv::COLOR_BGR2RGB);
    #endif
    return m_buffer;
}
