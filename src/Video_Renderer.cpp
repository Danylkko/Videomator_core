#pragma once 
//#include "Blurer.h"
//
//#include "Frame_Blurer.cpp"//questionable
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
//class VideoRenderer
//{
//public:
//    inline void init_blurer(const char* text_detector, const char* text_reader = nullptr) { m_blurer.init(text_detector, text_reader); }
//    void set_source(cv::VideoCapture& capture);
//    void reset();
//
//    double get_source_fps() const { return m_video->get(cv::CAP_PROP_FPS); }
//    int get_source_frames() const { return static_cast<int>(m_video->get(cv::CAP_PROP_FRAME_COUNT)); }
//
//    void render_async();
//
//    inline const std::vector<cv::Mat>& frames() const { return m_proccesed_frames; }
//
//    void save(const char* path);
//
//    void wait_render_finish();
//private:
//    static constexpr uint32_t num_render_threads = 4;
//
//    std::mutex m_frames_lock;
//    std::vector<cv::Mat> m_proccesed_frames;
//    //std::vector<cv::Mat>::const_iterator m_selected_frame;
//    bool m_render_active = false;
//
//    FrameBlurer m_blurer;
//
//    cv::VideoCapture* m_video = nullptr;
//
//    cv::VideoWriter m_writer;
//
//    std::vector<std::thread> m_rendering_threads;
//
//    void render_impl(uint32_t frame_count, uint32_t offset);
//
//
//};
//
//
//void VideoRenderer::set_source(cv::VideoCapture& video)
//{
//    m_render_active = false;
//    for (auto& thread : m_rendering_threads)
//        thread.join();
//
//    m_proccesed_frames.clear();
//    m_writer.release();
//
//    m_video = &video;
//}
//void VideoRenderer::reset()
//{
//
//}
//
//
//void VideoRenderer::render_impl(uint32_t frame_count, uint32_t offset)
//{
//    for (unsigned int i = 0; i < frame_count && m_render_active; i++)
//    {
//        cv::Mat frame;
//        *m_video >> frame;
//        std::lock_guard lock{ m_frames_lock };
//        m_proccesed_frames[i + offset] = (m_blurer.forward(frame));
//    }
//}
//
//void VideoRenderer::render_async()
//{
//
//    m_render_active = true;
//    int frame_count = get_source_frames();
//    m_proccesed_frames = std::vector<cv::Mat>(frame_count);
//
//    /*   int frames_per_thread = frame_count / num_render_threads;
//       int remainder = frame_count % num_render_threads;
//
//       for (int i = 0; i < num_render_threads; i++)
//       {
//           if (i == num_render_threads - 1)
//               frames_per_thread += remainder;
//
//           std::vector<cv::Mat> thread_frames;
//           thread_frames.reserve(frames_per_thread);
//           for (int j = 0; i < frames_per_thread; j++)
//           {
//               thread_frames.push_back(
//
//           )
//           }
//
//           m_rendering_threads.emplace_back(&VideoRenderer::render_impl, this, mode, frames_per_thread, i * frames_per_thread);
//       }*/
//
//    m_rendering_threads.emplace_back(&VideoRenderer::render_impl, this, frame_count, 0);
//}
//
//void VideoRenderer::save(const char* path)
//{
//
//    using namespace std::chrono_literals;
//    cv::VideoWriter output(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), get_source_fps(), cv::Size(m_video->get(cv::CAP_PROP_FRAME_WIDTH), m_video->get(cv::CAP_PROP_FRAME_HEIGHT)));
//
//    for (auto& frame : m_proccesed_frames)
//    {
//
//        while (frame.empty())
//        {
//            std::this_thread::sleep_for(10ms);
//        }
//
//        output.write(frame);
//    }
//    //for (auto& thread : m_rendering_threads)
//    //    thread.join();
//
//    m_rendering_threads.clear();
//}
//
//void VideoRenderer::wait_render_finish()
//{
//    for (auto& thread : m_rendering_threads)
//        thread.join();
//}