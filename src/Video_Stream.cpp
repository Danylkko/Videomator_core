//#pragma once
//
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
//class VideoStream
//{
//public:
//    inline VideoStream(std::vector<cv::Mat>::const_iterator start_frame, std::vector<cv::Mat>::const_iterator end) :m_iter(start_frame), m_end(end) { m_buffer = *m_iter; }
//    inline ~VideoStream() { pause(); }
//
//    const cv::Mat& buffer();
//
//
//    void play(uint32_t fps);
//
//    void pause();
//
//private:
//    cv::Mat m_buffer;
//    std::mutex buffer_lock;
//
//    std::vector<cv::Mat>::const_iterator m_iter;
//    std::vector<cv::Mat>::const_iterator m_end;
//
//
//    std::unique_ptr<std::thread> m_running_thread = nullptr;
//
//    bool m_play = false;
//
//    void increment_iterator(std::chrono::milliseconds wait_time);
//};
//
//void VideoStream::play(uint32_t fps)
//{
//    m_play = true;
//    auto increment_rate = std::chrono::milliseconds(1000 / fps);
//    m_running_thread = std::make_unique<std::thread>(&VideoStream::increment_iterator, this, increment_rate);
//}
//
//void VideoStream::pause()
//{
//    m_play = false;
//    m_running_thread->join();
//}
//
//void VideoStream::increment_iterator(std::chrono::milliseconds wait_time)
//{
//    while (m_iter != m_end && m_play)
//    {
//        if (!m_iter->empty())
//        {
//            std::lock_guard guard(buffer_lock);
//            m_buffer = *m_iter;
//            m_iter++;
//
//        }
//        std::this_thread::sleep_for(wait_time);
//    }
//}
//
//
//const cv::Mat& VideoStream::buffer()
//{
//    std::lock_guard<std::mutex> guard(buffer_lock);
//#ifndef _WIN32
//    cv::cvtColor(m_buffer, m_buffer, cv::COLOR_BGR2RGB);
//#endif
//    return m_buffer;
//}
