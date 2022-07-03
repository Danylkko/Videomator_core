#pragma once

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <string>
#include <string_view>
#include <array>
#include <memory>


#ifdef _WIN64
#ifdef BUILD_DLL

#define EXPORT __declspec(dllexport)

#else

#define EXPORT __declspec(dllimport)

#endif
#else

#define #define EXPORT
#endif

extern "C++"
{


	namespace core_api
	{

		struct DetectedRect
		{
			cv::Rect bbox;
			std::string text;
		};


		class Blurer
		{
		public:
			enum class detection_mode { all, license_plates_only };

			void init();

			void load(std::string_view filepath);

			const std::vector<DetectedRect>& detect(detection_mode mode = detection_mode::all);
			inline const std::vector<DetectedRect>& currently_detected() const { return m_currently_detected; }

			void add_exceptions(const std::vector<DetectedRect>& exceptions);

			void load_blurred_to_buffer(size_t frame_index = 0);

			inline const cv::Mat& matrix_buffer() const { return m_buffer; }
			const std::vector<char> buffer() const;

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

	}

}