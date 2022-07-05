#pragma once

#include <string>
#include <string_view>
#include <array>
#include <vector>
#include <memory>

#include <cstdint>


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
		struct DetectedRect;

		struct image_data
		{
			std::vector<uint8_t> data;
			int32_t width;
			int32_t height;
		};


		class EXPORT Blurer
		{
		public:
			Blurer(); 
			~Blurer();

			enum class detection_mode { all, license_plates_only };

			void init();

			void load(std::string filepath); 

			void detect(detection_mode mode = detection_mode::all); 
			//const std::vector<DetectedRect>& currently_detected() const; 

			void add_exceptions(const std::vector<DetectedRect>& exceptions); 

			void load_blurred_to_buffer(size_t frame_index = 0);

			//inline const cv::Mat& matrix_buffer() const { return m_impl->matrix_buffer(); }
			image_data buffer() const;

		private:
			class BlurerImpl;

			BlurerImpl* m_impl;
		};
	}

}