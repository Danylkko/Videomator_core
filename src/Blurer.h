#pragma once

#ifdef _WIN64
#ifdef BUILD_DLL

#define EXPORT __declspec(dllexport)

#else

#define EXPORT __declspec(dllimport)

#endif
#else

#define EXPORT
#endif


extern "C++"
{
	namespace core_api
	{
		struct DetectedRect;

		struct image_data
		{
			unsigned char* data;
			int width;
			int height;
		};

		class EXPORT Blurer
		{
		public:
			Blurer(); 
			~Blurer();

			enum class detection_mode { all, license_plates_only };

			void init();

			void load(const char* filepath); 

			void detect(detection_mode mode = detection_mode::all); 
			//const std::vector<DetectedRect>& currently_detected() const; 

			void add_exceptions(const DetectedRect* exceptions, size_t size); 

			void load_blurred_to_buffer(size_t frame_index = 0);

			//inline const cv::Mat& matrix_buffer() const { return m_impl->matrix_buffer(); }
			image_data buffer() const;

		private:
			class BlurerImpl;

			BlurerImpl* m_impl;
		};
	}

}