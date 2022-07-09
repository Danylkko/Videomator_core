#pragma once

#ifdef _WIN64

#define EXTERN_BEGIN extern "C++" {
#define EXTERN_END }

#ifdef BUILD_DLL

#define EXPORT __declspec(dllexport)

#else

#define EXPORT __declspec(dllimport)

#endif
#else

#define EXPORT

#define EXTERN_BEGIN 
#define EXTERN_END 

#endif


EXTERN_BEGIN
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

			void add_exceptions(const DetectedRect* exceptions, unsigned int size); 

			void load_blurred_to_buffer(unsigned int frame_index = 0);

			//inline const cv::Mat& matrix_buffer() const { return m_impl->matrix_buffer(); }
			image_data buffer() const;

		private:
			class BlurerImpl;

			BlurerImpl* m_impl;
		};
	}
EXTERN_END