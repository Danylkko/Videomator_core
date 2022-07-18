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

		struct Detective
		{
			int top;
			int left;
			int width;
			int heigh;
			const char* text;
		};

		struct image_data
		{
			unsigned char* data;
			int width;
			int height;

			Detective* detectives;
			unsigned int detectives_size;
		};

		using OnFrameCallback = void (*)(image_data);

		class EXPORT Blurer
		{
		public:
			Blurer(); 
			~Blurer();

			enum class detection_mode { all, license_plates_only };

			void init(const char* model_data = "frozen_inference_graph.pb", const char* model_format = "frozen_inference.pbtxt", const char* tesseract_data_path = nullptr);

			void load(const char* filepath); 
			int get_fps();
			int get_frame_count();

			bool done_rendering();
			float rendering_progress();

			void start_render(detection_mode mode = detection_mode::all);

			void add_exeption(const char* text);
			void remove_exeption(const char* text);

			void create_stream(unsigned int frame_index = 0);
			void play_stream(int fps);
			void pause_stream();

			image_data stream_buffer() const;
			image_data stream_buffer_preview() const;

			void set_on_update_callback(OnFrameCallback callback);
			void reset_on_update_callback();

			void stream_load_next();
			void stream_load_next_wait();

			void save_rendered(const char* filepath);

		private:
			class BlurerImpl;

			BlurerImpl* m_impl;
		};
	}
EXTERN_END