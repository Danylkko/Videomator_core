#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <iostream>
#include <vector>
#include <array>

#include "../../final_core/src/Blurer.h"


void decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh,
    std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences)
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

int test_raw()
{
    float confThreshold = 0.7;
    float nmsThreshold = 0.4;
    int inpWidth = 1280;
    int inpHeight = 1280;

    cv::String model = "frozen_east_text_detection.pb";

    CV_Assert(!model.empty());

    cv::dnn::Net net = cv::dnn::readNet(model);

    cv::VideoCapture cap;
    //cap.open("james-deane-drifting-s15.jpg");
    cap.open("rx7_license_plate.png");

    static const std::string kWinName = "EAST: An Efficient and Accurate Scene Text Detector";
    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

    std::vector<cv::Mat> output;
    std::vector<cv::String> outputLayers(2);
    outputLayers[0] = "feature_fusion/Conv_7/Sigmoid";
    outputLayers[1] = "feature_fusion/concat_3";

    cv::Mat frame, blob;
    for (int i=0; true; i++)
    {
        cap >> frame;
        if (frame.empty())
        {
            cv::waitKey();
            break;
        }

        cv::dnn::blobFromImage(frame, blob, 1.0, cv::Size(inpWidth, inpHeight), cv::Scalar(123.68, 116.78, 103.94), true, false);
        net.setInput(blob);
        net.forward(output, outputLayers);

        cv::Mat scores = output[0];
        cv::Mat geometry = output[1];

        std::vector<cv::RotatedRect> boxes;
        std::vector<float> confidences;
        decode(scores, geometry, confThreshold, boxes, confidences);

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        tesseract::TessBaseAPI* ocr = new tesseract::TessBaseAPI();
        ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
        ocr->SetImage(frame.data, frame.cols, frame.rows, 3, frame.step);
        //ocr->SetSourceResolution(2000);

        cv::Point2f ratio((float)frame.cols / inpWidth, (float)frame.rows / inpHeight);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            //cv::RotatedRect& box = boxes[indices[i]];

            //cv::Point2f vertices[4];
            //box.points(vertices);
            //for (int j = 0; j < 4; ++j)
            //{
            //    vertices[j].x *= ratio.x;
            //    vertices[j].y *= ratio.y;
            //}
            //for (int j = 0; j < 4; ++j)
            //{
            //    line(frame, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            //}
            //ocr->SetRectangle(vertices[1].x, vertices[1].y, std::abs(vertices[2].x - vertices[1].x), std::abs(vertices[1].y - vertices[0].y));

            cv::Rect bbox = boxes[indices[i]].boundingRect();
            cv::Rect normalized_bbox = cv::Rect{ int((float)bbox.x * ratio.x), int((float)bbox.y * ratio.y), int((float)bbox.width * ratio.x), int((float)bbox.height * ratio.y) };


            ocr->SetRectangle(normalized_bbox.x, normalized_bbox.y, normalized_bbox.width, normalized_bbox.height);


            ocr->SetSourceResolution(2000);
            std::string outText = ocr->GetUTF8Text();

            std::cout << outText << std::endl;

            cv::Mat blured_region;
            cv::GaussianBlur(frame(normalized_bbox), blured_region, cv::Size(0, 0), 4);
            blured_region.copyTo(frame(normalized_bbox));
            
        }
        

        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("Inference time: %.2f ms", t);
        putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

        imshow(kWinName, frame);

        ocr->End();
    }

    
    return 0;
}


int test_abstract()
{
	core_api::Blurer blurer;
	blurer.init();

    blurer.load("rx7_license_plate.png");
    //blurer.load("james-deane-drifting-s15.jpg");
    //blurer.load("book.jpeg");

	blurer.detect();
	blurer.load_blurred_to_buffer();

	static const std::string kWinName = "TEST";
	cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

    auto data = blurer.buffer();
    cv::Mat frame{ data.height,  data.width, CV_8UC3, (void*)data.data.data() };
    while (true)
    {
        cv::imshow(kWinName, frame);
        cv::waitKey();
    }

	return 0;
}


int main()
{
    //return test_raw();
    return test_abstract();
}
