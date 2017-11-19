#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/contrib/observers/time_observer.h"

#include <algorithm>
#include <cmath>
#include <google/protobuf/repeated_field.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace caffe2;

CAFFE2_DEFINE_string(file_path, ".", "the path of all the relative files");
CAFFE2_DEFINE_string(predict_net, "predict_net.pb", "the relative path of predict_net.pb");
CAFFE2_DEFINE_string(init_net, "init_net.pb", "the path of init_net.pb");
CAFFE2_DEFINE_string(image_file, "image_file.txt", "the image file contianing the images");
CAFFE2_DEFINE_string(label_file, "coco.names", "the image file contianing the images");
CAFFE2_DEFINE_int(input_size, 416, "the size of image to resize");
CAFFE2_DEFINE_int(num_workers, 1, "the num of workers");

vector<float> biases({1.322, 1.73, 3.19, 4.0, 5.05, 8.09, 9.47, 4.84, 11.23, 10.00});
const int boxes_of_each_grid = 5;
const int num_classes = 80;
const float thresh_score = 0.45, nms_thresh = 0.4;
const int num_channels = 3;

float logistic_activate(float a) {
  return 1.0 / ( 1.0 + std::exp(-a));
}

class Box {
 public:
  Box(const float* predictions, int n, int index, int col, int row, int width, int height) {
   center_x = (col + logistic_activate(predictions[index + 0])) / width;
   center_y = (row + logistic_activate(predictions[index + 1])) / height;
   w = std::exp(predictions[index + 2]) * biases[2 * n + 0] / width;
   h = std::exp(predictions[index + 3]) * biases[2 * n + 1] / height;
   category_ = -1;
   score_ = 0.0;
  }

  float overlap(float x1, float w1, float x2, float w2) {
    float left1 = x1 - w1 / 2.0;
    float left2 = x2 - w2 / 2.0;
    float right1 = x1 + w1 / 2.0;
    float right2 = x2 + w2 / 2.0;
    float left = left1 > left2 ? left1 : left2;
    float right = right1 > right2 ? right2 : right1;
    return right - left;
  }

  float iou(const Box* b) {
    float w_i = overlap(center_x, w, b->center_x, b->w);
    float h_i = overlap(center_y, h, b->center_y, b->h);
    if (w_i < 0 || h_i < 0) {
      return 0.0;
    }
    float inner_section = w_i * h_i;
    float union_section = w * h + b->w * b->h - inner_section;
    return inner_section / union_section;
  }

  inline void set_score(float s) {
    score_  = s;
  }
  inline void set_category(int c) {
    category_ = c;
  }
  inline float score() {
    return score_;
  }
  inline int category() {
    return category_;
  }
  inline vector<float> position() {
    vector<float> res;
    res.push_back(center_x);
    res.push_back(center_y);
    res.push_back(w);
    res.push_back(h);
    return res;
  }
 private:
  float center_y, center_x;
  float w, h;
  float score_;
  int category_;
};

void get_result_boxes(caffe2::TensorCPU& output_tensor, vector<Box*>& result_boxes) {
  vector<Box*> tmp_boxes;
  const float* output_tensor_data = output_tensor.template data<float>();
  int channel = output_tensor.dim(1), height = output_tensor.dim(2), width = output_tensor.dim(3);
  for (int i = 0; i < height * width; ++i) {
    int row = i / width;
    int col = i % width;
    for (int n = 0; n < boxes_of_each_grid; ++n) {
      int index = i * channel + (num_classes + 5) * n;
      float object_score = output_tensor_data[index + 4];
      if (object_score < thresh_score) continue;
      Box* box_tmp = new Box(output_tensor_data, n, index, col, row, width, height);
      int class_index = index + 5;
      int max_prob_index = -1;
      float max_score = 0.0;
      for (int j = 0; j < num_classes; ++j) {
        float prob = object_score * output_tensor_data[class_index + j];
        if (prob > thresh_score && prob > max_score) {
          max_score = prob;
          max_prob_index = j;
        }
      }
      if (max_prob_index != -1) {
        box_tmp->set_score(max_score);
        box_tmp->set_category(max_prob_index);
        tmp_boxes.push_back(box_tmp);
      }
    }
  }
  // nms
  for (int i = 0; i < tmp_boxes.size(); ++i) {
    for (int j = i + 1; j < tmp_boxes.size(); ++j) {
      if (tmp_boxes[i]->iou(tmp_boxes[j]) > nms_thresh) {
        if (tmp_boxes[i]->score() < tmp_boxes[j]->score()) {
          tmp_boxes[i]->set_score(0.0);
        } else {
          tmp_boxes[j]->set_score(0.0);
        }
      }
    }
  }
  for (int i = 0; i < tmp_boxes.size(); ++i) {
    if (tmp_boxes[i]->score() > 1e-6) {
      result_boxes.push_back(tmp_boxes[i]);
    }
  }
}

void get_point_position(const vector<float> pos, cv::Point& p1, cv::Point& p2, int h, int w) {
  int left = (pos[0] - pos[2] / 2.0) * w;
  int right = (pos[0] + pos[2] / 2.0) * w;
  int top = (pos[1] - pos[3] / 2.0) * h;
  int bottom = (pos[1] + pos[3] / 2.0) * h;
  if (left < 0) left = 0.0;
  if (top < 0) top = 0;
  if (right > w) right = w;
  if (bottom > h) bottom = h;
  p1.x = left;
  p1.y = top;
  p2.x = right;
  p2.y = bottom;
}

void Preprocess(const cv::Mat&, std::vector<cv::Mat>*, TensorCPU*, int);
//void PrintSummaryInfo();

void run() {
  string predict_net_filename = FLAGS_file_path + "/" + FLAGS_predict_net;
  string init_net_filename = FLAGS_file_path + "/" + FLAGS_init_net;
  string image_file_filename = FLAGS_file_path + "/" + FLAGS_image_file;
  string label_file_filename = FLAGS_file_path + "/" + FLAGS_label_file;
  // PrintSummaryInfo();

  vector<string> inputs_file;
  std::vector<cv::Mat> imgs;
  // read images from file, put into imgs and inputs_file
  {
    std::ifstream images_file(image_file_filename);
    CHECK(images_file) << "Unable to open image file! " << image_file_filename;
    string image_line;
    while (std::getline(images_file, image_line)) {
      if (!image_line.empty()) {
        image_line = FLAGS_file_path + "/" + image_line;
        inputs_file.push_back(image_line);
        cv::Mat img = cv::imread(image_line, -1);
        if (img.empty()) {
          std::cerr << "Warning: can't open image:" << image_line << std::endl;
          continue;
        }
        imgs.push_back(img);
      }
    }
    images_file.close();
  }

  // get label from file
  vector<string> labels_;
  {
    std::ifstream labels(label_file_filename);
    string line;
    while (std::getline(labels, line)) {
      labels_.push_back(string(line));
    }
    labels.close();
  }

  NetDef init_model, predict_model;
  CAFFE_ENFORCE(ReadProtoFromFile(init_net_filename, &init_model));
  CAFFE_ENFORCE(ReadProtoFromFile(predict_net_filename, &predict_model));
  Workspace workspace;
  auto init_net = CreateNet(init_model, &workspace);
  init_net->Run();

  auto output_name = predict_model.external_output(0);
  auto blob = workspace.CreateBlob(predict_model.external_input(0));
  auto* tensor = blob->GetMutable<TensorCPU>();
  tensor->Resize(1, num_channels, FLAGS_input_size, FLAGS_input_size);

  predict_model.set_type("dag");
  predict_model.set_num_workers(FLAGS_num_workers);
  auto predict_net = CreateNet(predict_model, &workspace); 
  unique_ptr<TimeObserver<NetBase>> net_ob = make_unique<TimeObserver<NetBase>>(predict_net.get());
  predict_net->SetObserver(std::move(net_ob));

  for (int i = 0; i < imgs.size(); ++i) {
    Timer* predict_time = new Timer();
    vector<cv::Mat> input_imgs;
    Preprocess(imgs[i], &input_imgs, tensor, FLAGS_input_size);
    std::cout << "start_time : " << predict_time->MicroSeconds() << std::endl;
    predict_net->Run();
    std::cout << "forward_time : " << predict_time->MicroSeconds() << std::endl;
    auto output_tensor = workspace.GetBlob(output_name)->Get<TensorCPU>();
    vector<Box*> result_boxes;
    get_result_boxes(output_tensor, result_boxes);
    std::cout << "get_boxes_time : " << predict_time->MicroSeconds() << std::endl;

    std::cout << "Detection for image " << inputs_file[i] << std::endl;
    for (int j = 0; j < result_boxes.size(); ++j) {
      cv::Point p1, p2;
      get_point_position(result_boxes[j]->position(), p1, p2, imgs[i].size().height, imgs[i].size().width);
      cv::rectangle(imgs[i], p1, p2, cv::Scalar(0, 255, 0), 8, 8, 0);
      std::stringstream s0;
      s0 << result_boxes[j]->score();
      string s1 = s0.str();
      cv::putText(imgs[i], labels_[(int)result_boxes[j]->category()], cv::Point(p1.x, (p1.y + p2.y) / 2 + 10), 2, 0.5, cv::Scalar(255, 0, 0), 0, 8, 0);
      cv::putText(imgs[i], s1.c_str(), cv::Point(p1.x, (p1.y + p2.y) / 2 + 10), 2, 0.5, cv::Scalar(255, 0, 0), 0, 8, 0);
      std::cout << ">> label : " << labels_[(int)result_boxes[j]->category()] << " score : " << result_boxes[j]->score() << " ";
      for (int idx = 0; idx < 4; ++idx) {
        std::cout << result_boxes[j]->position()[idx] << " ";
      }
      std::cout << std::endl;
    }
    int pos = inputs_file[i].find_last_of("/");
    string img_name;
    std::stringstream ss;
    ss << "yolo2_" << inputs_file[i].substr(pos + 1, inputs_file[i].length() - pos - 1);
    ss >> img_name;
    cv::imwrite(img_name, imgs[i]);
    std::cout << "predict_time : " << predict_time->MicroSeconds() << std::endl;
  }
}

void Preprocess(const cv::Mat& imgs, std::vector<cv::Mat>* input_imgs, TensorCPU* input_tensor, int size_to_fit) {
  int width = size_to_fit;
  int height = size_to_fit;
  float* input_data = input_tensor->template mutable_data<float>();
  for (int j = 0; j < input_tensor->dim32(1); ++j) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    (*input_imgs).push_back(channel);
    input_data += width * height;
  }

  auto input_geometry_ = cv::Size(height, width);
  cv::Mat sample;
  if (imgs.channels() == 1) {
    cv::cvtColor(imgs, sample, cv::COLOR_GRAY2BGR);
  } else {
    sample = imgs;
  }
  cv::Mat sample_float;
  sample.convertTo(sample_float, CV_32FC3);
  double min_val, max_val;
  cv::minMaxLoc(sample_float, &min_val, &max_val);
  cv::Mat sample_norm = (sample_float - min_val) / (max_val - min_val);
  cv::Mat sample_resized;
  if (sample_norm.size() != input_geometry_) {
    cv::resize(sample_norm, sample_resized, input_geometry_);
  } else {
    sample_resized = sample_norm;
  }
  cv::split(sample_resized, *input_imgs);
}

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  run();
  //google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
