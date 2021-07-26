/* Copyright 2019-2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Tests DMA buffer support for model pipelining.
//
// It assumes the following test data inside `FLAGS_test_data_dir` folder
//  - inception_v3_299_quant_edgetpu.tflite
//  - pipeline/inception_v3_299_quant_segment_0_of_2_edgetpu.tflite
//  - pipeline/inception_v3_299_quant_segment_1_of_2_edgetpu.tflite
#include <glib.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/gst.h>
#include <sys/mman.h>

#include <thread>  // NOLINT

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "coral/error_reporter.h"
#include "coral/pipeline/allocator.h"
#include "coral/pipeline/pipelined_model_runner.h"
#include "coral/pipeline/test_utils.h"
#include "coral/test_utils.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tflite/public/edgetpu.h"

namespace coral {
namespace {

const int kMinFrames = 10;
// This file is part of the system image as part of OOBE.
const char *kVideoPath = "/usr/share/edgetpudemo/video_device.mp4";
const char *kPipeline =
    "filesrc location=%s ! decodebin ! glfilterbin filter=glbox"
    " ! video/x-raw,format=RGB,width=%d,height=%d"
    " ! appsink name=appsink sync=false emit-signals=true";

const char *kModelBaseName = "inception_v3_299_quant";

class DmaBuffer : public Buffer {
 public:
  DmaBuffer(GstSample *sample, size_t requested_bytes)
      : sample_(CHECK_NOTNULL(sample)), requested_bytes_(requested_bytes) {}

  void *ptr() override { return nullptr; }

  void *MapToHost() override {
    if (!handle_) {
      handle_ = mmap(nullptr, requested_bytes_, PROT_READ, MAP_PRIVATE, fd(),
                     /*offset=*/0);
      if (handle_ == MAP_FAILED) {
        handle_ = nullptr;
      }
    }
    return handle_;
  }

  bool UnmapFromHost() override {
    if (munmap(handle_, requested_bytes_) != 0) {
      return false;
    }
    return true;
  }

  int fd() {
    if (fd_ == -1) {
      GstBuffer *buf = CHECK_NOTNULL(gst_sample_get_buffer(sample_));
      GstMemory *mem = gst_buffer_peek_memory(buf, 0);
      if (gst_is_dmabuf_memory(mem)) {
        fd_ = gst_dmabuf_memory_get_fd(mem);
      }
    }
    return fd_;
  }

 private:
  friend class DmaAllocator;

  GstSample *sample_ = nullptr;
  size_t requested_bytes_ = 0;

  int fd_ = -1;
  void *handle_ = nullptr;
};

class DmaAllocator : public Allocator {
 public:
  DmaAllocator(GstElement *sink) : sink_(CHECK_NOTNULL(sink)) {}

  Buffer *Alloc(size_t size_bytes) override {
    GstSample *sample;
    g_signal_emit_by_name(sink_, "pull-sample", &sample);
    return new DmaBuffer(sample, size_bytes);
  }

  void Free(Buffer *buffer) override {
    auto *sample = static_cast<DmaBuffer *>(buffer)->sample_;
    if (sample) {
      gst_sample_unref(sample);
    }

    delete buffer;
  }

 private:
  GstElement *sink_ = nullptr;
};

// Monitors bus for error messages.
gboolean OnBusCall(GstBus *bus, GstMessage *msg, GMainLoop *loop) {
  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      g_printerr("End of stream\n");
      g_main_loop_quit(loop);
      break;
    case GST_MESSAGE_ERROR: {
      GError *error;
      gst_message_parse_error(msg, &error, NULL);
      g_printerr("Error: %s\n", error->message);
      g_error_free(error);
      g_main_loop_quit(loop);
      break;
    }
    default:
      break;
  }

  return TRUE;
}

struct PipelinedModelState {
  PipelinedModelState(GMainLoop *loop, PipelinedModelRunner *runner,
                      tflite::Interpreter *first_segment_interpreter) {
    this->loop = loop;
    this->runner = runner;
    this->first_segment_interpreter = first_segment_interpreter;
    this->seen_frames = 0;
  }

  ~PipelinedModelState() { g_main_loop_unref(this->loop); }

  GMainLoop *loop;
  PipelinedModelRunner *runner;
  // Needed to get input tensor types and input tensor size.
  tflite::Interpreter *first_segment_interpreter;
  int seen_frames;
};

// Pushes to PipelinedModelRunner whenever a new frame is available. It returns
// immediately after the push, and results are consumed in a separate thread.
GstFlowReturn PipelinedModelOnNewSample(GstElement *sink,
                                        PipelinedModelState *state) {
  GstFlowReturn ret = GST_FLOW_ERROR;

  ++state->seen_frames;

  PipelineTensor input_buffer;
  const TfLiteTensor *input_tensor =
      state->first_segment_interpreter->input_tensor(0);
  input_buffer.name = input_tensor->name;
  input_buffer.buffer =
      state->runner->GetInputTensorAllocator()->Alloc(input_tensor->bytes);
  input_buffer.type = input_tensor->type;
  input_buffer.bytes = input_tensor->bytes;
  CHECK(state->runner->Push({input_buffer}).ok());

  if (state->seen_frames >= kMinFrames) {
    CHECK(state->runner->Push({}).ok());
  }

  ret = state->seen_frames >= kMinFrames ? GST_FLOW_EOS : GST_FLOW_OK;

  if (ret != GST_FLOW_OK) {
    g_main_loop_quit(state->loop);
  }
  return ret;
}

struct RefModelState {
  RefModelState(GMainLoop *loop, tflite::Interpreter *interpreter,
                std::vector<std::vector<uint8_t>> *ref_result) {
    this->loop = loop;
    this->interpreter = interpreter;
    this->ref_result = ref_result;
    this->seen_frames = 0;
  }

  ~RefModelState() { g_main_loop_unref(this->loop); }

  GMainLoop *loop;
  tflite::Interpreter *interpreter;
  std::vector<std::vector<uint8_t>> *ref_result;
  int seen_frames;
};

GstFlowReturn RefModelOnNewSample(GstElement *sink, RefModelState *state) {
  GstFlowReturn ret = GST_FLOW_ERROR;

  GstSample *sample;
  g_signal_emit_by_name(sink, "pull-sample", &sample);

  if (sample) {
    ++state->seen_frames;
    GstBuffer *buf = gst_sample_get_buffer(sample);
    GstMapInfo info;
    if (gst_buffer_map(buf, &info, GST_MAP_READ) == TRUE) {
      uint8_t *input_tensor =
          state->interpreter->typed_input_tensor<uint8_t>(0);
      std::memcpy(input_tensor, info.data, info.size);

      CHECK_EQ(state->interpreter->Invoke(), kTfLiteOk);

      CHECK_EQ(state->interpreter->outputs().size(), 1);
      const auto *score_tensor = state->interpreter->output_tensor(0);
      const auto *score = tflite::GetTensorData<uint8_t>(score_tensor);
      state->ref_result->push_back(
          std::vector<uint8_t>(score, score + score_tensor->bytes));

      gst_buffer_unmap(buf, &info);
      ret = state->seen_frames >= kMinFrames ? GST_FLOW_EOS : GST_FLOW_OK;
    }
  }

  if (sample) {
    gst_sample_unref(sample);
  }

  if (ret != GST_FLOW_OK) {
    g_main_loop_quit(state->loop);
  }
  return ret;
}

// Analyzes kMinFrames frames using non-partitioned model and stores the result
// as reference.
void GetRefModelResult(const std::string &model_path,
                       std::vector<std::vector<uint8_t>> *ref_result) {
  auto context =
      CHECK_NOTNULL(edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
          edgetpu::DeviceType::kApexPci));
  auto model =
      CHECK_NOTNULL(tflite::FlatBufferModel::BuildFromFile(model_path.c_str()));
  EdgeTpuErrorReporter error_reporter;
  auto interpreter =
      CHECK_NOTNULL(CreateInterpreter(*model, context.get(), &error_reporter));

  const TfLiteTensor *input_tensor =
      CHECK_NOTNULL(interpreter->input_tensor(0));
  int tensor_width = input_tensor->dims->data[1];
  int tensor_height = input_tensor->dims->data[2];

  gst_init(nullptr, nullptr);
  gchar *pipeline_desc =
      g_strdup_printf(kPipeline, kVideoPath, tensor_width, tensor_height);
  g_print("GStreamer pipeline:\n%s\n", pipeline_desc);
  GstElement *pipeline =
      CHECK_NOTNULL(gst_parse_launch(pipeline_desc, nullptr));
  g_free(pipeline_desc);

  GstElement *sink = gst_bin_get_by_name(GST_BIN(pipeline), "appsink");

  RefModelState state(g_main_loop_new(NULL, FALSE), interpreter.get(),
                      ref_result);
  g_signal_connect(sink, "new-sample", G_CALLBACK(RefModelOnNewSample), &state);
  gst_object_unref(sink);

  GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  gst_bus_add_watch(bus, reinterpret_cast<GstBusFunc>(OnBusCall), state.loop);
  gst_object_unref(bus);

  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(state.loop);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
}

class ModelPipeliningDmaBufDevboardTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const auto model_path =
        TestDataPath(absl::StrCat(kModelBaseName, "_edgetpu.tflite"));

    ref_result_ = new std::vector<std::vector<uint8_t>>();
    ref_result_->reserve(kMinFrames);
    GetRefModelResult(model_path, ref_result_);
  }

  void CheckPipelinedModelInferenceResult(
      std::vector<edgetpu::EdgeTpuContext *> contexts) {
    std::vector<std::string> model_segment_paths(num_segments);
    for (int i = 0; i < num_segments; ++i) {
      model_segment_paths[i] = TestDataPath(
          absl::Substitute("pipeline/$0_segment_$1_of_$2_edgetpu.tflite",
                           kModelBaseName, i, num_segments));
    }

    std::vector<std::unique_ptr<tflite::Interpreter>> managed_interpreters(
        num_segments);
    std::vector<tflite::Interpreter *> interpreters(num_segments);
    std::vector<EdgeTpuErrorReporter> error_reporters(num_segments);
    std::vector<std::unique_ptr<tflite::FlatBufferModel>> models(num_segments);
    for (int i = 0; i < num_segments; ++i) {
      models[i] = CHECK_NOTNULL(tflite::FlatBufferModel::BuildFromFile(
          model_segment_paths[i].c_str()));
      managed_interpreters[i] = CHECK_NOTNULL(
          CreateInterpreter(*(models[i]), contexts[i], &error_reporters[i]));
      interpreters[i] = managed_interpreters[i].get();
    }

    const TfLiteTensor *input_tensor = interpreters[0]->input_tensor(0);
    ASSERT_EQ(input_tensor->type, kTfLiteUInt8);
    const auto &output_indices = interpreters[num_segments - 1]->outputs();
    for (size_t i = 0; i < output_indices.size(); ++i) {
      const auto *out_tensor = CHECK_NOTNULL(
          interpreters[num_segments - 1]->tensor(output_indices[i]));
      ASSERT_EQ(out_tensor->type, kTfLiteUInt8);
    }
    int tensor_width = input_tensor->dims->data[1];
    int tensor_height = input_tensor->dims->data[2];
    ASSERT_EQ(input_tensor->dims->data[3], 3);  // channels, 3 for RGB

    gst_init(nullptr, nullptr);
    gchar *pipeline_desc =
        g_strdup_printf(kPipeline, kVideoPath, tensor_width, tensor_height);
    g_print("GStreamer pipeline:\n%s\n", pipeline_desc);
    GstElement *pipeline =
        CHECK_NOTNULL(gst_parse_launch(pipeline_desc, nullptr));
    g_free(pipeline_desc);

    GstElement *sink =
        CHECK_NOTNULL(gst_bin_get_by_name(GST_BIN(pipeline), "appsink"));

    std::unique_ptr<Allocator> dma_allocator(new DmaAllocator(sink));
    std::unique_ptr<PipelinedModelRunner> runner(
        new PipelinedModelRunner(interpreters, dma_allocator.get()));
    PipelinedModelState state(g_main_loop_new(NULL, FALSE), runner.get(),
                              interpreters[0]);
    g_signal_connect(sink, "new-sample", G_CALLBACK(PipelinedModelOnNewSample),
                     &state);
    gst_object_unref(sink);

    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_watch(bus, reinterpret_cast<GstBusFunc>(OnBusCall), state.loop);
    gst_object_unref(bus);

    auto check_result = [&runner, this]() {
      std::vector<PipelineTensor> output_tensors;
      int counter = 0;
      while (runner->Pop(&output_tensors).ok() && !output_tensors.empty()) {
        ASSERT_EQ(output_tensors.size(), 1);
        const auto &expected = (*ref_result_)[counter];
        const auto *actual =
            static_cast<uint8_t *>(output_tensors[0].buffer->ptr());
        ASSERT_EQ(output_tensors[0].bytes, expected.size());
        for (int i = 0; i < expected.size(); ++i) {
          EXPECT_EQ(actual[i], expected[i]);
        }
        runner->GetOutputTensorAllocator()->Free(output_tensors[0].buffer);

        output_tensors.clear();
        counter++;
      }
      EXPECT_EQ(counter, ref_result_->size());
    };

    auto consumer = std::thread(check_result);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(state.loop);

    consumer.join();
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
  }

  static std::vector<std::vector<uint8_t>> *ref_result_;
  const int num_segments = 2;
};

std::vector<std::vector<uint8_t>>
    *ModelPipeliningDmaBufDevboardTest::ref_result_ = nullptr;

TEST_F(ModelPipeliningDmaBufDevboardTest, DmaBufInputSupported) {
  auto pci_context =
      CHECK_NOTNULL(edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
          edgetpu::DeviceType::kApexPci));
  auto usb_context =
      CHECK_NOTNULL(edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
          edgetpu::DeviceType::kApexUsb));
  // PCI Edge TPU supports DMA buffer as input starting from Mendel Eagle.
  CheckPipelinedModelInferenceResult({pci_context.get(), usb_context.get()});
}

TEST_F(ModelPipeliningDmaBufDevboardTest, DmaBufInputNotSupported) {
  auto pci_context =
      CHECK_NOTNULL(edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
          edgetpu::DeviceType::kApexPci));
  auto usb_context =
      CHECK_NOTNULL(edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
          edgetpu::DeviceType::kApexUsb));
  // USB Edge TPU does not support DMA buffer as input.
  CheckPipelinedModelInferenceResult({usb_context.get(), pci_context.get()});
}

}  // namespace
}  // namespace coral
