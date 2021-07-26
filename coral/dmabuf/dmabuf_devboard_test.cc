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

#include <glib.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/gst.h>

#include "absl/status/status.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
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
const int kCarLabelIndex = 2;
const int kMinCarsPerFrame = 2;
const float kCarThreshold = 0.6;
// These files are part of the system image as part of OOBE.
const char *kModelPath =
    "/usr/share/edgetpudemo/"
    "mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite";
const char *kVideoPath = "/usr/share/edgetpudemo/video_device.mp4";
const char *kPipeline =
    "filesrc location=%s ! decodebin ! glfilterbin filter=glbox"
    " ! video/x-raw,format=RGB,width=%d,height=%d"
    " ! appsink name=appsink sync=false emit-signals=true";

struct State {
  State(GMainLoop *loop, std::unique_ptr<tflite::Interpreter> interpreter) {
    this->loop = loop;
    this->interpreter = std::move(interpreter);
    this->seen_frames = 0;
    this->detection_success_frames = 0;
  }

  ~State() { g_main_loop_unref(this->loop); }

  GMainLoop *loop;
  std::unique_ptr<tflite::Interpreter> interpreter;
  int seen_frames;
  int detection_success_frames;
};

// Monitors bus for error messages.
gboolean OnBusCall(GstBus *bus, GstMessage *msg, State *state) {
  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      g_printerr("End of stream\n");
      g_main_loop_quit(state->loop);
      break;
    case GST_MESSAGE_ERROR: {
      GError *error;
      gst_message_parse_error(msg, &error, NULL);
      g_printerr("Error: %s\n", error->message);
      g_error_free(error);
      g_main_loop_quit(state->loop);
      break;
    }
    default:
      break;
  }

  return TRUE;
}

GstFlowReturn OnNewSample(GstElement *sink, State *state) {
  GstSample *sample;
  GstFlowReturn ret = GST_FLOW_ERROR;

  g_signal_emit_by_name(sink, "pull-sample", &sample);

  if (sample) {
    ++state->seen_frames;
    GstBuffer *buf = gst_sample_get_buffer(sample);
    GstMemory *mem = gst_buffer_peek_memory(buf, 0);
    if (gst_is_dmabuf_memory(mem)) {
      GstMapInfo info;
      gst_buffer_map(buf, &info, GST_MAP_READ);
      if (InvokeWithDmaBuffer(state->interpreter.get(),
                              gst_dmabuf_memory_get_fd(mem), info.size)
              .ok()) {
        // The tensors are <bounding boxes, label ids, scores, number of
        // predictions>.
        const auto *label_tensor = state->interpreter->output_tensor(1);
        const auto *score_tensor = state->interpreter->output_tensor(2);
        const float *labels = tflite::GetTensorData<float>(label_tensor);
        const float *scores = tflite::GetTensorData<float>(score_tensor);
        const int num_values = label_tensor->bytes / sizeof(float);

        int num_cars = 0;
        for (int i = 0; i < num_values; ++i) {
          if (static_cast<int>(labels[i]) == kCarLabelIndex &&
              scores[i] >= kCarThreshold) {
            ++num_cars;
          }
        }
        g_print("Frame: %2d/%d, cars (>=%.2f): %d\n", state->seen_frames,
                kMinFrames, kCarThreshold, num_cars);
        if (num_cars >= kMinCarsPerFrame) {
          ++state->detection_success_frames;
        }
        ret = state->seen_frames >= kMinFrames ? GST_FLOW_EOS : GST_FLOW_OK;
      } else {
        g_printerr("Failed to invoke interpreter with dma-buf input\n");
      }
    } else {
      g_printerr("Received non dmabuf memory\n");
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

// Decodes OOBE video file on VPU to dma-buf, passes it to GPU for scaling, and
// to TPU for inference. At least kMinCarsPerFrame must be detected in each of
// the kMinFrames first frames for the test to pass. Input tensor is stored in a
// dma-buf, and there's no CPU access of the input tensor data. This is a test
// of dma-buf input tensor support, not OOBE model correctness.
TEST(DmaBufDevBoard, TestDmaBufInputTfLite) {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(kModelPath);
  ASSERT_NE(model, nullptr);

  std::shared_ptr<edgetpu::EdgeTpuContext> context =
      edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  ASSERT_NE(context, nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, context.get());
  interpreter->AllocateTensors();

  const TfLiteTensor *input_tensor = interpreter->input_tensor(0);
  ASSERT_EQ(input_tensor->type, kTfLiteUInt8);

  const auto &output_indices = interpreter->outputs();
  ASSERT_EQ(output_indices.size(), 4);
  for (size_t i = 0; i < output_indices.size(); ++i) {
    const auto *out_tensor = interpreter->tensor(output_indices[i]);
    ASSERT_NE(out_tensor, nullptr);
    ASSERT_EQ(out_tensor->type, kTfLiteFloat32);
  }

  int tensor_width = input_tensor->dims->data[1];
  int tensor_height = input_tensor->dims->data[2];
  ASSERT_EQ(input_tensor->dims->data[3], 3);  // channels, 3 for RGB

  gst_init(nullptr, nullptr);
  State state(g_main_loop_new(NULL, FALSE), std::move(interpreter));
  gchar *pipeline_desc =
      g_strdup_printf(kPipeline, kVideoPath, tensor_width, tensor_height);
  g_print("GStreamer pipeline:\n%s\n", pipeline_desc);
  GstElement *pipeline = gst_parse_launch(pipeline_desc, nullptr);
  g_free(pipeline_desc);
  ASSERT_NE(pipeline, nullptr);

  GstElement *sink = gst_bin_get_by_name(GST_BIN(pipeline), "appsink");
  ASSERT_NE(sink, nullptr);
  g_signal_connect(sink, "new-sample", G_CALLBACK(OnNewSample), &state);
  gst_object_unref(sink);

  GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  gst_bus_add_watch(bus, reinterpret_cast<GstBusFunc>(OnBusCall), &state);
  gst_object_unref(bus);

  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(state.loop);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);

  ASSERT_GE(state.seen_frames, kMinFrames);
  ASSERT_EQ(state.seen_frames, state.detection_success_frames);
}

}  // namespace
}  // namespace coral
