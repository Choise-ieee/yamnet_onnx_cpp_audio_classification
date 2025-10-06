#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <iomanip>
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include <Windows.h>

constexpr int MAX_SAMPLES = 48000;  // 3s * 16000Hz
constexpr int SAMPLE_RATE = 16000;
constexpr int CHANNELS = 1;

// 新增结果结构体
struct TopResult {
	int index;
	float score;
};

class WavLoader {
public:
	explicit WavLoader(const std::string& filename) {
		load_wav_file(filename);
	}

	const std::vector<float>& data() const { return audio_data_; }

private:
#pragma pack(push, 1)
	struct WavHeader {
		char riff[4];
		uint32_t file_size;
		char wave[4];
		char fmt[4];
		uint32_t fmt_size;
		uint16_t format;
		uint16_t channels;
		uint32_t sample_rate;
		uint32_t byte_rate;
		uint16_t block_align;
		uint16_t bits_per_sample;
		char data[4];
		uint32_t data_size;
	};
#pragma pack(pop)

	std::vector<float> audio_data_;

	void load_wav_file(const std::string& filename) {
		std::ifstream file(filename, std::ios::binary);
		if (!file) throw std::runtime_error("Failed to open WAV file");

		WavHeader header;
		file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));

		// Validate WAV format
		if (memcmp(header.riff, "RIFF", 4) != 0 ||
			memcmp(header.wave, "WAVE", 4) != 0 ||
			memcmp(header.fmt, "fmt ", 4) != 0 ||
			memcmp(header.data, "data", 4) != 0) {
			throw std::runtime_error("Invalid WAV file format");
		}

		if (header.channels != CHANNELS ||
			header.sample_rate != SAMPLE_RATE ||
			header.bits_per_sample != 16) {
			throw std::runtime_error("Unsupported audio format");
		}

		// Read PCM data
		const size_t sample_count = header.data_size / sizeof(int16_t);
		std::vector<int16_t> pcm_data(sample_count);
		file.read(reinterpret_cast<char*>(pcm_data.data()), header.data_size);

		// Convert to float32 [-1.0, 1.0]
		audio_data_.resize(MAX_SAMPLES, 0.0f);
		const size_t copy_length = std::min<size_t>(sample_count, MAX_SAMPLES);
		for (size_t i = 0; i < copy_length; ++i) {
			audio_data_[i] = pcm_data[i] / 32768.0f;
		}
		int a;
	}
};

class OnnxModel {
public:
	explicit OnnxModel(std::wstring& modelPath, bool use_cuda = false)
		: env_(ORT_LOGGING_LEVEL_WARNING, "AudioClassifier"),
		// 关键修改：在初始化列表中创建 session_options 并初始化 session_
		session_(env_, modelPath.c_str(), [use_cuda]() {
		Ort::SessionOptions session_options;
		if (use_cuda) 
		{
			// 检查CUDA设备
			/*int num_devices = 0;
			cudaError_t err = cudaGetDeviceCount(&num_devices);
			if (err != cudaSuccess || num_devices == 0) {
				throw std::runtime_error("No available CUDA devices");
			}*/

			// 添加CUDA执行提供器
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(
				session_options,
				0  // 设备ID
			));
		}
		// 启用内存优化
		session_options.EnableCpuMemArena();
		session_options.EnableMemPattern();
		return session_options;
			}()),  // 立即调用lambda返回配置好的session_options
		memory_info_(Ort::MemoryInfo::CreateCpu(
			OrtAllocatorType::OrtArenaAllocator,
			OrtMemType::OrtMemTypeDefault))
	{
		// 验证内存信息
		if (!memory_info_) {
			throw std::runtime_error("Failed to create memory info");
		}
	}

		std::vector<TopResult> predict(const float* input_data) {
		// 创建输入Tensor（正确参数顺序）
		const std::array<int64_t, 2> input_shape = { 1, MAX_SAMPLES };

		// 使用初始化过的 memory_info_
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
			memory_info_,          // 使用成员变量
			const_cast<float*>(input_data),
			MAX_SAMPLES,
			input_shape.data(),
			input_shape.size()
			);


		// 验证Tensor创建是否成功
		if (!input_tensor.IsTensor()) 
		{
			throw std::runtime_error("Failed to create input tensor");
		}

		std::vector<std::string> input_node_names;
		std::vector<std::string> output_node_names;

		size_t numInputNodes = session_.GetInputCount();
		size_t numOutputNodes = session_.GetOutputCount();
		Ort::AllocatorWithDefaultOptions allocator;
		input_node_names.reserve(numInputNodes);
		for (int i = 0; i < numInputNodes; i++) {
			//onnx newest version-1.14
			/*auto input_name = session_.GetInputNameAllocated(i, allocator);
			input_node_names.push_back(input_name.get());*/

			//onnx old version-1.8
			input_node_names.push_back(session_.GetInputName(i, allocator));
		}

		for (int i = 0; i < numOutputNodes; i++)
		{
			//onnx newest version-1.14
			/*auto out_name = session_.GetOutputNameAllocated(i, allocator);
			output_node_names.push_back(out_name.get());*/

			//onnx old version-1.8
			output_node_names.push_back(session_.GetOutputName(i, allocator));
		}


		// Run inference
		const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
		const std::array<const char*, 1> outNames = { output_node_names[2].c_str() };

		auto outputs = session_.Run(
			Ort::RunOptions{ nullptr },
			inputNames.data(),
			&input_tensor,
			1,
			outNames.data(),
			1
		);

		// Process output
		float* scores = outputs[0].GetTensorMutableData<float>();
		//*values_evcuate = *scores;
		//return find_max_score_index(scores);
		return get_top_scores(scores, 5);
	}

private:
	Ort::Env env_;
	Ort::Session session_;
	Ort::MemoryInfo memory_info_;  // 使用MemoryInfo替代Allocator

	/*int find_max_score_index(const float* scores) const {
		int max_index = 0;
		float max_score = -FLT_MAX;

		for (int i = 0; i < 521; ++i) {
			if (scores[i] > max_score) {
				max_score = scores[i];
				max_index = i;
			}
		}
		return max_index;
	}*/
	std::vector<TopResult> get_top_scores(const float* scores, int top_k) {
		using ScorePair = std::pair<float, int>;
		std::priority_queue<ScorePair, std::vector<ScorePair>, std::greater<>> min_heap;

		for (int i = 0; i < 521; ++i) {
			if (scores[i] > 0) {
				min_heap.emplace(scores[i], i);
				if (min_heap.size() > top_k) {
					min_heap.pop();
				}
			}
		}

		std::vector<TopResult> results;
		results.reserve(top_k);
		while (!min_heap.empty()) {
			results.emplace_back(TopResult{ min_heap.top().second, min_heap.top().first });
			min_heap.pop();
		}
		std::reverse(results.begin(), results.end());
		return results;
	}
};

class LabelMap {
public:
	explicit LabelMap(const std::string& filename) {
		std::ifstream file(filename);
		if (!file) throw std::runtime_error("Failed to open label file");

		std::string line;
		while (std::getline(file, line)) {
			size_t pos = line.find(' ');
			if (pos != std::string::npos) {
				labels_[line.substr(0, pos)] = line.substr(pos + 1);
			}
		}
	}

	std::string get_label(int index) const {
		auto it = labels_.find(std::to_string(index));
		return (it != labels_.end()) ? it->second : "Unknown";
	}

private:
	std::unordered_map<std::string, std::string> labels_;
};

int main(int argc, char** argv) {
	try 
	{		
		//ffmpeg -i input.wav -c:a pcm_s16le -fflags +bitexact output.wav

		// Load audio
		WavLoader audio_loader("ting.wav");

		// Initialize model
		std::string onnxpath = "yamnet_3s.onnx";
		std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
		OnnxModel model(modelPath);

		// Get label
		LabelMap label_map("yamnet_class_map.txt");

		double start_time = GetTickCount();
		// Run prediction
		/*float values_evcuate = 0;
		int class_index = model.predict(audio_loader.data().data(), &values_evcuate);

		std::cout << "Predicted sound: " << label_map.get_label(class_index) << std::endl;
		std::cout << "Percent: " << values_evcuate << std::endl;*/

		auto results = model.predict(audio_loader.data().data());
		std::cout << "Top 5 Predictions:\n";
		for (const auto& res : results) {
			std::cout << label_map.get_label(res.index)
				<< " : " << std::fixed << std::setprecision(2)
				<< res.score * 100 << "%\n";
		}
		
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
