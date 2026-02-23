#include <NvInfer.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <csignal>
#ifdef __linux__
#include <execinfo.h>
#include <unistd.h>
#include <sys/wait.h>
#endif

#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <regex>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cuda_fp16.h>
#include <NvInfer.h>

// V7: true streaming pipeline (no per-frame cudaStreamSynchronize)
// - Hybrid mode: triple-buffered A(DLA) + B(GPU) with async submit/consume
// - GPU-only mode: triple-buffered async submit/consume
// - Timing is recorded per-frame when the frame's output is actually ready.

namespace fs = std::filesystem;
static bool measure_only = true;

// ---------------- Power logging (tegrastats) ----------------
// tegrastats example (Orin NX):
//   ... VDD_IN 7701mW/7701mW VDD_CPU_GPU_CV ...

struct PowerSample {
    uint64_t t_ns = 0; // steady_clock based
    float p_w = 0.f;   // instant power (W)
    float p_avg_w = 0.f; // optional tegrastats average (W) if present
};

class TegraStatsLogger {
public:
    ~TegraStatsLogger() { stop(); }

    bool start(const std::string& powerKey, int intervalMs) {
        if (running_.load()) return true;
        key_ = powerKey;
        intervalMs_ = intervalMs;
        stop_.store(false);

        // Match both formats:
        //  1) "VDD_IN 7701mW/7701mW"  -> capture inst/avg
        //  2) "POM_5V_IN 2345/3000"   -> capture inst/avg (no mW suffix)
        // Prefer mW form if present.
        re_mw_ = std::regex(key_ + R"(\s+([0-9]+)mW/([0-9]+)mW)");
        re_plain_ = std::regex(key_ + R"(\s+([0-9]+)\s*/\s*([0-9]+))");

        std::string cmd = "tegrastats --interval " + std::to_string(intervalMs_);
        pipe_ = popen(cmd.c_str(), "r");
        if (!pipe_) return false;

        running_.store(true);
        th_ = std::thread([this](){ this->run(); });
        return true;
    }

    void stop() {
        if (!running_.load()) return;
        stop_.store(true);
        if (th_.joinable()) th_.join();
        if (pipe_) {
            pclose(pipe_);
            pipe_ = nullptr;
        }
        running_.store(false);
    }

    std::vector<PowerSample> samples() const {
        std::lock_guard<std::mutex> lk(mu_);
        return samples_;
    }

private:
    void run() {
        char buf[4096];
        while (!stop_.load()) {
            if (!fgets(buf, sizeof(buf), pipe_)) break;

            long mw_inst = -1;
            long mw_avg = -1;
            std::cmatch m;
            if (std::regex_search(buf, m, re_mw_) && m.size() >= 3) {
                mw_inst = std::strtol(m[1].str().c_str(), nullptr, 10);
                mw_avg  = std::strtol(m[2].str().c_str(), nullptr, 10);
            } else if (std::regex_search(buf, m, re_plain_) && m.size() >= 3) {
                // Some tegrastats variants omit "mW" and print e.g. "2345/3000".
                mw_inst = std::strtol(m[1].str().c_str(), nullptr, 10);
                mw_avg  = std::strtol(m[2].str().c_str(), nullptr, 10);
            } else {
                continue;
            }

            auto now = std::chrono::steady_clock::now().time_since_epoch();
            uint64_t t_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

            PowerSample s;
            s.t_ns = t_ns;
            s.p_w = (mw_inst >= 0) ? (mw_inst / 1000.0f) : 0.f;
            s.p_avg_w = (mw_avg >= 0) ? (mw_avg / 1000.0f) : 0.f;

            std::lock_guard<std::mutex> lk(mu_);
            samples_.push_back(s);
        }
    }

    std::string key_;
    int intervalMs_ = 50;
    std::regex re_mw_;
    std::regex re_plain_;

    FILE* pipe_ = nullptr;
    std::thread th_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};

    mutable std::mutex mu_;
    std::vector<PowerSample> samples_;
};

// ---------------- Util logging (jtop exporter; optional) ----------------
// We spawn a small Python helper that uses jetson-stats (jtop) to log GPU/CPU/EMC/DLA utilizations as CSV.
// This avoids fragile tegrastats text parsing for utilization fields (and jtop can expose DLA0/DLA1).
class JtopLogger {
public:
    ~JtopLogger() { stop(); }

    bool start(const std::string& outCsv, int intervalMs, const std::string& pyPath = "jtop_export.py") {
        if (running_.load()) return true;
        outCsv_ = outCsv;
        intervalMs_ = intervalMs;
        pyPath_ = pyPath;

#ifdef __linux__
        // fork/exec: python3 -u jtop_export.py --interval_ms <ms> --out <csv>
        pid_ = fork();
        if (pid_ < 0) {
            std::cerr << "[UTIL] fork() failed; jtop logging disabled\n";
            pid_ = -1;
            return false;
        }
        if (pid_ == 0) {
            // child
            std::string ms = std::to_string(intervalMs_);
            execlp("python3", "python3", "-u",
                   pyPath_.c_str(),
                   "--interval_ms", ms.c_str(),
                   "--out", outCsv_.c_str(),
                   (char*)nullptr);
            // if exec fails:
            std::perror("[UTIL] execlp(python3) failed");
            _exit(127);
        }
        running_.store(true);
        return true;
#else
        (void)outCsv; (void)intervalMs; (void)pyPath;
        std::cerr << "[UTIL] jtop logger only supported on Linux\n";
        return false;
#endif
    }

    void stop() {
        if (!running_.load()) return;
#ifdef __linux__
        if (pid_ > 0) {
            // Ask politely first, then force-kill if needed.
            kill(pid_, SIGTERM);
            int status = 0;
            // wait with a short timeout
            for (int i = 0; i < 50; ++i) { // ~0.5s
                pid_t r = waitpid(pid_, &status, WNOHANG);
                if (r == pid_) { pid_ = -1; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (pid_ > 0) {
                kill(pid_, SIGKILL);
                (void)waitpid(pid_, &status, 0);
                pid_ = -1;
            }
        }
#endif
        running_.store(false);
    }

private:
    std::atomic<bool> running_{false};
    pid_t pid_ = -1;
    std::string outCsv_;
    std::string pyPath_;
    int intervalMs_ = 100;
};


// forward decl (used by power-summary helpers)
static float percentile(std::vector<float> v, double p);

static float mean_f(const std::vector<float>& v){
    double s=0; for(float x:v) s+=x; return v.empty()?0.f:(float)(s/v.size());
}
static float stddev_f(const std::vector<float>& v){
    if (v.size()<2) return 0.f;
    float m = mean_f(v);
    double s=0; for(float x:v){ double d=x-m; s+=d*d; }
    return (float)std::sqrt(s/(v.size()-1));
}
static float median_f(std::vector<float> v){
    if(v.empty()) return 0.f;
    size_t n=v.size()/2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    float med=v[n];
    if(v.size()%2==0){
        std::nth_element(v.begin(), v.begin()+n-1, v.end());
        med = 0.5f*(med + v[n-1]);
    }
    return med;
}
static float mad_f(const std::vector<float>& v){
    if(v.empty()) return 0.f;
    float med = median_f(v);
    std::vector<float> dev; dev.reserve(v.size());
    for(float x:v) dev.push_back(std::fabs(x-med));
    return median_f(dev);
}

static float dpdt_p99_ws(const std::vector<PowerSample>& s){
    if (s.size()<2) return 0.f;
    std::vector<float> v; v.reserve(s.size()-1);
    for(size_t i=1;i<s.size();i++){
        double dt = (double)(s[i].t_ns - s[i-1].t_ns) * 1e-9;
        if (dt <= 0) continue;
        double dp = (double)(s[i].p_w - s[i-1].p_w);
        v.push_back((float)(dp/dt));
    }
    return v.empty()?0.f:percentile(v,99);
}

static void write_power_csv_and_summary(
    const std::string& path,
    const std::string& key,
    const std::vector<PowerSample>& s
){
    if (path.empty()) return;
    if (s.empty()) {
        std::cerr << "[POWER] no samples (key=" << key << ")\n";
        return;
    }

    std::vector<float> p; p.reserve(s.size());
    for (auto& x: s) p.push_back(x.p_w);

    float p_mean = mean_f(p);
    float p_p50  = percentile(p,50);
    float p_p95  = percentile(p,95);
    float p_p99  = percentile(p,99);
    float p_std  = stddev_f(p);
    float p_mad  = mad_f(p);
    float ratio  = (p_mean>0.f)? (p_p99/p_mean) : 0.f;
    float dpdt99 = dpdt_p99_ws(s);

    std::cerr << "[POWER] key=" << key
              << " samples=" << s.size()
              << " mean=" << p_mean
              << " p50=" << p_p50
              << " p95=" << p_p95
              << " p99=" << p_p99
              << " std=" << p_std
              << " mad=" << p_mad
              << " p99/mean=" << ratio
              << " dpdt_p99(W/s)=" << dpdt99
              << "\n";

    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "[POWER] failed to open: " << path << "\n";
        return;
    }
    ofs << "t_ns,p_w,p_avg_w\n";
    for (auto& x: s) ofs << x.t_ns << "," << x.p_w << "," << x.p_avg_w << "\n";
    ofs.close();
    std::cerr << "[POWER] wrote: " << path << "\n";
}

// Frame-level power join: aggregate power samples within [cpu_t_submit_ns, cpu_t_done_ns]
// Writes per-frame CSV and prints jitter summary.
//
// Notes:
// - Uses steady_clock ns for both power samples and FrameTiming timestamps.
// - Integration is approximated as E_frame ≈ P_mean_window * dt_window.
// - If a frame window contains no power samples, fields are left as NaN and excluded from summary.
struct FrameTiming {
    int frame = -1;
    int slot  = -1;              // ring slot used (debug)
    std::string name;

    // CPU stage times (host-side)
    float cpu_decode_ms     = 0.f;  // decode / image load
    float cpu_queue_wait_ms = 0.f;  // 슬롯을 덮어쓰기 전에 슬롯 출력이 끝났는지 확인하는 시간
    float cpu_sleep_ms      = 0.f;  // --fps 모드 일 떄, 의도적으로 쉰 시간
    float cpu_submit_ms     = 0.f;  // submit/enqueue 등 호출하는 시간
    float cpu_block_ms      = 0.f;  // 동기화 호출로 멈춘 시간
    float cpu_post_ms       = 0.f;  // 후처리 시간
    float cpu_write_ms      = 0.f;  // 기록 시간

    // Existing preprocess breakdown
    float pre_meta_ms = 0.f;
    float pre_lb_ms   = 0.f;    //Letterbox 처리시간
    float pre_pack_ms = 0.f;    //입력 변경 처리 시간
    float pre_ms      = 0.f;    //총 시간

    // Device timings
    float h2d_ms  = 0.f;    // in
    float a_ms    = 0.f;    // dla 실행 시간
    float wait_ms = 0.f;    // gpu가 dla를 기다린 시간
    float b_ms    = 0.f;    // gpu 실행 시간
    float d2h_ms  = 0.f;    // out

    // Post / end-to-end
    float post_ms     = 0.f;        // 출력 준비 완료 부터 CPU post 끝(기록) 까지
    float e2e_ms      = 0.f;        // e2e_cap_ms와 같음
    float e2e_cap_ms  = 0.f;        // capture/service latency (includes pacing/queueing)
    float e2e_proc_ms = 0.f;        // processing latency (excludes pacing sleep)

    // CPU timestamps (steady_clock based, in ns). Optional but useful for debugging.
    uint64_t cpu_t_cap_ns    = 0;   // capture / read done
    uint64_t cpu_t_pre_ns    = 0;   // preprocess begin
    uint64_t cpu_t_submit_ns = 0;   // first submit (H2D/enqueue) begin
    uint64_t cpu_t_done_ns   = 0;   // consume/post done

    // Flags
    int dropped = 0;
    int paced   = 0;
};
struct FramePowerAgg {
    int frame = -1;
    int slot  = -1;
    uint64_t t0_ns = 0;
    uint64_t t1_ns = 0;
    double dt_s = 0.0;

    int n = 0;
    double p_sum = 0.0;
    double p_avg_sum = 0.0;
    float p_max = 0.f;

    float p_mean = std::numeric_limits<float>::quiet_NaN();
    float p_avg_mean = std::numeric_limits<float>::quiet_NaN();
    float e_j = std::numeric_limits<float>::quiet_NaN(); // Joules
};

static void write_power_frames_csv_and_summary(
    const std::string& path,
    const std::string& key,
    const std::vector<PowerSample>& s,
    const std::vector<FrameTiming>& frames
){
    if (path.empty()) return;
    if (s.empty() || frames.empty()) {
        std::cerr << "[POWER_FR] skipped (samples=" << s.size() << ", frames=" << frames.size() << ")\n";
        return;
    }

    // Ensure samples are time-sorted (they should already be)
    // One-pass join using moving index.
    size_t i = 0;
    std::vector<FramePowerAgg> out;
    out.reserve(frames.size());

    for (const auto& f : frames) {
        FramePowerAgg a;
        a.frame = f.frame;
        a.slot  = f.slot;
        a.t0_ns = f.cpu_t_submit_ns;
        a.t1_ns = f.cpu_t_done_ns;
        if (a.t0_ns == 0 || a.t1_ns == 0 || a.t1_ns <= a.t0_ns) {
            out.push_back(a);
            continue;
        }
        a.dt_s = (double)(a.t1_ns - a.t0_ns) * 1e-9;

        // advance i to first sample >= t0
        while (i < s.size() && s[i].t_ns < a.t0_ns) i++;

        // aggregate samples within window
        size_t j = i;
        for (; j < s.size() && s[j].t_ns <= a.t1_ns; j++) {
            a.n++;
            a.p_sum += (double)s[j].p_w;
            a.p_avg_sum += (double)s[j].p_avg_w;
            a.p_max = std::max(a.p_max, s[j].p_w);
        }

        if (a.n > 0) {
            a.p_mean = (float)(a.p_sum / (double)a.n);
            a.p_avg_mean = (float)(a.p_avg_sum / (double)a.n);
            a.e_j = (float)(a.p_mean * a.dt_s); // coarse integration
        }

        // Keep i where it is (do not rewind). Next frame times are non-decreasing.
        out.push_back(a);
    }

    // Write CSV
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "[POWER_FR] failed to open: " << path << "\n";
        return;
    }
    ofs << "frame,slot,t0_ns,t1_ns,dt_s,n,p_mean_w,p_max_w,p_avg_mean_w,e_j\n";
    for (const auto& a : out) {
        ofs << a.frame << "," << a.slot << ","
            << a.t0_ns << "," << a.t1_ns << ","
            << a.dt_s << "," << a.n << ",";
        if (std::isfinite(a.p_mean)) {
            ofs << a.p_mean << "," << a.p_max << "," << a.p_avg_mean << "," << a.e_j;
        } else {
            ofs << "nan,nan,nan,nan";
        }
        ofs << "\n";
    }
    ofs.close();

    // Summary over frames with valid samples
    std::vector<float> pmean, pmax, ej;
    pmean.reserve(out.size());
    pmax.reserve(out.size());
    ej.reserve(out.size());
    for (const auto& a : out) {
        if (!std::isfinite(a.p_mean)) continue;
        pmean.push_back(a.p_mean);
        pmax.push_back(a.p_max);
        ej.push_back(a.e_j);
    }
    if (pmean.empty()) {
        std::cerr << "[POWER_FR] wrote: " << path << " (no valid frame windows)\n";
        return;
    }

    auto pr = [&](const char* name, const std::vector<float>& v){
        std::vector<float> vv = v;
        float m = mean_f(vv);
        float p50 = percentile(vv,50);
        float p95 = percentile(vv,95);
        float p99 = percentile(vv,99);
        float sd = stddev_f(vv);
        float md = mad_f(vv);
        float ratio = (m>0.f)? (p99/m) : 0.f;
        std::cerr << "[POWER_FR] " << name
                  << " mean=" << m
                  << " p50=" << p50
                  << " p95=" << p95
                  << " p99=" << p99
                  << " std=" << sd
                  << " mad=" << md
                  << " p99/mean=" << ratio
                  << "\n";
    };

    std::cerr << "[POWER_FR] key=" << key << " frames_valid=" << pmean.size()
              << " wrote: " << path << "\n";
    pr("Pmean(W)", pmean);
    pr("Pmax(W)",  pmax);
    pr("E/frame(J)", ej);
}



struct CachedImage {
    std::string path;
    cv::Mat bgr;
};

static void printDims(const nvinfer1::Dims& d) {
    std::cerr << "[dims] nbDims=" << d.nbDims << " : [";
    for (int i=0;i<d.nbDims;i++) {
        std::cerr << d.d[i] << (i+1<d.nbDims ? "," : "");
    }
    std::cerr << "]\n";
}

void dumpIO(nvinfer1::ICudaEngine& engine, nvinfer1::IExecutionContext& ctx) {
    int n = engine.getNbIOTensors();
    std::cerr << "NbIOTensors=" << n << "\n";
    for (int i=0;i<n;i++) {
        const char* name = engine.getIOTensorName(i);
        auto mode = engine.getTensorIOMode(name);
        std::cerr << (mode==nvinfer1::TensorIOMode::kINPUT ? "[IN ] " : "[OUT] ")
                  << name << " ";
        auto d = ctx.getTensorShape(name); // 실행 전에/후에 shape 확인 가능
        printDims(d);
    }
}
static std::vector<CachedImage> preload_images(const std::vector<std::string>& paths) {
    std::vector<CachedImage> cache;
    cache.reserve(paths.size());
    size_t ok = 0, fail = 0;
    for (const auto& p : paths) {
        cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
        if (img.empty()) {
            fprintf(stderr, "[WARN] imread failed: %s\n", p.c_str());
            ++fail;
            continue;
        }
        cache.push_back({p, img});
        ++ok;
    }
    fprintf(stderr, "[PRELOAD] loaded=%zu failed=%zu\n", ok, fail);
    return cache;
}

struct LetterboxParams {
    int in_w = 640, in_h = 640;
    float r = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
    int orig_w = 0, orig_h = 0;
};

struct Detection {
    int cls = -1;
    float conf = 0.0f;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
};

struct CocoDet {
    int image_id = -1;      // COCO image id
    int category_id = -1;   // COCO category id (not contiguous)
    float x = 0.f, y = 0.f, w = 0.f, h = 0.f; // COCO bbox: xywh in pixels
    float score = 0.f;
};

// COCO 80-class category_id mapping for the standard COCO label order used by YOLO models.
// If your model uses a different class order, adjust this table.
static inline int coco_category_id_from_class(int cls) {
    static const int kCoco80Ids[80] = {
        1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,
        22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,
        58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,
        78,79,80,81,82,84,85,86,87,88,89,90
    };
    if (cls < 0 || cls >= 80) return -1;
    return kCoco80Ids[cls];
}

// Parse COCO image_id from file name like "000000397133.jpg" -> 397133.
// Returns -1 if parsing fails.
static inline int coco_image_id_from_filename(const std::string& file_name) {
    // Strip directory
    size_t p = file_name.find_last_of("/\\\\");
    std::string base = (p == std::string::npos) ? file_name : file_name.substr(p + 1);
    // Strip extension
    size_t dot = base.find_last_of('.');
    std::string stem = (dot == std::string::npos) ? base : base.substr(0, dot);
    // Parse integer (allow leading zeros)
    try {
        size_t idx = 0;
        int id = std::stoi(stem, &idx);
        if (idx == 0) return -1;
        return id;
    } catch (...) {
        return -1;
    }
}

static inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char uc : s) {
        char c = static_cast<char>(uc);
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (uc < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)uc);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static void write_coco_detections_json(const std::string& path, const std::vector<CocoDet>& dets) {
    printf("%s", path.c_str());
    std::ofstream ofs(path);
    if (!ofs) throw std::runtime_error("Failed to open predictions json for write: " + path);
    ofs << "[\n";
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        ofs << "{"
            << "\"image_id\":" << d.image_id << ","
            << "\"category_id\":" << d.category_id << ","
            << "\"bbox\":[" << d.x << "," << d.y << "," << d.w << "," << d.h << "],"
            << "\"score\":" << d.score
            << "}";
        if (i + 1 < dets.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "]\n";
    ofs.flush();
}


static float percentile(std::vector<float> v, double p) {
    if (v.empty()) return 0.f;
    std::sort(v.begin(), v.end());
    double pos = (p / 100.0) * (double)(v.size() - 1);
    size_t i0 = (size_t)std::floor(pos);
    size_t i1 = (size_t)std::ceil(pos);
    if (i0 == i1) return v[i0];
    double t = pos - (double)i0;
    return (float)((1.0 - t) * v[i0] + t * v[i1]);

}

static inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

static inline float iou_xyxy(const Detection& a, const Detection& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    float uni = area_a + area_b - inter;
    return (uni > 0.0f) ? (inter / uni) : 0.0f;
}

// 기본은 class-aware, 단 IoU가 아주 높으면 (다른 클래스라도) 억제하는 NMS
static std::vector<Detection> nms_class_aware(
    std::vector<Detection>& dets,
    float iou_thres_same_cls = 0.45f
) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.conf > b.conf; });

    std::vector<Detection> out;
    out.reserve(dets.size());
    std::vector<uint8_t> suppressed(dets.size(), 0);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;

        out.push_back(dets[i]);

        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;

            // 다른 클래스는 비교 자체를 skip하면 조금 더 빠름
            if (dets[i].cls != dets[j].cls) continue;

            const float iou = iou_xyxy(dets[i], dets[j]);
            if (iou > iou_thres_same_cls) suppressed[j] = 1;
        }
    }
    return out;
}



static cv::Mat letterbox_640(const cv::Mat& img_bgr, LetterboxParams& lb) {
    lb.orig_w = img_bgr.cols;
    lb.orig_h = img_bgr.rows;
    lb.in_w = 640;
    lb.in_h = 640;

    float r = std::min(lb.in_w / (float)lb.orig_w, lb.in_h / (float)lb.orig_h);
    int new_w = (int)std::round(lb.orig_w * r);
    int new_h = (int)std::round(lb.orig_h * r);
    float pad_x = (lb.in_w - new_w) / 2.0f;
    float pad_y = (lb.in_h - new_h) / 2.0f;
    lb.r = r;
    lb.pad_x = pad_x;
    lb.pad_y = pad_y;

    cv::Mat resized;
    cv::resize(img_bgr, resized, cv::Size(new_w, new_h));
    cv::Mat out(lb.in_h, lb.in_w, CV_8UC3, cv::Scalar(114,114,114));
    resized.copyTo(out(cv::Rect((int)pad_x, (int)pad_y, new_w, new_h)));
    return out;
}

static void save_preds_yolo(
    const std::string& out_txt_path,
    const std::vector<Detection>& dets,
    int orig_w, int orig_h
){
    std::filesystem::create_directories(std::filesystem::path(out_txt_path).parent_path());
    FILE* f = std::fopen(out_txt_path.c_str(), "w");
    if (!f) {
        std::perror(("fopen failed: " + out_txt_path).c_str());
        return;
    }
    auto clamp01 = [](float v){ return std::max(0.f, std::min(1.f, v)); };
    
    for (const auto& det : dets) {
        float cx = 0.5f * (det.x1 + det.x2) / orig_w;
        float cy = 0.5f * (det.y1 + det.y2) / orig_h;
        float w  = (det.x2 - det.x1) / orig_w;
        float h  = (det.y2 - det.y1) / orig_h;
        cx = clamp01(cx); cy = clamp01(cy);
        w  = clamp01(w);  h  = clamp01(h);
        if (w <= 0.f || h <= 0.f) continue;
        std::fprintf(f, "%d %.6f %.6f %.6f %.6f %.6f\n",
                     det.cls, cx, cy, w, h, det.conf);
    }
    std::fclose(f);
}

static inline void fill_input_fp32_nchw_rgb01_direct(const cv::Mat& bgr, float* dst) {
    constexpr int H = 640;
    constexpr int W = 640;
    constexpr int HW = H * W;
    constexpr float inv255 = 1.0f / 255.0f;

    float* outR = dst + 0 * HW;
    float* outG = dst + 1 * HW;
    float* outB = dst + 2 * HW;

    for (int y = 0; y < H; ++y) {
        const uint8_t* row = bgr.ptr<uint8_t>(y);
        int base = y * W;
        for (int x = 0; x < W; ++x) {
            const uint8_t B = row[3*x + 0];
            const uint8_t G = row[3*x + 1];
            const uint8_t R = row[3*x + 2];
            int idx = base + x;
            outR[idx] = R * inv255;
            outG[idx] = G * inv255;
            outB[idx] = B * inv255;
        }
    }
}

static inline void fill_input_fp16_nchw_rgb01_direct(const cv::Mat& bgr, __half* dst){
    constexpr int H=640, W=640, HW=H*W;
    constexpr float inv255 = 1.f/255.f;
    __half* outR = dst + 0*HW;
    __half* outG = dst + 1*HW;
    __half* outB = dst + 2*HW;
    for(int y=0;y<H;y++){
        const uint8_t* row = bgr.ptr<uint8_t>(y);
        int base=y*W;
        for(int x=0;x<W;x++){
            uint8_t B=row[3*x+0], G=row[3*x+1], R=row[3*x+2];
            int idx=base+x;
            outR[idx]=__float2half(R*inv255);
            outG[idx]=__float2half(G*inv255);
            outB[idx]=__float2half(B*inv255);
        }
    }
}

static inline float _sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static std::vector<Detection> decode_yolo84_8400(
    const float* out,
    const LetterboxParams& lb,
    float conf_thres = 0.25f,
    float iou_thres1 = 0.45f,
    int   max_det    = 300,
    bool  apply_sigmoid_to_cls = false
) {

    constexpr int K  = 8400;
    constexpr int NC = 80;

    // C-major: out[c*K + k]
    auto at = [&](int c, int k) -> float { return out[c * K + k]; };

    std::vector<Detection> candidates;
    candidates.reserve(5000);

    for (int k = 0; k < K; ++k) {
        float cx = at(0, k);
        float cy = at(1, k);
        float w  = at(2, k);
        float h  = at(3, k);

        // (안전) w,h가 음수/0이면 스킵
        if (!(w > 0.0f && h > 0.0f) || !std::isfinite(w) || !std::isfinite(h)) continue;

        int best_cls = -1;
        float best_conf = -1.0f;
        for (int c = 0; c < NC; ++c) {
            float s = at(4 + c, k);
            if (!std::isfinite(s)) continue;
            float score = apply_sigmoid_to_cls ? sigmoidf(s) : s;
            if (score > best_conf) { best_conf = score; best_cls = c; }
        }

        if (best_conf < conf_thres) { continue; }

        float x1_in = cx - 0.5f * w;
        float y1_in = cy - 0.5f * h;
        float x2_in = cx + 0.5f * w;
        float y2_in = cy + 0.5f * h;

        // letterbox 역변환
        float x1 = (x1_in - lb.pad_x) / lb.r;
        float y1 = (y1_in - lb.pad_y) / lb.r;
        float x2 = (x2_in - lb.pad_x) / lb.r;
        float y2 = (y2_in - lb.pad_y) / lb.r;

        x1 = clampf(x1, 0.0f, (float)(lb.orig_w - 1));
        y1 = clampf(y1, 0.0f, (float)(lb.orig_h - 1));
        x2 = clampf(x2, 0.0f, (float)(lb.orig_w - 1));
        y2 = clampf(y2, 0.0f, (float)(lb.orig_h - 1));

        if (x2 <= x1 || y2 <= y1) continue;
        candidates.push_back({best_cls, best_conf, x1, y1, x2, y2});
    }

    if (candidates.empty()) return {};
    auto kept = nms_class_aware(candidates, iou_thres1);
    if ((int)kept.size() > max_det) kept.resize(max_det);
    return kept;
}


#define CHECK_CUDA(call) do {                               \
  cudaError_t _e = (call);                                  \
  if (_e != cudaSuccess) {                                  \
    std::cerr << "CUDA error: " << cudaGetErrorString(_e)    \
              << " at " << __FILE__ << ":" << __LINE__      \
              << std::endl;                                 \
    std::exit(1);                                           \
  }                                                         \
} while(0)

// ---- Crash diagnostics (helps pinpoint silent SIGSEGV) ----
static volatile int g_stage = 0;
static volatile int g_frame = -1;
static volatile int g_slot  = -1;

static void segv_handler(int sig) {
    fprintf(stderr, "\n[FATAL] signal=%d stage=%d frame=%d slot=%d\n", sig, g_stage, g_frame, g_slot);
#ifdef __linux__
    void* bt[64];
    int n = backtrace(bt, 64);
    backtrace_symbols_fd(bt, n, 2);
#endif
    _Exit(128 + sig);
}
// ----------------------------------------------------------

static float read_input_pixel_device(
    void* dAin, nvinfer1::DataType dt,
    int c, int y, int x,
    cudaStream_t stream
){
    constexpr int H=640, W=640;
    size_t idx = (size_t)c*H*W + (size_t)y*W + (size_t)x;

    float out = 0.f;

    if (dt == nvinfer1::DataType::kFLOAT) {
        float v;
        CHECK_CUDA(cudaMemcpyAsync(&v, ((float*)dAin) + idx, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        out = v;
    } else if (dt == nvinfer1::DataType::kHALF) {
        __half hv;
        CHECK_CUDA(cudaMemcpyAsync(&hv, ((const __half*)dAin) + idx, sizeof(__half),
                                   cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        out = __half2float(hv);
    } else {
        printf("Unsupported input dtype %d\n", (int)dt);
    }
    return out;
}
void dump_bytes_async(const void* dptr, size_t offset, size_t nbytes,
                      cudaStream_t stream, const char* tag)
{
    std::vector<unsigned char> h(nbytes);

    // 반드시 에러 체크
    CHECK_CUDA(cudaMemcpyAsync(
        h.data(),
        static_cast<const unsigned char*>(dptr) + offset,
        nbytes,
        cudaMemcpyDeviceToHost,
        stream
    ));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("%s (offset=%zu):", tag, offset);
    for (size_t i = 0; i < nbytes; ++i) printf(" %02X", h[i]);
    printf("\n");
}

void stats_feat_fp32(const float* dFeat, size_t N, cudaStream_t stream)
{
    std::vector<float> h(N);
    CHECK_CUDA(cudaMemcpyAsync(h.data(), dFeat, N*sizeof(float),
                            cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    double sum = 0.0;

    for (size_t i=0; i<N; ++i) {
        float v = h[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
    }
    printf("feat stats(fp32): min=%g max=%g mean=%g\n", mn, mx, sum / (double)N);
}

class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << msg << "\n";
  }
};

static std::vector<char> readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Failed to open: " + path);
  f.seekg(0, std::ios::end);
  size_t sz = (size_t)f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> buf(sz);
  f.read(buf.data(), sz);
  return buf;
}

static size_t dtypeSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT8:  return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL:  return 1;
    default: throw std::runtime_error("Unknown DataType");
  }
}

struct SampleStats {
    float min = 0.f;
    float max = 0.f;
    double sum = 0.0;
    double mean = 0.0;
    size_t count;

};


static inline float read_elem_as_float(const void* buf, nvinfer1::DataType dt, size_t i) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return ((const float*)buf)[i];
        case nvinfer1::DataType::kHALF:  return __half2float(((const __half*)buf)[i]);
        case nvinfer1::DataType::kINT8:  return (float)((const int8_t*)buf)[i];
        case nvinfer1::DataType::kINT32: return (float)((const int32_t*)buf)[i];
        default: return 0.f;
    }
}
static SampleStats sample_stats_host(const void* buf, nvinfer1::DataType dt, size_t bytes, size_t max_elems = 8400) {
    const size_t esz = dtypeSize(dt);
    const size_t elems = (esz == 0) ? 0 : std::min(bytes / esz, max_elems);
    SampleStats s;
    s.min = std::numeric_limits<float>::infinity();
    s.max = -std::numeric_limits<float>::infinity();
    s.sum = 0.0;

    auto read_as_float = [&](size_t i) -> float {
        switch (dt) {
            case nvinfer1::DataType::kFLOAT: return ((const float*)buf)[i];
            case nvinfer1::DataType::kHALF:  return __half2float(((const __half*)buf)[i]);
            case nvinfer1::DataType::kINT8:  return (float)((const int8_t*)buf)[i];
            case nvinfer1::DataType::kINT32: return (float)((const int32_t*)buf)[i];
            default: return 0.f;
        }
    };

    for (size_t i=0;i<elems;i++) {
        const float v = read_as_float(i);
        s.min = std::min(s.min, v);
        s.max = std::max(s.max, v);
        s.sum += (double)v;
    }
    if (elems == 0) { s.min = s.max = 0.f; s.sum = 0.0; }
    return s;
}

static std::vector<SampleStats>
channel_stats_cmajor_host(const void* buf,
                          nvinfer1::DataType dt,
                          size_t bytes,
                          int C,
                          int K,
                          size_t max_k = 0)
{
    std::vector<SampleStats> out;
    out.resize((size_t)C);

    const size_t esz = dtypeSize(dt);
    if (esz == 0 || buf == nullptr || C <= 0 || K <= 0) {
        for (auto& s : out) { s.min = s.max = 0.f; s.sum = s.mean = 0.0; s.count = 0; }
        return out;
    }

    // buf가 실제로 갖고 있는 element 수
    const size_t total_elems = bytes / esz;
    const size_t needed = (size_t)C * (size_t)K;

    // 방어: 버퍼가 작으면 가능한 범위까지만
    const size_t safe_total = std::min(total_elems, needed);
    const int safe_C = (int)std::min<size_t>((size_t)C, safe_total / (size_t)K);

    const size_t kk = (max_k == 0) ? (size_t)K : std::min(max_k, (size_t)K);

    for (int c = 0; c < C; ++c) {
        SampleStats s;
        s.min = std::numeric_limits<float>::infinity();
        s.max = -std::numeric_limits<float>::infinity();
        s.sum = 0.0;
        s.mean = 0.0;
        s.count = 0;

        // 버퍼 부족 시 나머지 채널은 0 처리
        if (c >= safe_C) {
            s.min = s.max = 0.f;
            out[(size_t)c] = s;
            continue;
        }

        const size_t base = (size_t)c * (size_t)K;

        for (size_t k = 0; k < kk; ++k) {
            const size_t idx = base + k;
            if (idx >= safe_total) break;

            const float v = read_elem_as_float(buf, dt, idx);
            s.min = std::min(s.min, v);
            s.max = std::max(s.max, v);
            s.sum += (double)v;
            s.count++;
        }

        if (s.count == 0) {
            s.min = s.max = 0.f;
            s.sum = 0.0;
            s.mean = 0.0;
        } else {
            s.mean = s.sum / (double)s.count;
        }

        out[(size_t)c] = s;
    }

    return out;
}

static int64_t vol(const nvinfer1::Dims& d) {
  int64_t v = 1;
  for (int i=0;i<d.nbDims;i++) v *= d.d[i];
  return v;
}

static size_t tensorBytes(nvinfer1::ICudaEngine* eng, const char* name) {
  auto dt = eng->getTensorDataType(name);
  auto dims = eng->getTensorShape(name);
  if (dims.nbDims < 0) throw std::runtime_error(std::string("Bad dims for ") + name);
  return (size_t)vol(dims) * dtypeSize(dt);
}

static void printTensorInfo(nvinfer1::ICudaEngine* eng, const char* name) {
  auto dt = eng->getTensorDataType(name);
  auto dims = eng->getTensorShape(name);
  std::cout << "  " << name << "  dtype=" << (int)dt << "  dims=[";
  for (int i=0;i<dims.nbDims;i++) {
    std::cout << dims.d[i] << (i+1<dims.nbDims ? "," : "");
  }
  std::cout << "] bytes=" << tensorBytes(eng, name) << "\n";
}

struct TrtEngine {
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx;
};

static TrtEngine loadEngine(const std::string& planPath, Logger& logger) {
  auto blob = readFile(planPath);
  TrtEngine te;
  te.runtime.reset(nvinfer1::createInferRuntime(logger));
  if (!te.runtime) throw std::runtime_error("createInferRuntime failed");
  te.engine.reset(te.runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!te.engine) throw std::runtime_error("deserializeCudaEngine failed: " + planPath);
  te.ctx.reset(te.engine->createExecutionContext());
  if (!te.ctx) throw std::runtime_error("createExecutionContext failed");
  return te;
}

struct CliArgs {
    bool idle_pipeline = false;  // C-2 baseline: run IO+preprocess(+throttle) but skip TRT enqueue/copies
    enum class Mode { Hybrid, Gpu } mode = Mode::Hybrid;
    enum class Sched { Throughput, Latency } sched = Sched::Throughput;

    enum class FpsCapMode { Off, Input, Done } fps_cap_mode = FpsCapMode::Off;

    std::string planA;
    std::string planB;
    std::string planGPU;
    std::string inputPath;
    std::string timingCsv;
    std::string predJson; // write COCO detection results (.json)
    double fps = 0.0; // 0 = unthrottled (as-fast-as-possible)

    // Power logging (tegrastats)
    // Util logging (jtop)
    std::string utilCsv;          // jtop raw utilization samples CSV (empty=disabled)
    std::string utilPy = "jtop_export.py"; // path to jtop_export.py
    int utilIntervalMs = 100;     // jtop sampling interval (ms)

    std::string powerCsv;             // e.g., power.csv (empty=disabled)
    std::string powerFramesCsv;       // per-frame aggregated power CSV (optional)
    std::string powerKey = "VDD_IN"; // tegrastats field to parse
    int powerIntervalMs = 50;         // tegrastats --interval

    // [New] Thresholds
    float confThres = 0.25f;  // Default for speed bench
    float iouThres = 0.45f;   // Default NMS iou
    int maxDet = 300;         // Max detections per image

    // Debug / sanity-check options
    bool debug_io = false;        // print per-frame input/output sample stats
    bool debug_ptrs = false;      // print buffer addresses and dtypes once
    bool debug_feat = false;      // sample A feature tensors before running B
    bool debug_zero_out = false;  // memset B output buffer to 0 before enqueue
    
    //connect tensor
    std::string connectTensorNames = "";
};

static void printUsage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  --idle_pipeline       Baseline mode: keep IO/preprocess/throttle, but skip TRT enqueue/copies\n"
        << "  (hybrid/split, default)\n"
        << "    " << prog << " partA.plan partB.plan <image_or_folder>\n"
        << "  (explicit hybrid)\n"
        << "    " << prog << " --mode=split partA.plan partB.plan <image_or_folder>\n"
        << "  (full gpu-only)\n"
        << "    " << prog << " --mode=gpu full.plan <image_or_folder>\n"
        << "Options:\n"
        << "  --timing=timing.csv   Write per-frame timing CSV\n";
    std::cerr
        << "  --fps=30              Throttle input to fixed FPS (e.g., 30 or 60). 0 = no throttle\n";
    std::cerr
        << "  --fps_cap_mode=off    FPS cap semantics: off|input|done (default: off)\n"
        << "                        input: cap submit/capture cadence (camera-like)\n"
        << "                        done : cap completion cadence (consume/done-based; best for fair E/frame)\n";
    std::cerr
        << "  --sched=throughput    Hybrid scheduling: throughput (pipeline A(f)+B(f-1)) [default]\n"
        << "  --sched=latency       Hybrid scheduling: latency (A(f)->B(f) immediately; lower FPS)\n";
    std::cerr
        << "Power options:\n"
        << "  --power=power.csv     Enable tegrastats logging (writes raw samples + prints summary)\n"
        << "  --power_key=VDD_IN    tegrastats key to parse (see 'tegrastats --interval 100')\n"
        << "  --power_ms=50         tegrastats interval in ms (typ. 20~200)\n"
        << "  --power_frames=pf.csv Write per-frame power aggregates (default: <power>.frames.csv)\n";
    std::cerr
        << "Util options:\n"
        << "  --util=util.csv      Enable jtop utilization logging (GPU/CPU/EMC/DLA)\n"
        << "  --util_ms=100        jtop sampling interval in ms (typ. 50~200)\n"
        << "  --util_py=jtop_export.py  Path to the Python exporter (default: jtop_export.py)\n";
    std::cerr
        << "Debug options:\n"
        << "  --debug_io            Print per-frame input/output sample stats\n"
        << "  --debug_ptrs          Print buffer addresses and TensorRT dtypes\n"
        << "  --debug_feat          Sample A feature tensors before running B\n"
        << "  --debug_zero_out      Zero B output buffer before enqueue (detect missing writes)\n"
        << "  --debug               Enable all debug options (--debug_io/--debug_ptrs/--debug_feat/--debug_zero_out)\n";

}

static inline const char* to_string(CliArgs::FpsCapMode m) {
    switch (m) {
        case CliArgs::FpsCapMode::Off:   return "off";
        case CliArgs::FpsCapMode::Input: return "input";
        case CliArgs::FpsCapMode::Done:  return "done";
        default:                         return "off";
    }
}

static inline CliArgs::FpsCapMode parse_fps_cap_mode(const std::string& s) {
    if (s == "off")   return CliArgs::FpsCapMode::Off;
    if (s == "input") return CliArgs::FpsCapMode::Input;
    if (s == "done")  return CliArgs::FpsCapMode::Done;
    throw std::runtime_error("Unknown --fps_cap_mode=" + s + " (use off|input|done)");
}

static std::vector<std::string> splitString(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

static bool startsWith(const std::string& fullString, const std::string& prefix) {
    return fullString.rfind(prefix, 0) == 0;
}

static CliArgs parseArgs(int argc, char** argv) {
    CliArgs cli;
    std::vector<std::string> pos; // 파일 경로(Positional Args) 저장용

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];

        // -------------------------------------------------------
        // 1. Key=Value 형태의 옵션 파싱 (startsWith 사용)
        // -------------------------------------------------------
        if (startsWith(a, "--mode=")) {
            auto v = a.substr(7);
            if (v == "gpu") cli.mode = CliArgs::Mode::Gpu;
            else if (v == "split" || v == "hybrid") cli.mode = CliArgs::Mode::Hybrid;
            else throw std::runtime_error("Unknown --mode=" + v);
        } 
        else if (startsWith(a, "--timing=")) {
            cli.timingCsv = a.substr(9);
        } 
        else if (startsWith(a, "--pred_json=")) {
            cli.predJson = a.substr(12);
        } 
        else if (startsWith(a, "--fps=")) {
            cli.fps = std::stod(a.substr(6));
        } 
        else if (startsWith(a, "--fps_cap_mode=")) {
            cli.fps_cap_mode = parse_fps_cap_mode(a.substr(15));
        } 
        else if (startsWith(a, "--sched=")) {
            auto v = a.substr(8);
            if (v == "throughput" || v == "pipe" || v == "pipelined") cli.sched = CliArgs::Sched::Throughput;
            else if (v == "latency" || v == "serial") cli.sched = CliArgs::Sched::Latency;
            else throw std::runtime_error("Unknown --sched=" + v);
        } 
        else if (startsWith(a, "--power=")) {
            cli.powerCsv = a.substr(8);
        } 
        else if (startsWith(a, "--power_key=")) {
            cli.powerKey = a.substr(12);
        } 
        else if (startsWith(a, "--power_ms=")) {
            cli.powerIntervalMs = std::stoi(a.substr(11));
        } 
        else if (startsWith(a, "--power_frames=")) {
            cli.powerFramesCsv = a.substr(15);
        } 
        else if (startsWith(a, "--util=")) {
            cli.utilCsv = a.substr(7);
        } 
        else if (startsWith(a, "--util_ms=")) {
            cli.utilIntervalMs = std::stoi(a.substr(10));
        } 
        else if (startsWith(a, "--util_py=")) {
            cli.utilPy = a.substr(10);
        }
        
        // [New] Thresholds & Connection
        else if (startsWith(a, "--conf_thres=")) {
            cli.confThres = std::stod(a.substr(13)); 
        } 
        else if (startsWith(a, "--iou_thres=")) {
            cli.iouThres = std::stod(a.substr(12)); 
        } 
        else if (startsWith(a, "--max_det=")) {
            cli.maxDet = std::stoi(a.substr(10)); 
        } 
        else if (startsWith(a, "--connect=")) {
            cli.connectTensorNames = a.substr(10); 
        }

        // -------------------------------------------------------
        // 2. Boolean Flags (단순 일치 확인)
        // -------------------------------------------------------
        else if (a == "--idle_pipeline") {
            cli.idle_pipeline = true;
        } 
        else if (a == "--debug_io") {
            cli.debug_io = true;
        } 
        else if (a == "--debug_ptrs") {
            cli.debug_ptrs = true;
        } 
        else if (a == "--debug_feat") {
            cli.debug_feat = true;
        } 
        else if (a == "--debug_zero_out") {
            cli.debug_zero_out = true;
        } 
        else if (a == "--debug") {
            cli.debug_io = cli.debug_ptrs = cli.debug_feat = cli.debug_zero_out = true;
        } 
        else if (a == "-h" || a == "--help") {
            // printUsage(argv[0]); 
            std::exit(0);
        } 
        
        // -------------------------------------------------------
        // 3. Positional Arguments (파일 경로 등)
        // -------------------------------------------------------
        else {
            // '-'로 시작하지 않는 인자는 파일 경로로 간주
            if (a.rfind("-", 0) != 0) { 
                pos.push_back(a);
            } else {
                std::cerr << "[Warning] Unknown option ignored: " << a << std::endl;
            }
        }
    }

    // -------------------------------------------------------
    // [Bottom Part] 파일 경로 할당 로직
    // -------------------------------------------------------
    if (cli.mode == CliArgs::Mode::Hybrid) {
        // Hybrid 모드: 엔진 파일 2개 + 이미지 폴더 1개 = 총 3개 필요
        if (pos.size() != 3) {
            // printUsage(argv[0]);
            throw std::runtime_error("Hybrid mode requires 3 args: <DLA.plan> <GPU.plan> <ImageDir>");
        }
        cli.planA  = pos[0]; // DLA part
        cli.planB = pos[1]; // GPU part
        cli.inputPath    = pos[2]; // Images
    } else {
        // GPU 모드: 엔진 파일 1개 + 이미지 폴더 1개 = 총 2개 필요
        if (pos.size() != 2) {
            // printUsage(argv[0]);
            throw std::runtime_error("GPU mode requires 2 args: <Engine.plan> <ImageDir>");
        }
        cli.planGPU = pos[0]; // Single engine
        cli.inputPath   = pos[1]; // Images
    }

    // 파생 옵션 자동 설정
    if (!cli.powerCsv.empty() && cli.powerFramesCsv.empty()) {
        cli.powerFramesCsv = cli.powerCsv + ".frames.csv";
    }

    if (cli.fps <= 0.0) {
        cli.fps_cap_mode = CliArgs::FpsCapMode::Off; 
    }

    return cli;
}

static void writeTimingCsv(const std::string& path, const std::vector<FrameTiming>& t) {
    if (path.empty()) return;
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "Failed to open timing csv: " << path << "\n";
        return;
    }
    ofs << "frame,name,pre_meta,pre_lb,pre_pack,pre_total,h2d,a,wait,b,d2h,post,e2e,e2e_cap,e2e_proc";
    for (auto& r : t) {
        ofs << r.frame << "," << r.name << ","
            << r.pre_meta_ms << "," << r.pre_lb_ms << "," << r.pre_pack_ms << "," << r.pre_ms << ","
            << r.h2d_ms << "," << r.a_ms << "," << r.wait_ms << "," << r.b_ms << "," << r.d2h_ms << ","
            << r.post_ms << "," << r.e2e_ms << "," << r.e2e_cap_ms << "," << r.e2e_proc_ms << "\n";
    }
    std::cerr << "Timing CSV written: " << path << " (rows=" << t.size() << ")\n";
}

static void printTimingStats(const std::vector<FrameTiming>& ts) {
    auto dump = [&](const char* key, auto getter){
        std::vector<float> v;
        v.reserve(ts.size());
        for (auto& t: ts) v.push_back(getter(t));
        float mean = 0.f;
        for (auto x: v) mean += x;
        mean /= std::max<size_t>(1, v.size());
        std::cout << "[TIMING] " << key
                  << " mean=" << mean
                  << " p50=" << percentile(v,50)
                  << " p90=" << percentile(v,90)
                  << " p99=" << percentile(v,99)
                  << " max=" << (*std::max_element(v.begin(), v.end()))
                  << " (ms)\n";
    };

    dump("pre_meta", [](const FrameTiming& t){return t.pre_meta_ms;});
    dump("pre_lb",   [](const FrameTiming& t){return t.pre_lb_ms;});
    dump("pre_pack", [](const FrameTiming& t){return t.pre_pack_ms;});
    dump("pre_total",[](const FrameTiming& t){return t.pre_ms;});
    dump("h2d",      [](const FrameTiming& t){return t.h2d_ms;});
    dump("A/infer",  [](const FrameTiming& t){return t.a_ms;});
    dump("wait",     [](const FrameTiming& t){return t.cpu_queue_wait_ms;});
    dump("B",        [](const FrameTiming& t){return t.b_ms;});
    dump("d2h",      [](const FrameTiming& t){return t.d2h_ms;});
    dump("post",     [](const FrameTiming& t){return t.post_ms;});
    dump("e2e",      [](const FrameTiming& t){return t.e2e_ms;});
    dump("e2e_cap",  [](const FrameTiming& t){return t.e2e_cap_ms;});
    dump("e2e_proc", [](const FrameTiming& t){return t.e2e_proc_ms;});
}
struct SlotState {
    int frame_id = -1;
    LetterboxParams lb{};
    // 필요시 원본 이미지(그리기용)를 유지하려면:
    // cv::Mat orig_bgr;  // clone()로 보관 (메모리 큼)
    std::string img_path; // 파일 기반이면 path만 저장해도 됨
};


int main(int argc, char** argv) {
    try {
            std::signal(SIGSEGV, segv_handler);
            std::signal(SIGABRT, segv_handler);

        CliArgs cli = parseArgs(argc, argv);

        TegraStatsLogger pwr;

        JtopLogger util;
        if (!cli.utilCsv.empty()) {
            if (!util.start(cli.utilCsv, cli.utilIntervalMs, cli.utilPy)) {
                std::cerr << "[WARN] failed to start jtop logger; util logging disabled\n";
            } else {
                std::cerr << "[UTIL] jtop logger started: out=" << cli.utilCsv
                          << " interval_ms=" << cli.utilIntervalMs << "\n";
            }
        }

        if (!cli.powerCsv.empty()) {
            if (!pwr.start(cli.powerKey, cli.powerIntervalMs)) {
                std::cerr << "[WARN] failed to start tegrastats; power logging disabled\n";
            } else {
                std::cerr << "[POWER] tegrastats started: key=" << cli.powerKey
                          << " interval_ms=" << cli.powerIntervalMs << "\n";
            }
        }
        

        std::vector<CocoDet> allDetections; allDetections.reserve(200000);
        const bool doTiming = !cli.timingCsv.empty();

        // Collect image paths (single image / list / dir)
        fs::path inPath(cli.inputPath);
        std::vector<std::string> image_paths;
        auto toLower = [](std::string s){
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){return (char)std::tolower(c);} );
            return s;
        };

        if (fs::is_regular_file(inPath)) {
            auto ext = toLower(inPath.extension().string());
            if (ext==".jpg"||ext==".jpeg"||ext==".png"||ext==".bmp") {
                image_paths.push_back(inPath.string());
            } else {
                std::ifstream ifs(inPath.string());
                if (!ifs) { std::cerr << "Failed to open list: " << inPath << "\n"; return -1; }
                std::string line;
                while (std::getline(ifs, line)) {
                    if (line.empty()) continue;
                    image_paths.push_back(line);
                }
            }
        } else if (fs::is_directory(inPath)) {
            for (auto& e : fs::directory_iterator(inPath)) {
                if (!e.is_regular_file()) continue;
                auto ext = toLower(e.path().extension().string());
                if (ext==".jpg"||ext==".jpeg"||ext==".png"||ext==".bmp")
                    image_paths.push_back(e.path().string());
            }
            std::sort(image_paths.begin(), image_paths.end());
        } else {
            std::cerr << "Input path not found: " << inPath << "\n";
            return -1;
        }

        auto image_cache = preload_images(image_paths);
        if (image_cache.empty()) {
            std::cerr << "No images loaded. Abort.\n";
            return -1;
        }

        // Output folders (optional)
        std::string predDir = "preds";
        std::string visDir  = "vis";
        fs::create_directories(predDir);
        fs::create_directories(visDir);

        Logger logger;

        // Streaming buffers
        constexpr int NBUF = 3; // triple buffer is much stabler than double buffer

        // Common: per-frame timing
        std::vector<FrameTiming> timings;
        timings.reserve(image_cache.size());
        

        // --------- Build engines & allocate buffers ---------
        const char* A_IN = "images";
        const char* B_OUT = "output0";
        
        std::vector<std::string> _feats_storage;

        if (!cli.connectTensorNames.empty()) {
            // CLI(--connect)로 들어온 값이 있으면 그걸 쉼표로 잘라서 넣음
            std::stringstream ss(cli.connectTensorNames);
            std::string segment;
            while (std::getline(ss, segment, ',')) {
                _feats_storage.push_back(segment);
            }
        } else {
            // CLI 입력이 없으면 기존 하드코딩 값을 기본값(Default)으로 사용
            _feats_storage = {
                "/model.4/cv2/act/Mul_output_0",
                "/model.6/cv2/act/Mul_output_0",
                "/model.10/cv1/act/Mul_output_0"
            };
        }
        std::vector<const char*> FEATS;
        for (const auto& s : _feats_storage) {
            FEATS.push_back(s.c_str());
        }

        const int FEATS_N = (int)FEATS.size();
        if (cli.mode != CliArgs::Mode::Gpu) {
            if (FEATS_N <= 0) {
                throw std::runtime_error("[HYBRID] --connect is empty (no feature tensors).");
            }
        }
        SlotState slot_state[NBUF];

        // Pinned host IO per slot
        void* hInPinned[NBUF] = {nullptr, nullptr, nullptr};
        void* hOutPinned[NBUF] = {nullptr, nullptr, nullptr};
        size_t bytesInput = 0;
        size_t bytesOutput = 0;

        // Per-slot metadata needed at consume time
        LetterboxParams lb_slot[NBUF];
        std::chrono::steady_clock::time_point t0_cpu[NBUF];
        std::chrono::steady_clock::time_point tproc_cpu[NBUF];
        uint64_t cpu_cap_ns[NBUF]    = {0,0,0};
        uint64_t cpu_pre_ns[NBUF]    = {0,0,0};
        uint64_t cpu_submit_ns[NBUF] = {0,0,0};
        uint64_t cpu_done_ns[NBUF]   = {0,0,0};

        auto now_ns = [&]()->uint64_t {
            auto now = std::chrono::steady_clock::now().time_since_epoch();
            return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
        };

        std::string name_slot[NBUF];
        int frame_slot[NBUF] = {-1,-1,-1};
        float pre_meta_slot[NBUF] = {0,0,0};
        float pre_lb_slot[NBUF] = {0,0,0};
        float pre_pack_slot[NBUF] = {0,0,0};
        float pre_total_slot[NBUF] = {0,0,0};

        // Per-slot CPU-side auxiliary metrics for frame-centric accounting
        float cpu_sleep_slot[NBUF]      = {0,0,0};
        float cpu_queue_wait_slot[NBUF] = {0,0,0};
        float cpu_submit_slot[NBUF]     = {0,0,0};
        float cpu_block_slot[NBUF]      = {0,0,0};
        float cpu_write_slot[NBUF]      = {0,0,0};
        std::chrono::steady_clock::time_point tsubmit_cpu[NBUF];


        // GPU-only resources
        std::unique_ptr<TrtEngine> G;
        void* dGin[NBUF] = {nullptr,nullptr,nullptr};
        void* dGout[NBUF] = {nullptr,nullptr,nullptr};
        cudaStream_t streamG = nullptr;
        cudaEvent_t evOutG[NBUF] = {nullptr};
        cudaEvent_t tg_h2d_s[NBUF] = {nullptr}, tg_h2d_e[NBUF] = {nullptr};
        cudaEvent_t tg_inf_s[NBUF] = {nullptr}, tg_inf_e[NBUF] = {nullptr};
        cudaEvent_t tg_d2h_s[NBUF] = {nullptr}, tg_d2h_e[NBUF] = {nullptr};

        // Hybrid resources
        std::unique_ptr<TrtEngine> A, B;
        void* dAin[NBUF] = {nullptr,nullptr,nullptr};
        void* dBout[NBUF] = {nullptr,nullptr,nullptr};
        std::vector<std::array<void*, NBUF>> dFeat;   // [k][slot]
        std::vector<size_t> featBytes;
        cudaStream_t streamDLA = nullptr;
        cudaStream_t streamGPU = nullptr;
        cudaStream_t streamH2D = nullptr;
        cudaEvent_t evH2D[NBUF]  = {nullptr};
        cudaEvent_t evA[NBUF]    = {nullptr};
        cudaEvent_t evOut[NBUF]  = {nullptr};

        cudaEvent_t t_h2d_s[NBUF] = {nullptr}, t_h2d_e[NBUF] = {nullptr};
        cudaEvent_t t_a_s[NBUF]   = {nullptr}, t_a_e[NBUF]   = {nullptr};
        cudaEvent_t t_wait_s[NBUF]= {nullptr}, t_wait_e[NBUF]= {nullptr};
        cudaEvent_t t_b_s[NBUF]   = {nullptr}, t_b_e[NBUF]   = {nullptr};
        cudaEvent_t t_d2h_s[NBUF] = {nullptr}, t_d2h_e[NBUF] = {nullptr};

        // TensorRT dtypes (used for debug sampling)
        nvinfer1::DataType dtInput  = nvinfer1::DataType::kFLOAT;
        nvinfer1::DataType dtOutput = nvinfer1::DataType::kFLOAT;
        std::vector<nvinfer1::DataType> dtFeat;       // [k]

        // Debug sample stats for input per slot
        SampleStats inStats[NBUF];
        bool inStatsValid[NBUF] = {false,false,false};

        // Optional pinned host buffers for feature sampling
        std::vector<void*>  hFeatSample;
        std::vector<size_t> hFeatSampleBytes;

        if (cli.mode == CliArgs::Mode::Gpu) {
            G = std::make_unique<TrtEngine>(loadEngine(cli.planGPU, logger));
            std::cout << "[GPU] tensors:\n";
            printTensorInfo(G->engine.get(), A_IN);
            printTensorInfo(G->engine.get(), B_OUT);
            bytesInput  = tensorBytes(G->engine.get(), A_IN);
            bytesOutput = tensorBytes(G->engine.get(), B_OUT);

            dtInput  = G->engine->getTensorDataType(A_IN);
            dtOutput = G->engine->getTensorDataType(B_OUT);

            if (cli.debug_ptrs) {
                std::cerr << "[DBG_PTRS] dtInput=" << (int)dtInput << " dtOutput=" << (int)dtOutput << "\n";
            }

            // pinned host in/out per slot
            for (int i=0;i<NBUF;i++) {
                CHECK_CUDA(cudaHostAlloc((void**)&hInPinned[i],  bytesInput,  cudaHostAllocPortable));
                CHECK_CUDA(cudaHostAlloc((void**)&hOutPinned[i], bytesOutput, cudaHostAllocPortable));
                CHECK_CUDA(cudaMalloc(&dGin[i],  bytesInput));
                CHECK_CUDA(cudaMalloc(&dGout[i], bytesOutput));
            }
            CHECK_CUDA(cudaStreamCreateWithFlags(&streamG, cudaStreamNonBlocking));
            for (int i=0;i<NBUF;i++) {
                CHECK_CUDA(cudaEventCreateWithFlags(&evOutG[i], cudaEventDisableTiming));
                if (doTiming) {
                    CHECK_CUDA(cudaEventCreate(&tg_h2d_s[i])); CHECK_CUDA(cudaEventCreate(&tg_h2d_e[i]));
                    CHECK_CUDA(cudaEventCreate(&tg_inf_s[i])); CHECK_CUDA(cudaEventCreate(&tg_inf_e[i]));
                    CHECK_CUDA(cudaEventCreate(&tg_d2h_s[i])); CHECK_CUDA(cudaEventCreate(&tg_d2h_e[i]));
                }
            }
        } else {
            A = std::make_unique<TrtEngine>(loadEngine(cli.planA, logger));
            B = std::make_unique<TrtEngine>(loadEngine(cli.planB, logger));

            auto dumpIO = [&](nvinfer1::ICudaEngine* e, const char* tag){
            int nb = e->getNbIOTensors();
            std::cout << "[" << tag << " I/O] nb=" << nb << "\n";
            for(int i=0;i<nb;i++){
                const char* name = e->getIOTensorName(i);
                auto mode = e->getTensorIOMode(name);
                auto dt   = e->getTensorDataType(name);
                auto dims = e->getTensorShape(name);
                std::cout << "  " << i << " name=" << name
                        << " mode=" << (mode==nvinfer1::TensorIOMode::kINPUT?"IN":"OUT")
                        << " dtype=" << (int)dt
                        << " dims=" << (int)dims.d[0] << "," << (int)dims.d[1] << "," << (int)dims.d[2] << "," << (int)dims.d[3] << "\n";
            }
            };

            dumpIO(A->engine.get(), "A");
            dumpIO(B->engine.get(), "B");

            std::cout << "[A] tensors:\n";
            printTensorInfo(A->engine.get(), A_IN);
            for (auto n : FEATS) printTensorInfo(A->engine.get(), n);
            std::cout << "[B] tensors:\n";
            for (auto n : FEATS) printTensorInfo(B->engine.get(), n);
            printTensorInfo(B->engine.get(), B_OUT);

            dtInput  = A->engine->getTensorDataType(A_IN);
            dtOutput = B->engine->getTensorDataType(B_OUT);
            dFeat.assign(FEATS_N, {});
            for (auto& a : dFeat) a.fill(nullptr);

            featBytes.assign(FEATS_N, 0);
            dtFeat.assign(FEATS_N, nvinfer1::DataType::kFLOAT);

            hFeatSample.assign(FEATS_N, nullptr);
            hFeatSampleBytes.assign(FEATS_N, 0);

            // dtype/bytes 채우기
            for (int k = 0; k < FEATS_N; ++k) {
                dtFeat[k] = A->engine->getTensorDataType(FEATS[k]);
                featBytes[k] = tensorBytes(A->engine.get(), FEATS[k]);
            }

            if (dtInput != nvinfer1::DataType::kFLOAT) {
                std::cerr << "[WARN] Engine input dtype is not FP32 (dtInput=" << (int)dtInput << "). "
                        << "This program packs FP32 input; results may be invalid unless you rebuild engine or add FP16 packing." << "\n";
            }

            bytesInput  = tensorBytes(A->engine.get(), A_IN);
            bytesOutput = tensorBytes(B->engine.get(), B_OUT);

            for (int i = 0; i < NBUF; i++) {
                CHECK_CUDA(cudaHostAlloc((void**)&hInPinned[i],  bytesInput,  cudaHostAllocPortable));
                CHECK_CUDA(cudaHostAlloc((void**)&hOutPinned[i], bytesOutput, cudaHostAllocPortable));
                CHECK_CUDA(cudaMalloc(&dAin[i],  bytesInput));
                CHECK_CUDA(cudaMalloc(&dBout[i], bytesOutput));

                for (int k = 0; k < FEATS_N; ++k) {
                    CHECK_CUDA(cudaMalloc(&dFeat[k][i], featBytes[k]));
                }
            }
        }
        CHECK_CUDA(cudaStreamCreateWithFlags(&streamH2D, cudaStreamNonBlocking));
        CHECK_CUDA(cudaStreamCreateWithFlags(&streamDLA, cudaStreamNonBlocking));
        CHECK_CUDA(cudaStreamCreateWithFlags(&streamGPU, cudaStreamNonBlocking));
        for (int i=0;i<NBUF;i++) {
            CHECK_CUDA(cudaEventCreateWithFlags(&evH2D[i], cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&evA[i], cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&evOut[i], cudaEventDisableTiming));
            if (doTiming) {
                CHECK_CUDA(cudaEventCreate(&t_h2d_s[i])); CHECK_CUDA(cudaEventCreate(&t_h2d_e[i]));
                CHECK_CUDA(cudaEventCreate(&t_a_s[i]));   CHECK_CUDA(cudaEventCreate(&t_a_e[i]));
                CHECK_CUDA(cudaEventCreate(&t_wait_s[i]));CHECK_CUDA(cudaEventCreate(&t_wait_e[i]));
                CHECK_CUDA(cudaEventCreate(&t_b_s[i]));   CHECK_CUDA(cudaEventCreate(&t_b_e[i]));
                CHECK_CUDA(cudaEventCreate(&t_d2h_s[i])); CHECK_CUDA(cudaEventCreate(&t_d2h_e[i]));
            }
        }
        // --------- Helper bindings ---------
        auto bindA = [&](int bi){
            if (!A->ctx->setTensorAddress(A_IN, dAin[bi])){
                throw std::runtime_error("bind A_IN failed");
            }
            for (int k=0;k<FEATS_N;k++) {
                if (!A->ctx->setTensorAddress(FEATS[k], dFeat[k][bi])){ 
                    throw std::runtime_error(std::string("bind A_OUT failed: ") + FEATS[k]);
                }
            }
        };
        auto bindB = [&](int bi){
            for (int k=0;k<FEATS_N;k++) {
                if (!B->ctx->setTensorAddress(FEATS[k], dFeat[k][bi]))
                    throw std::runtime_error(std::string("bind B_IN failed: ") + FEATS[k]);
            }
            if (!B->ctx->setTensorAddress(B_OUT, dBout[bi]))
                throw std::runtime_error("bind B_OUT failed");
        };
        // --------- Streaming loop ---------
        const int num_frames = (int)image_cache.size();

        using Clock = std::chrono::steady_clock;

        const bool throttle = (cli.fps > 0.0) && (cli.fps_cap_mode != CliArgs::FpsCapMode::Off);

        Clock::duration period{};
        if (throttle) {
            period = std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(1.0 / cli.fps)
            );
        }

        // Input-cap (camera-like pacing) uses a single global deadline (NOT per-slot).
        bool input_deadline_inited = false;
        Clock::time_point next_deadline{};

        auto throttle_sleep_input = [&]()->float {
            if (!throttle || cli.fps_cap_mode != CliArgs::FpsCapMode::Input) return 0.f;
            if (!input_deadline_inited) {
                next_deadline = Clock::now() + period;
                input_deadline_inited = true;
                return 0.f;
            }
            auto t0 = Clock::now();
            std::this_thread::sleep_until(next_deadline);
            auto t1 = Clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

            // advance schedule
            next_deadline += period;

            // if we fell behind, re-sync to now+period
            auto now = Clock::now();
            if (next_deadline < now) next_deadline = now + period;
            return ms;
        };
        
        auto preprocess_into_slot = [&](int f, int slot, Clock::time_point t_capture) {
            auto& st = slot_state[slot];
            g_frame = f; g_slot = slot; g_stage = 100;

            st.frame_id = f;
            st.img_path = image_cache[f].path;

            const cv::Mat& orig = image_cache[f].bgr;
            if (orig.empty()) {
                st.frame_id = -1;
                st.lb = {};
                return;
            }

            // Service/capture reference timestamp (used for e2e_cap_ms)
            t0_cpu[slot] = t_capture;

            cpu_cap_ns[slot] = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t_capture.time_since_epoch()).count();
            // Processing start timestamp (used for e2e_proc_ms)
            tproc_cpu[slot] = Clock::now();

            cpu_pre_ns[slot] = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(tproc_cpu[slot].time_since_epoch()).count();
            auto t_pre0 = Clock::now();
            cv::Mat in640 = letterbox_640(orig, st.lb);
            g_stage = 110;
            auto t_lb = Clock::now();

            if (dtInput == nvinfer1::DataType::kFLOAT) {
                fill_input_fp32_nchw_rgb01_direct(in640, (float*)hInPinned[slot]);
            } else if (dtInput == nvinfer1::DataType::kHALF) {
                fill_input_fp16_nchw_rgb01_direct(in640, (__half*)hInPinned[slot]); // 새로 구현
            } else {
                throw std::runtime_error("unsupported input dtype");
            }
            g_stage = 120;
            auto t_pack = Clock::now();

            pre_lb_slot[slot]   = std::chrono::duration<float, std::milli>(t_lb   - t_pre0).count();
            pre_pack_slot[slot] = std::chrono::duration<float, std::milli>(t_pack - t_lb).count();
            pre_total_slot[slot]= std::chrono::duration<float, std::milli>(t_pack - t_pre0).count();

            if (cli.debug_io) {
                inStats[slot] = sample_stats_host(hInPinned[slot], dtInput, bytesInput, 8400);
                inStatsValid[slot] = true;
            }
        };

        // Measure host-side overhead of "submit" calls (enqueue/memcpy/event record). This is not GPU time.
        auto add_cpu_submit = [&](int slot, auto&& fn) {
            auto _t0 = Clock::now();
            fn();
            auto _t1 = Clock::now();
            cpu_submit_slot[slot] += std::chrono::duration<float, std::milli>(_t1 - _t0).count();
        };

        auto consume_slot = [&](int slot, bool isHybrid = true) {
            // output is ready (event already synchronized by caller)
            auto t_inf_done = Clock::now();
            g_slot = slot; g_stage = 200;

            const auto& st = slot_state[slot];
            int fid = st.frame_id;

            // Guard
            if (fid < 0 || fid >= (int)image_cache.size() || hOutPinned[slot] == nullptr) {
                cpu_sleep_slot[slot] = cpu_queue_wait_slot[slot] = cpu_submit_slot[slot] = 0.f;
                cpu_block_slot[slot] = cpu_write_slot[slot] = 0.f;
                return;
            }
            const auto& lb = st.lb;

            // post/decode
            g_stage = 210;

            std::vector<float> out_fp32; // 필요 시 변환 버퍼
            const float* out_ptr = nullptr;

            if (dtOutput == nvinfer1::DataType::kFLOAT) {
                out_ptr = (const float*)hOutPinned[slot];
            } else if (dtOutput == nvinfer1::DataType::kHALF) {
                // bytesOutput / 2 만큼 half -> float 변환
                size_t n = bytesOutput / sizeof(__half);
                out_fp32.resize(n);
                const __half* hp = (const __half*)hOutPinned[slot];
                for (size_t i=0;i<n;i++) out_fp32[i] = __half2float(hp[i]);
                out_ptr = out_fp32.data();
            } else {
                throw std::runtime_error("unsupported output dtype");
            }
            auto dets = decode_yolo84_8400(out_ptr, lb, cli.confThres , cli.iouThres, cli.maxDet, false);
            auto t_post_done = Clock::now();

            // Collect detections as COCO-format JSON records (for COCOeval), if enabled.
            // We assume Detection coordinates are in original image pixel space.
            if (!cli.predJson.empty()) {
                int image_id = coco_image_id_from_filename(st.img_path);
                for (const auto& det : dets) {
                    int cat_id = coco_category_id_from_class(det.cls);
                    if (image_id < 0 || cat_id < 0) continue;
                    float x = det.x1;
                    float y = det.y1;
                    float w = det.x2 - det.x1;
                    float h = det.y2 - det.y1;
                    if (w <= 0.f || h <= 0.f) continue;
                    CocoDet cd;
                    cd.image_id = image_id;
                    cd.category_id = cat_id;
                    cd.x = x; cd.y = y; cd.w = w; cd.h = h;
                    cd.score = det.conf;
                    allDetections.push_back(cd);
                }
            }

            if (!measure_only) {
                const auto& item = image_cache[(size_t)fid];
                cv::Mat vis = item.bgr.clone();
                for (auto& d: dets) {
                    cv::rectangle(vis, {int(d.x1), int(d.y1)}, {int(d.x2), int(d.y2)}, {0,255,0}, 2);
                }
                fs::path p(item.path);
                std::string outImg = (fs::path(visDir) / p.filename()).string();
                std::string outTxt = (fs::path(predDir) / p.stem()).string() + ".txt";
                auto _w0 = Clock::now();
                cv::imwrite(outImg, vis);
                save_preds_yolo(outTxt, dets, lb.orig_w, lb.orig_h);
                auto _w1 = Clock::now();
                cpu_write_slot[slot] += std::chrono::duration<float, std::milli>(_w1 - _w0).count();
            }

            auto t_end = Clock::now();

            cpu_done_ns[slot] = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t_end.time_since_epoch()).count();
            // ---------- Timing record (per completed frame) ----------
            FrameTiming ft;
            ft.frame = fid;
            ft.slot  = slot;
            ft.name  = st.img_path.empty() ? std::to_string(fid) : fs::path(st.img_path).filename().string();

            // CPU stage times
            ft.cpu_sleep_ms      = cpu_sleep_slot[slot];
            ft.cpu_queue_wait_ms = cpu_queue_wait_slot[slot];
            ft.cpu_submit_ms     = cpu_submit_slot[slot];
            ft.cpu_block_ms      = cpu_block_slot[slot];
            ft.cpu_write_ms      = cpu_write_slot[slot];

            // preprocess breakdown
            ft.pre_lb_ms   = pre_lb_slot[slot];
            ft.pre_pack_ms = pre_pack_slot[slot];
            ft.pre_ms      = pre_total_slot[slot];

            // post and e2e
            ft.post_ms      = std::chrono::duration<float, std::milli>(t_end - t_inf_done).count();
            ft.e2e_cap_ms   = std::chrono::duration<float, std::milli>(t_inf_done - t0_cpu[slot]).count();
            ft.e2e_proc_ms  = std::chrono::duration<float, std::milli>(t_inf_done - tproc_cpu[slot]).count();
            ft.e2e_ms       = ft.e2e_cap_ms;


            // CPU timestamps (steady_clock ns) for power/latency join
            ft.cpu_t_cap_ns    = cpu_cap_ns[slot];
            ft.cpu_t_pre_ns    = cpu_pre_ns[slot];
            ft.cpu_t_submit_ns = cpu_submit_ns[slot];
            ft.cpu_t_done_ns   = cpu_done_ns[slot];
            // device timings (optional)
            if (doTiming) {
                float ms = 0.f;
                if (cli.mode == CliArgs::Mode::Gpu) {
                    CHECK_CUDA(cudaEventElapsedTime(&ms, tg_h2d_s[slot], tg_h2d_e[slot])); ft.h2d_ms = ms;
                    CHECK_CUDA(cudaEventElapsedTime(&ms, tg_inf_s[slot], tg_inf_e[slot])); ft.b_ms   = ms;
                    CHECK_CUDA(cudaEventElapsedTime(&ms, tg_d2h_s[slot], tg_d2h_e[slot])); ft.d2h_ms = ms;
                    ft.a_ms = 0.f;
                    ft.wait_ms = 0.f;
                } else if (cli.mode == CliArgs::Mode::Hybrid && isHybrid) {
                    CHECK_CUDA(cudaEventElapsedTime(&ms, t_h2d_s[slot],  t_h2d_e[slot]));  ft.h2d_ms  = ms;
                    CHECK_CUDA(cudaEventElapsedTime(&ms, t_a_s[slot],    t_a_e[slot]));    ft.a_ms    = ms;
                    CHECK_CUDA(cudaEventElapsedTime(&ms, t_wait_s[slot], t_wait_e[slot])); ft.wait_ms = ms;
                    CHECK_CUDA(cudaEventElapsedTime(&ms, t_b_s[slot],    t_b_e[slot]));    ft.b_ms    = ms;
                    CHECK_CUDA(cudaEventElapsedTime(&ms, t_d2h_s[slot],  t_d2h_e[slot]));  ft.d2h_ms  = ms;
                }
            }

            timings.push_back(std::move(ft));

            // reset per-slot CPU accounting (so next frame is clean)
            cpu_sleep_slot[slot] = cpu_queue_wait_slot[slot] = cpu_submit_slot[slot] = 0.f;
            cpu_block_slot[slot] = cpu_write_slot[slot] = 0.f;
        };
        // Optional input pacing: emulate a fixed-FPS source (e.g., camera).
        // Note: this is *pacing*, not a separate producer queue. If processing is slower than the target FPS,
        // the schedule will fall behind and the loop will effectively run as fast as it can (no artificial sleep).
        const auto t_stream0 = Clock::now();
        const std::chrono::duration<double> frame_period_sec(throttle ? (1.0 / cli.fps) : 0.0);
        auto capture_time_for = [&](int frame_idx) -> Clock::time_point {
            if (!throttle) return Clock::now();
            auto dt = std::chrono::duration_cast<Clock::duration>(frame_period_sec * (double)frame_idx);
            return t_stream0 + dt;
        };
        if (cli.mode == CliArgs::Mode::Hybrid) {
            // Prime events to avoid accidental reuse hazards
            for (int i=0;i<NBUF;i++) {
                // Mark evH2D/evA/evOut as "not ready" by recording on default stream once
                CHECK_CUDA(cudaEventRecord(evH2D[i], 0));
                CHECK_CUDA(cudaEventRecord(evA[i], 0));
                CHECK_CUDA(cudaEventRecord(evOut[i], 0));
            }

            if (cli.sched == CliArgs::Sched::Throughput) {
                // Pipelined A(f) on DLA + B(f-1) on GPU. Optimizes throughput; adds >= 1-frame pipeline latency.
                size_t done = 0;
                bool started = false;
                Clock::time_point t_first{}, t_last{};
                Clock::time_point t_stream0_done{};

                for (int f = 0; f < num_frames + 2; ++f) {
                    int slot = f % NBUF;
                    int prev = (f - 1 + NBUF) % NBUF;
                    int out  = (f - 2 + NBUF) % NBUF;
                    
                    // (consume) output for frame f-2
                    if (f >= 2) {
                        auto _blk0 = Clock::now();
                        CHECK_CUDA(cudaEventSynchronize(evOut[out]));
                        auto _blk1 = Clock::now();
                        cpu_block_slot[out] += std::chrono::duration<float, std::milli>(_blk1 - _blk0).count();

                        if (!started) {
                            t_first = Clock::now();
                            t_stream0_done = t_first;
                            started = true;
                        }

                        // Count this frame as completed (its output is ready) and optionally throttle by completion cadence.
                        done++;

                        // DONE-cap: throttle by completion cadence (consume/done-based).
                        // IMPORTANT: do this *before* consume_slot() so the sleep time is attributed to the same frame.
                        if (throttle && cli.fps_cap_mode == CliArgs::FpsCapMode::Done) {
                            auto target = t_stream0_done + period * done;
                            auto t0 = Clock::now();
                            std::this_thread::sleep_until(target);
                            auto t1 = Clock::now();
                            cpu_sleep_slot[out] += std::chrono::duration<float, std::milli>(t1 - t0).count();
                        }

                        consume_slot(out, /*isHybrid=*/true);
                    }

                    // (submit) B+D2H for previous frame (f-1)
                    if (f >= 1 && (f - 1) < num_frames) {
                        cpu_submit_ns[prev] = now_ns();
                        if (!cli.idle_pipeline) {
                            add_cpu_submit(prev, [&](){
                                if (doTiming) CHECK_CUDA(cudaEventRecord(t_wait_s[prev], streamGPU));
                                CHECK_CUDA(cudaStreamWaitEvent(streamGPU, evA[prev], 0));
                                if (doTiming) CHECK_CUDA(cudaEventRecord(t_wait_e[prev], streamGPU));

                                bindB(prev);
                                if (doTiming) CHECK_CUDA(cudaEventRecord(t_b_s[prev], streamGPU));
                                if (!B->ctx->enqueueV3(streamGPU)) { throw std::runtime_error("enqueue B failed"); }
                                if (doTiming) CHECK_CUDA(cudaEventRecord(t_b_e[prev], streamGPU));

                                if (doTiming) CHECK_CUDA(cudaEventRecord(t_d2h_s[prev], streamGPU));
                                CHECK_CUDA(cudaMemcpyAsync(hOutPinned[prev], dBout[prev], bytesOutput,
                                                        cudaMemcpyDeviceToHost, streamGPU));
                                if (doTiming) CHECK_CUDA(cudaEventRecord(t_d2h_e[prev], streamGPU));

                                CHECK_CUDA(cudaEventRecord(evOut[prev], streamGPU));
                            });
                        }else{
                            add_cpu_submit(prev, [&](){
                                std::memset(hOutPinned[prev], 0, bytesOutput);
                                CHECK_CUDA(cudaEventRecord(evOut[prev], streamGPU));
                            });
                        }

                    }
                        
                    // (submit) preprocess + H2D + A for frame f
                    if (f < num_frames) {

                        // Ensure the ring slot is free before we overwrite its buffers.
                        // With NBUF=3 and this schedule it should already be free, but keep this for robustness.
                        {
                            auto _qw0 = Clock::now();
                            auto _q = cudaEventQuery(evOut[slot]);
                            if (_q == cudaErrorNotReady) {
                                auto _blk0 = Clock::now();
                                CHECK_CUDA(cudaEventSynchronize(evOut[slot]));
                                auto _blk1 = Clock::now();
                                cpu_block_slot[slot] += std::chrono::duration<float, std::milli>(_blk1 - _blk0).count();
                            } else if (_q != cudaSuccess) {
                                CHECK_CUDA(_q);
                            }
                            auto _qw1 = Clock::now();
                            cpu_queue_wait_slot[slot] = std::chrono::duration<float, std::milli>(_qw1 - _qw0).count();
                        }

                        cpu_sleep_slot[slot] = throttle_sleep_input();
                        auto t_cap = Clock::now();
                        preprocess_into_slot(f, slot, t_cap);

                        // record first-submit timestamp for this frame (power join window start)

                        if (!cli.idle_pipeline) {
                            add_cpu_submit(slot, [&](){
                                // H2D on a separate stream to allow overlap with DLA/GPU work
                                if (doTiming) CHECK_CUDA(cudaEventRecord(t_h2d_s[slot], streamH2D));
                                CHECK_CUDA(cudaMemcpyAsync(dAin[slot], hInPinned[slot], bytesInput,
                                                        cudaMemcpyHostToDevice, streamH2D));
                                if (doTiming) CHECK_CUDA(cudaEventRecord(t_h2d_e[slot], streamH2D));
                                CHECK_CUDA(cudaEventRecord(evH2D[slot], streamH2D));

                                // A waits for H2D completion
                                CHECK_CUDA(cudaStreamWaitEvent(streamDLA, evH2D[slot], 0));

                                bindA(slot);
                                if (doTiming) CHECK_CUDA(cudaEventRecord(t_a_s[slot], streamDLA));
                                if (!A->ctx->enqueueV3(streamDLA)) throw std::runtime_error("enqueue A failed");
                                if (doTiming) CHECK_CUDA(cudaEventRecord(t_a_e[slot], streamDLA));
                                CHECK_CUDA(cudaEventRecord(evA[slot], streamDLA));
                            });
                        }else{
                            add_cpu_submit(slot, [&](){
                                CHECK_CUDA(cudaEventRecord(evA[slot], streamDLA));
                            });
                        }
                    }
                    if (started) t_last = Clock::now();
                }

                if (started) {
                    double sec = std::chrono::duration<double>(t_last - t_first).count();
                    double fps_meas = (sec > 0.0) ? (double(done) / sec) : 0.0;
                    std::cout << "[BENCH] mode=hybrid sched=throughput frames=" << done
                              << " time_s=" << sec << " FPS=" << fps_meas
                              << " cap_mode=" << to_string(cli.fps_cap_mode)
                              << " cap_fps=" << cli.fps << "\n";
                }
            }else {
                // Latency schedule: for each frame, run A(f) then immediately B(f), and wait for output.
                // This removes pipeline latency but will typically reduce throughput.
                size_t done_lat = 0;
                bool started_lat = false;
                Clock::time_point t_first_lat{}, t_last_lat{};

                for (int f = 0; f < num_frames; ++f) {
                    int slot = f % NBUF;

                    // Ensure slot is not reused while prior GPU work is still in-flight.
                    // Attribute this to queue-wait (slot availability), not to "output wait" for the current frame.
                    {
                        auto _qw0 = Clock::now();
                        auto _q = cudaEventQuery(evOut[slot]);
                        if (_q == cudaErrorNotReady) {
                            auto _blk0 = Clock::now();
                            CHECK_CUDA(cudaEventSynchronize(evOut[slot]));
                            auto _blk1 = Clock::now();
                            cpu_block_slot[slot] += std::chrono::duration<float, std::milli>(_blk1 - _blk0).count();
                        } else if (_q != cudaSuccess) {
                            CHECK_CUDA(_q);
                        }
                        auto _qw1 = Clock::now();
                        cpu_queue_wait_slot[slot] = std::chrono::duration<float, std::milli>(_qw1 - _qw0).count();
                    }
                    // pacing sleep (service-side delay)
                    cpu_sleep_slot[slot] = throttle_sleep_input();
                    auto t_cap = Clock::now(); 
                    preprocess_into_slot(f, slot, t_cap);

                    // record first-submit timestamp for this frame (power join window start)
                    cpu_submit_ns[slot] = now_ns();
                    if (!cli.idle_pipeline) {
                        add_cpu_submit(slot, [&](){
                        // H2D
                        if (doTiming) CHECK_CUDA(cudaEventRecord(t_h2d_s[slot], streamH2D));
                        CHECK_CUDA(cudaMemcpyAsync(dAin[slot], hInPinned[slot], bytesInput,
                                                cudaMemcpyHostToDevice, streamH2D));
                        if (doTiming) CHECK_CUDA(cudaEventRecord(t_h2d_e[slot], streamH2D));
                        CHECK_CUDA(cudaEventRecord(evH2D[slot], streamH2D));

                        // A waits for H2D completion
                        CHECK_CUDA(cudaStreamWaitEvent(streamDLA, evH2D[slot], 0));
                        bindA(slot);
                        if (doTiming) CHECK_CUDA(cudaEventRecord(t_a_s[slot], streamDLA));
                        if (!A->ctx->enqueueV3(streamDLA)) throw std::runtime_error("enqueue A failed");
                        if (doTiming) CHECK_CUDA(cudaEventRecord(t_a_e[slot], streamDLA));
                        CHECK_CUDA(cudaEventRecord(evA[slot], streamDLA));

                        // Wait for A completion on GPU stream (measured as wait_ms)
                        if (doTiming) CHECK_CUDA(cudaEventRecord(t_wait_s[slot], streamGPU));
                        CHECK_CUDA(cudaStreamWaitEvent(streamGPU, evA[slot], 0));
                        if (doTiming) CHECK_CUDA(cudaEventRecord(t_wait_e[slot], streamGPU));

                        // B + D2H
                        bindB(slot);
                        if (doTiming) CHECK_CUDA(cudaEventRecord(t_b_s[slot], streamGPU));
                        if (!B->ctx->enqueueV3(streamGPU)) throw std::runtime_error("enqueue B failed");
                        if (doTiming) CHECK_CUDA(cudaEventRecord(t_b_e[slot], streamGPU));

                        if (doTiming) CHECK_CUDA(cudaEventRecord(t_d2h_s[slot], streamGPU));
                        CHECK_CUDA(cudaMemcpyAsync(hOutPinned[slot], dBout[slot], bytesOutput,
                                                cudaMemcpyDeviceToHost, streamGPU));
                        if (doTiming) CHECK_CUDA(cudaEventRecord(t_d2h_e[slot], streamGPU));

                        CHECK_CUDA(cudaEventRecord(evOut[slot], streamGPU));
                        });
                    }else{
                        add_cpu_submit(slot, [&](){
                            std::memset(hOutPinned[slot], 0, bytesOutput);
                            CHECK_CUDA(cudaEventRecord(evOut[slot], streamGPU));
                        });
                    }

                    // Latency schedule waits for output
                    {
                        auto _blk0 = Clock::now();
                        CHECK_CUDA(cudaEventSynchronize(evOut[slot]));
                        auto _blk1 = Clock::now();
                        cpu_block_slot[slot] += std::chrono::duration<float, std::milli>(_blk1 - _blk0).count();
                    }
                    
                    consume_slot(slot, /*isHybrid=*/true);

                    if (!started_lat) { t_first_lat = Clock::now(); started_lat = true; }
                    t_last_lat = Clock::now();
                    done_lat++;
                }

                if (started_lat) {
                    double sec = std::chrono::duration<double>(t_last_lat - t_first_lat).count();
                    double fps_meas = (sec > 0.0) ? (double(done_lat) / sec) : 0.0;
                    std::cout << "[BENCH] mode=hybrid sched=latency frames=" << done_lat
                              << " time_s=" << sec << " FPS=" << fps_meas << "\n";
                }
            }
        }else {
            // GPU-only streaming
            for (int i=0;i<NBUF;i++) {
                CHECK_CUDA(cudaEventRecord(evOutG[i], 0));
            }
            std::cerr << "[GPU] start loop frames=" << num_frames << " NBUF="<<NBUF<<" doTiming="<< (doTiming?1:0)
                      << " bytesIn="<< bytesInput << " bytesOut="<< bytesOutput << "\n";

            size_t done_g = 0;
            bool started_g = false;
            Clock::time_point t_first_g{}, t_last_g{};

            // Bind once per slot (addresses differ per slot)
            for (int f = 0; f < num_frames + 1; ++f) {
                int slot = f % NBUF;
                int out  = (f - 1 + NBUF) % NBUF;

                if (f >= 1) {
                    {
                        auto _blk0 = Clock::now();
                    CHECK_CUDA(cudaEventSynchronize(evOutG[out]));
                        auto _blk1 = Clock::now();
                        cpu_block_slot[out] += std::chrono::duration<float, std::milli>(_blk1 - _blk0).count();
                    }
                    consume_slot(out, /*isHybrid=*/false);
                    if (!started_g) { t_first_g = Clock::now(); started_g = true; }
                    t_last_g = Clock::now();
                    done_g++;
                }

                if (f < num_frames) {
                    cpu_sleep_slot[slot] = throttle_sleep_input();
                    

                    // Optional: if slot reuse is still busy, wait and attribute to queue-wait
                    {
                        auto _qw0 = Clock::now();

                        // GPU-only MUST use evOutG
                        auto _q = cudaEventQuery(evOutG[slot]);
                        if (_q == cudaErrorNotReady) {
                            auto _blk0 = Clock::now();
                            CHECK_CUDA(cudaEventSynchronize(evOutG[slot]));
                            auto _blk1 = Clock::now();
                            cpu_block_slot[slot] += std::chrono::duration<float, std::milli>(_blk1 - _blk0).count();
                        } else if (_q != cudaSuccess) {
                            CHECK_CUDA(_q);
                        }

                        auto _qw1 = Clock::now();
                        cpu_queue_wait_slot[slot] = std::chrono::duration<float, std::milli>(_qw1 - _qw0).count();
                    }   
                        auto t_cap = Clock::now();
                        preprocess_into_slot(f, slot, t_cap);
                        cpu_submit_ns[slot] = now_ns();
                        if (!cli.idle_pipeline) {
                            add_cpu_submit(slot, [&](){

                                if (!G->ctx->setTensorAddress(A_IN, dGin[slot]))
                                    throw std::runtime_error("bind GPU input failed");
                                if (!G->ctx->setTensorAddress(B_OUT, dGout[slot]))
                                    throw std::runtime_error("bind GPU output failed");

                                if (doTiming) CHECK_CUDA(cudaEventRecord(tg_h2d_s[slot], streamG));
                                CHECK_CUDA(cudaMemcpyAsync(dGin[slot], hInPinned[slot], bytesInput, cudaMemcpyHostToDevice, streamG));
                                if (doTiming) CHECK_CUDA(cudaEventRecord(tg_h2d_e[slot], streamG));

                                if (doTiming) CHECK_CUDA(cudaEventRecord(tg_inf_s[slot], streamG));
                                if (!G->ctx->enqueueV3(streamG)) throw std::runtime_error("enqueue GPU engine failed");
                                if (doTiming) CHECK_CUDA(cudaEventRecord(tg_inf_e[slot], streamG));

                                if (doTiming) CHECK_CUDA(cudaEventRecord(tg_d2h_s[slot], streamG));
                                {
                                    auto _sub0 = Clock::now();
                                CHECK_CUDA(cudaMemcpyAsync(hOutPinned[slot], dGout[slot], bytesOutput, cudaMemcpyDeviceToHost, streamG));
                                    auto _sub1 = Clock::now();
                                    cpu_submit_slot[slot] += std::chrono::duration<float, std::milli>(_sub1 - _sub0).count();
                                }
                                if (doTiming) CHECK_CUDA(cudaEventRecord(tg_d2h_e[slot], streamG));
                                CHECK_CUDA(cudaEventRecord(evOutG[slot], streamG));
                            });
                        }else{
                            add_cpu_submit(slot, [&](){
                                std::memset(hOutPinned[slot], 0, bytesOutput);
                                CHECK_CUDA(cudaEventRecord(evOutG[slot], streamG));
                            });
                        }
                }
            }

            if (started_g) {
                double sec = std::chrono::duration<double>(t_last_g - t_first_g).count();
                double fps_meas = (sec > 0.0) ? (double(done_g) / sec) : 0.0;
                std::cout << "[BENCH] mode=gpu frames=" << done_g << " time_s=" << sec
                          << " FPS=" << fps_meas << "\n";
            }
        }

        // Flush any pending GPU work
        CHECK_CUDA(cudaDeviceSynchronize());
    
        if (!timings.empty()) {
            printTimingStats(timings);
            writeTimingCsv(cli.timingCsv, timings);
        }

        if (!cli.predJson.empty()) {
            write_coco_detections_json(cli.predJson, allDetections);
            std::cerr << "[PRED_JSON] wrote " << allDetections.size() << " detections to " << cli.predJson << "\n";
        }

        if (!cli.powerCsv.empty()) {
            // Stop tegrastats first to flush buffered lines and avoid trailing samples mismatch.
            pwr.stop();
            auto samples = pwr.samples();
            write_power_csv_and_summary(cli.powerCsv, cli.powerKey, samples);
            // Per-frame aggregates (requires FrameTiming timestamps)
            if (!timings.empty()) {
                write_power_frames_csv_and_summary(cli.powerFramesCsv, cli.powerKey, samples, timings);
            } else {
                std::cerr << "[POWER_FR] timings empty; per-frame power join skipped\n";
            }
        }
        
        if (!cli.utilCsv.empty()) {
            util.stop();
            std::cerr << "[UTIL] jtop logger stopped: " << cli.utilCsv << "\n";
        }
        auto destroyEvent = [&](cudaEvent_t &e){ if (e) { cudaEventDestroy(e); e = nullptr; } };
        auto destroyStream = [&](cudaStream_t &s){ if (s) { cudaStreamDestroy(s); s = nullptr; } };

        /* --- cleanup begins --- */

        // events (GPU-only)
        for (int i=0;i<NBUF;i++) {
            destroyEvent(evOutG[i]);
            destroyEvent(tg_h2d_s[i]); destroyEvent(tg_h2d_e[i]);
            destroyEvent(tg_inf_s[i]); destroyEvent(tg_inf_e[i]);
            destroyEvent(tg_d2h_s[i]); destroyEvent(tg_d2h_e[i]);
        }

        // events (hybrid/common)
        for (int i=0;i<NBUF;i++) {
            destroyEvent(evH2D[i]);
            destroyEvent(evA[i]);
            destroyEvent(evOut[i]);

            destroyEvent(t_h2d_s[i]); destroyEvent(t_h2d_e[i]);
            destroyEvent(t_a_s[i]);   destroyEvent(t_a_e[i]);
            destroyEvent(t_wait_s[i]);destroyEvent(t_wait_e[i]);
            destroyEvent(t_b_s[i]);   destroyEvent(t_b_e[i]);
            destroyEvent(t_d2h_s[i]); destroyEvent(t_d2h_e[i]);
        }

        // streams
        destroyStream(streamG);
        destroyStream(streamH2D);
        destroyStream(streamDLA);
        destroyStream(streamGPU);

        // device buffers (GPU)
        for (int i=0;i<NBUF;i++) {
            if (dGin[i])  { cudaFree(dGin[i]);  dGin[i]=nullptr; }
            if (dGout[i]) { cudaFree(dGout[i]); dGout[i]=nullptr; }
        }

        // device buffers (Hybrid)
        for (int i=0;i<NBUF;i++) {
            if (dAin[i])  { cudaFree(dAin[i]);  dAin[i]=nullptr; }
            if (dBout[i]) { cudaFree(dBout[i]); dBout[i]=nullptr; }
        }

        // feature buffers (Hybrid, dynamic)
        for (int k=0;k<(int)dFeat.size();k++) {
            for (int i=0;i<NBUF;i++) {
                if (dFeat[k][i]) { cudaFree(dFeat[k][i]); dFeat[k][i]=nullptr; }
            }
        }

        // pinned host
        for (int i=0;i<NBUF;i++) {
            if (hInPinned[i])  { cudaFreeHost(hInPinned[i]);  hInPinned[i]=nullptr; }
            if (hOutPinned[i]) { cudaFreeHost(hOutPinned[i]); hOutPinned[i]=nullptr; }
        }

        // optional pinned host samples
        for (int k=0;k<(int)hFeatSample.size();k++) {
            if (hFeatSample[k]) { cudaFreeHost(hFeatSample[k]); hFeatSample[k]=nullptr; }
        }

        /* --- cleanup ends --- */
        return 0;
    }catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return -1;
    }
}