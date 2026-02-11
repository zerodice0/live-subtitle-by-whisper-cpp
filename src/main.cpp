// Live Subtitle - Real-time speech recognition with SSE streaming
//
// Audio capture (SDL2) -> whisper.cpp inference -> HTTP server (SSE) -> Browser

#include "common-sdl.h"
#include "common.h"
#include "whisper.h"
#include "ggml-backend.h"
#include "httplib.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Embedded HTML (web/index.html)
// ---------------------------------------------------------------------------

static const char * const INDEX_HTML = R"html(<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Subtitle</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #000;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            align-items: center;
            padding: 2rem;
            overflow: hidden;
        }
        #subtitle-container {
            text-align: center;
            max-width: 90%;
            transition: opacity 0.5s ease;
        }
        #subtitle {
            font-size: 2.5rem;
            font-weight: 600;
            line-height: 1.4;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
            word-wrap: break-word;
        }
        #original {
            font-size: 1.2rem;
            line-height: 1.4;
            margin-top: 0.5rem;
            opacity: 0.5;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
            word-wrap: break-word;
        }
        #language-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            margin-bottom: 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            background: rgba(255,255,255,0.2);
            opacity: 0.7;
        }
        #status {
            position: fixed;
            top: 1rem;
            right: 1rem;
            font-size: 0.8rem;
            opacity: 0.5;
        }
        #lang-select {
            position: fixed;
            top: 1rem;
            left: 1rem;
            background: rgba(255,255,255,0.15);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 6px;
            padding: 0.4rem 0.6rem;
            font-size: 0.85rem;
            cursor: pointer;
            outline: none;
            display: none;
        }
        #lang-select option { background: #222; color: #fff; }
        .connected { color: #4ade80; }
        .disconnected { color: #f87171; }
        .fade { opacity: 0.3; }
    </style>
</head>
<body>
    <div id="status" class="disconnected">&#9679; Disconnected</div>
    <select id="lang-select"><option value="">Loading...</option></select>
    <div id="subtitle-container">
        <div id="language-badge"></div>
        <div id="subtitle"></div>
        <div id="original"></div>
    </div>
    <script>
        const subtitle = document.getElementById('subtitle');
        const original = document.getElementById('original');
        const langBadge = document.getElementById('language-badge');
        const container = document.getElementById('subtitle-container');
        const status = document.getElementById('status');
        const langSelect = document.getElementById('lang-select');
        let fadeTimer = null;
        let translateEnabled = false;

        function clearSelectOptions(select) {
            while (select.firstChild) select.removeChild(select.firstChild);
        }

        function addOption(select, value, text) {
            const opt = document.createElement('option');
            opt.value = value;
            opt.textContent = text;
            select.appendChild(opt);
        }

        async function loadConfig() {
            try {
                const res = await fetch('/api/config');
                const cfg = await res.json();
                translateEnabled = cfg.translate_enabled;
                if (!translateEnabled) return;

                langSelect.style.display = 'block';

                const langRes = await fetch('/api/languages');
                const languages = await langRes.json();
                if (!languages.length) { langSelect.style.display = 'none'; return; }

                clearSelectOptions(langSelect);
                addOption(langSelect, '', 'Translate off');
                for (const lang of languages) {
                    addOption(langSelect, lang.code, lang.name);
                }
                langSelect.value = cfg.target_lang || '';
            } catch (e) {
                langSelect.style.display = 'none';
            }
        }

        langSelect.addEventListener('change', async () => {
            try {
                await fetch('/api/config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({target_lang: langSelect.value})
                });
            } catch (e) { /* ignore */ }
        });

        function connect() {
            const es = new EventSource('/events');

            es.onopen = () => {
                status.textContent = '\u25CF Connected';
                status.className = 'connected';
            };

            es.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.text) {
                        if (data.translated) {
                            subtitle.textContent = data.translated;
                            original.textContent = data.text;
                        } else {
                            subtitle.textContent = data.text;
                            original.textContent = '';
                        }
                        container.classList.remove('fade');

                        if (data.language) {
                            langBadge.textContent = data.language.toUpperCase();
                        }

                        if (fadeTimer) clearTimeout(fadeTimer);
                        fadeTimer = setTimeout(() => {
                            container.classList.add('fade');
                        }, 5000);
                    }
                } catch (e) { /* ignore parse errors */ }
            };

            es.onerror = () => {
                status.textContent = '\u25CF Disconnected';
                status.className = 'disconnected';
                es.close();
                setTimeout(connect, 2000);
            };
        }

        loadConfig();
        connect();
    </script>
</body>
</html>)html";

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

static std::string escape_json(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Build a JSON string field: "key":"escaped_value"
static std::string json_str(const char * key, const std::string & value) {
    return std::string("\"") + key + "\":\"" + escape_json(value) + "\"";
}

// Build a JSON bool field: "key":true/false
static std::string json_bool(const char * key, bool value) {
    return std::string("\"") + key + "\":" + (value ? "true" : "false");
}

// ---------------------------------------------------------------------------
// Translation via LibreTranslate
// ---------------------------------------------------------------------------

static std::string translate_text(httplib::Client & client,
                                  const std::string & text,
                                  const std::string & source_lang,
                                  const std::string & target_lang) {
    std::string body = "{" + json_str("q", text) +
                       "," + json_str("source", source_lang) +
                       "," + json_str("target", target_lang) + "}";

    auto res = client.Post("/translate", body, "application/json");
    if (!res || res->status != 200) {
        return "";
    }

    // Minimal JSON extraction for {"translatedText":"..."}
    // Handles \" and \\ escapes (sufficient for LibreTranslate responses)
    const std::string needle = "\"translatedText\":\"";
    auto pos = res->body.find(needle);
    if (pos == std::string::npos) {
        return "";
    }
    pos += needle.size();

    std::string result;
    for (size_t i = pos; i < res->body.size(); i++) {
        if (res->body[i] == '\\' && i + 1 < res->body.size()) {
            result += res->body[i + 1];
            i++;
        } else if (res->body[i] == '"') {
            break;
        } else {
            result += res->body[i];
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Shared state between main loop and SSE clients
// ---------------------------------------------------------------------------

struct subtitle_state {
    std::mutex              mtx;
    std::condition_variable cv;
    std::string             text;
    std::string             translated;
    std::string             language;
    std::string             target_lang;
    uint64_t                version = 0;
    bool                    running = true;
};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

struct params {
    int32_t n_threads  = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t port       = 8080;

    float vad_thold = 0.6f;

    bool use_gpu   = true;
    bool flash_attn = true;

    std::string language      = "auto";
    std::string model         = "models/ggml-large-v3-turbo.bin";
    std::string translate_url;
};

static void print_usage(const char * prog) {
    fprintf(stderr, "\nUsage: %s [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --model PATH       Whisper model path      (default: models/ggml-large-v3-turbo.bin)\n");
    fprintf(stderr, "  --port N           HTTP server port        (default: 8080)\n");
    fprintf(stderr, "  --step N           Audio step size in ms   (default: 3000)\n");
    fprintf(stderr, "  --length N         Audio length in ms      (default: 10000)\n");
    fprintf(stderr, "  --keep N           Audio keep in ms        (default: 200)\n");
    fprintf(stderr, "  --threads N        Inference threads       (default: %d)\n",
            std::min(4, (int)std::thread::hardware_concurrency()));
    fprintf(stderr, "  --capture N        Audio device ID         (default: -1 = auto)\n");
    fprintf(stderr, "  --language LANG    Language or 'auto'      (default: auto)\n");
    fprintf(stderr, "  --vad-thold F      VAD energy threshold    (default: 0.6)\n");
    fprintf(stderr, "  --translate-url URL LibreTranslate server   (default: disabled)\n");
    fprintf(stderr, "  --no-gpu           Disable GPU\n");
    fprintf(stderr, "  --no-flash-attn    Disable flash attention\n");
    fprintf(stderr, "  -h, --help         Show this help\n\n");
}

static bool parse_params(int argc, char ** argv, params & p) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--model"     && i+1 < argc) { p.model      = argv[++i]; }
        else if (arg == "--port"      && i+1 < argc) { p.port       = std::stoi(argv[++i]); }
        else if (arg == "--step"      && i+1 < argc) { p.step_ms    = std::stoi(argv[++i]); }
        else if (arg == "--length"    && i+1 < argc) { p.length_ms  = std::stoi(argv[++i]); }
        else if (arg == "--keep"      && i+1 < argc) { p.keep_ms    = std::stoi(argv[++i]); }
        else if (arg == "--threads"   && i+1 < argc) { p.n_threads  = std::stoi(argv[++i]); }
        else if (arg == "--capture"   && i+1 < argc) { p.capture_id = std::stoi(argv[++i]); }
        else if (arg == "--language"  && i+1 < argc) { p.language   = argv[++i]; }
        else if (arg == "--vad-thold" && i+1 < argc) { p.vad_thold  = std::stof(argv[++i]); }
        else if (arg == "--translate-url" && i+1 < argc) { p.translate_url = argv[++i]; }
        else if (arg == "--no-gpu")         { p.use_gpu    = false; }
        else if (arg == "--no-flash-attn")  { p.flash_attn = false; }
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return false; }
        else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------

static std::atomic<bool> g_running{true};

static void signal_handler(int) {
    g_running = false;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    params par;
    if (!parse_params(argc, argv, par)) {
        return 1;
    }

    par.keep_ms   = std::min(par.keep_ms,   par.step_ms);
    par.length_ms = std::max(par.length_ms,  par.step_ms);

    const int n_samples_step = (int)(1e-3 * par.step_ms    * WHISPER_SAMPLE_RATE);
    const int n_samples_len  = (int)(1e-3 * par.length_ms  * WHISPER_SAMPLE_RATE);
    const int n_samples_keep = (int)(1e-3 * par.keep_ms    * WHISPER_SAMPLE_RATE);

    // ── Whisper context ──────────────────────────────────────────────────

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = par.use_gpu;
    cparams.flash_attn = par.flash_attn;

    struct whisper_context * ctx = whisper_init_from_file_with_params(par.model.c_str(), cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to load model '%s'\n", par.model.c_str());
        return 1;
    }

    // ── SDL audio capture ────────────────────────────────────────────────

    audio_async audio(par.length_ms);
    if (!audio.init(par.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "error: audio.init() failed\n");
        whisper_free(ctx);
        return 1;
    }
    audio.resume();

    fprintf(stderr, "\n");
    fprintf(stderr, "model:    %s\n", par.model.c_str());
    fprintf(stderr, "language: %s\n", par.language.c_str());
    fprintf(stderr, "step:     %d ms\n", par.step_ms);
    fprintf(stderr, "length:   %d ms\n", par.length_ms);
    fprintf(stderr, "threads:  %d\n", par.n_threads);
    fprintf(stderr, "\n");

    // ── Shared subtitle state ────────────────────────────────────────────

    subtitle_state state;

    // ── Signal handler ───────────────────────────────────────────────────

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ── HTTP server ──────────────────────────────────────────────────────

    httplib::Server svr;

    svr.Get("/", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(INDEX_HTML, "text/html; charset=utf-8");
    });

    svr.Get("/events", [&state](const httplib::Request &, httplib::Response & res) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Access-Control-Allow-Origin", "*");

        uint64_t client_version = 0;

        res.set_chunked_content_provider("text/event-stream",
            [&state, client_version](size_t /*offset*/, httplib::DataSink & sink) mutable {
                std::unique_lock<std::mutex> lock(state.mtx);

                state.cv.wait_for(lock, std::chrono::seconds(15), [&] {
                    return state.version > client_version || !state.running;
                });

                if (!state.running) {
                    sink.done();
                    return false;
                }

                if (state.version > client_version) {
                    client_version = state.version;
                    std::string json = "{" + json_str("text", state.text) +
                                       "," + json_str("translated", state.translated) +
                                       "," + json_str("language", state.language) + "}";
                    std::string event = "data: " + json + "\n\n";
                    if (!sink.write(event.c_str(), event.size())) {
                        return false;
                    }
                } else {
                    // SSE keepalive comment
                    if (!sink.write(": keepalive\n\n", 13)) {
                        return false;
                    }
                }
                return true;
            }
        );
    });

    // ── Translation API endpoints ───────────────────────────────────────

    svr.Get("/api/languages", [&par](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        if (par.translate_url.empty()) {
            res.set_content("[]", "application/json");
            return;
        }
        httplib::Client client(par.translate_url);
        client.set_connection_timeout(2);
        client.set_read_timeout(3);
        auto r = client.Get("/languages");
        if (r && r->status == 200) {
            res.set_content(r->body, "application/json");
        } else {
            res.set_content("[]", "application/json");
        }
    });

    svr.Get("/api/config", [&state, &par](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        std::lock_guard<std::mutex> lock(state.mtx);
        std::string json = "{" + json_str("target_lang", state.target_lang) +
                           "," + json_bool("translate_enabled", !par.translate_url.empty()) + "}";
        res.set_content(json, "application/json");
    });

    svr.Post("/api/config", [&state](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        const std::string needle = "\"target_lang\":\"";
        auto pos = req.body.find(needle);
        std::string lang;
        if (pos != std::string::npos) {
            pos += needle.size();
            auto end = req.body.find('"', pos);
            if (end != std::string::npos) {
                lang = req.body.substr(pos, end - pos);
            }
        }
        {
            std::lock_guard<std::mutex> lock(state.mtx);
            state.target_lang = lang;
        }
        res.set_content("{\"ok\":true}", "application/json");
    });

    std::thread server_thread([&svr, &par]() {
        fprintf(stderr, "listening on http://localhost:%d\n\n", par.port);
        svr.listen("0.0.0.0", par.port);
    });

    // ── Main audio processing loop ───────────────────────────────────────

    std::vector<float> pcmf32;
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new;

    std::string prev_text;
    int         repeat_count = 0;

    // Translation client (created only if --translate-url is set)
    std::unique_ptr<httplib::Client> translate_client;
    if (!par.translate_url.empty()) {
        translate_client = std::make_unique<httplib::Client>(par.translate_url);
        translate_client->set_connection_timeout(2);
        translate_client->set_read_timeout(3);
        fprintf(stderr, "translation: %s\n\n", par.translate_url.c_str());
    }

    // 1-entry translation cache
    std::string cache_key;
    std::string cache_result;

    while (g_running) {
        // Collect step_ms worth of audio samples
        {
            bool collected = false;
            while (g_running) {
                if (!sdl_poll_events()) {
                    g_running = false;
                    break;
                }

                audio.get(par.step_ms, pcmf32_new);

                if ((int)pcmf32_new.size() > 2 * n_samples_step) {
                    fprintf(stderr, "warning: cannot process audio fast enough, dropping samples\n");
                    audio.clear();
                    continue;
                }
                if ((int)pcmf32_new.size() >= n_samples_step) {
                    audio.clear();
                    collected = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (!collected) break;
        }

        const int n_samples_new = (int)pcmf32_new.size();

        // Energy-based VAD — skip silent audio to prevent hallucination
        {
            constexpr float silence_threshold = 0.0001f;
            float energy = 0.0f;
            for (float sample : pcmf32_new) {
                energy += fabsf(sample);
            }
            energy /= n_samples_new;
            if (energy < silence_threshold) {
                continue;
            }
        }

        // Combine previous (keep) + new audio
        const int n_samples_take = std::min((int)pcmf32_old.size(),
            std::max(0, n_samples_keep + n_samples_len - n_samples_new));

        pcmf32.resize(n_samples_new + n_samples_take);

        if (n_samples_take > 0) {
            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[(int)pcmf32_old.size() - n_samples_take + i];
            }
        }
        std::copy(pcmf32_new.begin(), pcmf32_new.end(),
                 pcmf32.begin() + n_samples_take);

        pcmf32_old = pcmf32;

        // ── Whisper inference ────────────────────────────────────────────

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        wparams.print_progress   = false;
        wparams.print_special    = false;
        wparams.print_realtime   = false;
        wparams.print_timestamps = false;
        wparams.translate        = false;
        wparams.single_segment   = true;
        wparams.max_tokens       = 0;
        wparams.language         = par.language.c_str();
        wparams.n_threads        = par.n_threads;
        wparams.audio_ctx        = 0;
        wparams.temperature_inc  = 0.0f;   // no fallback — faster

        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
            fprintf(stderr, "warning: whisper_full() failed\n");
            continue;
        }

        // ── Collect result ───────────────────────────────────────────────

        std::string text;
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; i++) {
            text += whisper_full_get_segment_text(ctx, i);
        }

        text = trim(text);
        if (text.empty()) continue;

        // Anti-hallucination: skip repeated identical text
        if (text == prev_text) {
            if (++repeat_count >= 3) continue;
        } else {
            repeat_count = 0;
            prev_text = text;
        }

        // Detected language
        const int lang_id = whisper_full_lang_id(ctx);
        const std::string lang = (lang_id >= 0) ? whisper_lang_str(lang_id) : "??";

        // ── Translation (outside mutex) ──────────────────────────────────

        std::string translated;
        std::string target_lang;

        if (translate_client) {
            {
                std::lock_guard<std::mutex> lock(state.mtx);
                target_lang = state.target_lang;
            }

            if (!target_lang.empty() && target_lang != lang) {
                // Tab separator avoids collision with text/lang content
                std::string cache_check = text + "\t" + target_lang;
                if (cache_check == cache_key) {
                    translated = cache_result;
                } else {
                    translated = translate_text(*translate_client, text, lang, target_lang);
                    cache_key = cache_check;
                    cache_result = translated;
                    if (translated.empty()) {
                        fprintf(stderr, "warning: translation failed\n");
                    }
                }
            }
        }

        // ── Update shared state → notify SSE clients ─────────────────────

        {
            std::lock_guard<std::mutex> lock(state.mtx);
            state.text       = text;
            state.translated = translated;
            state.language   = lang;
            state.version++;
        }
        state.cv.notify_all();

        if (!translated.empty()) {
            fprintf(stderr, "[%s->%s] %s -> %s\n", lang.c_str(), target_lang.c_str(),
                    text.c_str(), translated.c_str());
        } else {
            fprintf(stderr, "[%s] %s\n", lang.c_str(), text.c_str());
        }
    }

    // ── Graceful shutdown ────────────────────────────────────────────────

    fprintf(stderr, "\nshutting down...\n");

    {
        std::lock_guard<std::mutex> lock(state.mtx);
        state.running = false;
    }
    state.cv.notify_all();

    svr.stop();
    if (server_thread.joinable()) {
        server_thread.join();
    }

    audio.pause();
    whisper_free(ctx);

    return 0;
}
