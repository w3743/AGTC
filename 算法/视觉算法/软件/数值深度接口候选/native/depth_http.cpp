#include "depth_http.h"
#include <utility>

namespace agtc {
namespace {
int hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
bool decode(std::string_view input, std::string& output) {
    output.clear();
    for (std::size_t i = 0; i < input.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(input[i]);
        if (c == '+') c = ' ';
        else if (c == '%') {
            if (i + 2 >= input.size() || hex(input[i + 1]) < 0 || hex(input[i + 2]) < 0) return false;
            c = static_cast<unsigned char>(hex(input[i + 1]) * 16 + hex(input[i + 2]));
            i += 2;
        }
        if (c < 0x20 || c == 0x7f) return false;
        output.push_back(static_cast<char>(c));
    }
    return true;
}
const char* status_text(int status) {
    switch (status) { case 200: return "OK"; case 400: return "Bad Request";
        case 404: return "Not Found"; case 409: return "Conflict";
        default: return "Service Unavailable"; }
}
HttpDepthResponse error(int status, const char* state) {
    std::string body = std::string("{\"ok\":false,\"state\":\"") + state + "\"}";
    std::string head = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) +
        "\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    return {status, std::move(head), std::move(body), {}};
}
} // namespace

HttpDepthResponse route_depth_get(DepthStore& store, std::string_view target,
                                 std::chrono::nanoseconds timeout,
                                 std::chrono::nanoseconds max_age,
                                 const std::function<void()>& on_valid_request) {
    if (target.size() > 8192 || target.find('#') != std::string_view::npos) return error(400, "invalid_target");
    auto question = target.find('?');
    if (target.substr(0, question) != "/depth.bin") return error(404, "not_found");
    std::optional<std::vector<std::string>> sequence;
    if (question != std::string_view::npos) {
        auto query = target.substr(question + 1);
        while (!query.empty()) {
            auto amp = query.find('&');
            auto entry = query.substr(0, amp);
            auto equal = entry.find('=');
            std::string key, value;
            if (!decode(entry.substr(0, equal), key) ||
                !decode(equal == std::string_view::npos ? std::string_view{} : entry.substr(equal + 1), value))
                return error(400, "invalid_query");
            if (key == "sequence") {
                if (!sequence) sequence.emplace();
                sequence->push_back(std::move(value));
            }
            if (amp == std::string_view::npos) break;
            query.remove_prefix(amp + 1);
        }
    }
    auto result = handle_depth_request(store, sequence, timeout, max_age, on_valid_request);
    if (result.status != 200 || !result.frame) {
        const char* state = result.reason == DepthStatus::warming ? "warming" :
                            result.status == 400 ? "invalid_sequence" :
                            result.status == 409 ? "conflict" : "unavailable";
        return error(result.status, state);
    }
    const auto& metadata = result.frame->metadata_ascii;
    if (metadata.empty() || metadata.size() > 8192 || result.frame->payload.size() != 131 * 240 * 2)
        return error(503, "invalid_frame");
    for (unsigned char c : metadata) if (c < 0x20 || c > 0x7e) return error(503, "invalid_metadata");
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: " +
        std::to_string(result.frame->payload.size()) + "\r\nCache-Control: no-store\r\nX-Depth-Metadata: " +
        metadata + "\r\nConnection: close\r\n\r\n";
    return {200, std::move(head), {}, std::move(result.frame)};
}
} // namespace agtc
