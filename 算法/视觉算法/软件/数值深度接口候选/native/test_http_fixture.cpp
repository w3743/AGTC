// Loopback-only integration fixture. It is not a production HTTP server.
#include "depth_http.h"
#include "depth_producer.h"
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
using Socket = SOCKET;
static void close_socket(Socket s) { closesocket(s); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using Socket = int;
static void close_socket(Socket s) { close(s); }
#endif

static bool send_all(Socket socket, const char* data, std::size_t length) {
    while (length) {
#ifdef _WIN32
        int n = send(socket, data, static_cast<int>(length), 0);
#else
        auto n = send(socket, data, length, MSG_NOSIGNAL);
#endif
        if (n <= 0) return false;
        data += n;
        length -= static_cast<std::size_t>(n);
    }
    return true;
}
int main() {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return 1;
#endif
    Socket server = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 || listen(server, 4) != 0) return 2;
#ifdef _WIN32
    int address_length = sizeof(address);
#else
    socklen_t address_length = sizeof(address);
#endif
    if (getsockname(server, reinterpret_cast<sockaddr*>(&address), &address_length) != 0) return 3;
    agtc::DepthStore store(2);
    store.set_capture_running(true);
    store.set_geometry_valid(true);
    auto token = store.token();
    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::vector<float> depth(131 * 240, 1.25f);
    agtc::DepthContext context;
    context.session_id = "loopback-fixture";
    context.monotonic_clock_id = "fixture-steady-clock";
    if (!agtc::publish_depth_f32(store, token, 1, now, depth.data(), depth.size(), context)) return 4;
    std::cout << ntohs(address.sin_port) << std::endl;
    for (int count = 0; count < 8; ++count) {
        Socket peer = accept(server, nullptr, nullptr);
#ifdef _WIN32
        if (peer == INVALID_SOCKET) break;
#else
        if (peer < 0) break;
#endif
        std::string request;
        char buffer[2048];
        while (request.find("\r\n\r\n") == std::string::npos && request.size() < 8192) {
            auto n = recv(peer, buffer, sizeof(buffer), 0);
            if (n <= 0) break;
            request.append(buffer, static_cast<std::size_t>(n));
        }
        auto line_end = request.find("\r\n");
        auto separator = request.find(' ', 4);
        if (request.rfind("GET ", 0) == 0 && line_end != std::string::npos &&
            separator != std::string::npos && separator < line_end) {
            auto target = request.substr(4, separator - 4);
            auto response = agtc::route_depth_get(store, target, std::chrono::milliseconds(0), std::chrono::seconds(60));
            if (send_all(peer, response.headers.data(), response.headers.size())) {
                if (response.frame) send_all(peer, reinterpret_cast<const char*>(response.frame->payload.data()), response.frame->payload.size());
                else send_all(peer, response.error_body.data(), response.error_body.size());
            }
        }
        close_socket(peer);
    }
    close_socket(server);
#ifdef _WIN32
    WSACleanup();
#endif
}
