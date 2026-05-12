#include <string>
#include <stdexcept>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

class UdpSocket {
public:
    // Constructor: creates and binds a UDP socket to INADDR_ANY:port (default 8090)
    UdpSocket(int port = 8090) : sockfd(-1), have_peer(false) {
        // Create socket
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) {
            throw std::runtime_error("Failed to create socket: " + std::string(strerror(errno)));
        }

        // Allow reuse of address (helps when restarting the program)
        int opt = 1;
        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            close(sockfd);
            throw std::runtime_error("Failed to set SO_REUSEADDR: " + std::string(strerror(errno)));
        }

        // Bind to local address and port
        memset(&local_addr, 0, sizeof(local_addr));
        local_addr.sin_family = AF_INET;
        local_addr.sin_addr.s_addr = INADDR_ANY;
        local_addr.sin_port = htons(port);

        if (bind(sockfd, (struct sockaddr*)&local_addr, sizeof(local_addr)) < 0) {
            close(sockfd);
            throw std::runtime_error("Failed to bind socket to port " + std::to_string(port) +
                                     ": " + std::string(strerror(errno)));
        }
    }

    ~UdpSocket() {
        if (sockfd >= 0) {
            close(sockfd);
        }
    }

    // Send a message to the last sender (stored by the last get() call)
    void send(const std::string& message) {
        if (!have_peer) {
            throw std::runtime_error("No peer address available. Call get() first to receive a message.");
        }
        ssize_t n = sendto(sockfd, message.c_str(), message.size(), 0,
                           (struct sockaddr*)&peer_addr, peer_addr_len);
        if (n < 0) {
            throw std::runtime_error("Failed to send message: " + std::string(strerror(errno)));
        }
        if (static_cast<size_t>(n) != message.size()) {
            throw std::runtime_error("Partial send on UDP? (should not happen)");
        }
    }

    // Receive a message, store the sender's address for subsequent send()
    std::string get() {
        char buffer[65536];  // Maximum practical UDP datagram size
        peer_addr_len = sizeof(peer_addr);
        ssize_t n = recvfrom(sockfd, buffer, sizeof(buffer) - 1, 0,
                             (struct sockaddr*)&peer_addr, &peer_addr_len);
        if (n < 0) {
            throw std::runtime_error("Failed to receive message: " + std::string(strerror(errno)));
        }
        buffer[n] = '\0';  // Null-terminate for safety (though string constructor uses n)
        have_peer = true;
        return std::string(buffer, n);
    }

    // Optional: send to an explicit address and port (ignores stored peer)
    void sendTo(const std::string& message, const std::string& ip, int port) {
        struct sockaddr_in dest_addr;
        memset(&dest_addr, 0, sizeof(dest_addr));
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(port);
        if (inet_pton(AF_INET, ip.c_str(), &dest_addr.sin_addr) <= 0) {
            throw std::runtime_error("Invalid IP address: " + ip);
        }
        ssize_t n = sendto(sockfd, message.c_str(), message.size(), 0,
                           (struct sockaddr*)&dest_addr, sizeof(dest_addr));
        if (n < 0) {
            throw std::runtime_error("Failed to send message: " + std::string(strerror(errno)));
        }
    }

private:
    int sockfd;
    struct sockaddr_in local_addr;
    struct sockaddr_in peer_addr;   // Address of the last sender
    socklen_t peer_addr_len;        // Length of peer_addr
    bool have_peer;                 // True after a successful get()
};
