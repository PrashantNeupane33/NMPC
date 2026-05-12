#include <arpa/inet.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

struct recv_data {
  float x;
  float y;
  float z;
};

struct send_data {
  float vx;
  float w;
};

std::atomic<recv_data> shared_recv;
std::atomic<send_data> shared_send;

// ---------------- TCP CLIENT THREAD ----------------
static void *tcp_client(void *arg) {

	printf("TCP thread started\n");

  int sock = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in serverAddr{};
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(8080);
  inet_pton(AF_INET, "10.100.61.34", &serverAddr.sin_addr);

reconnect:
  while (connect(sock, (sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
    std::cerr << "TCP connect failed\n";
  }

  char buffer[1024];
  std::string stream;

  while (true) {

    // ---- Receive from server ----
    int bytes = recv(sock, buffer, sizeof(buffer) - 1, 0);
    printf("helo\n");
    if (bytes <= 0) {
      std::cerr << "Server closed\n";
      goto reconnect;
    }
    buffer[bytes] = '\0';
    stream += buffer;

    // Process full lines
    size_t pos;
    while ((pos = stream.find('\n')) != std::string::npos) {
      std::string line = stream.substr(0, pos);
      stream.erase(0, pos + 1);

      // Parse "x,y,z"
      recv_data d{};
      sscanf(line.c_str(), "%f,%f,%f", &d.x, &d.y, &d.z);
      shared_recv.store(d, std::memory_order_relaxed);
    }

    // ---- Send to server ----
    send_data s = shared_send.load(std::memory_order_relaxed);
    std::string reply = std::to_string(s.vx) + "," + std::to_string(s.w) + "\n";
    int sent = send(sock, reply.c_str(), reply.size(), 0);
    if (sent < 0) {
      std::cerr << "Send failed\n";
      goto reconnect;
    }
  }

  close(sock);
  return nullptr;
}
