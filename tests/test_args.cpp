#include <iostream>
#include <QString>

int main(int argc, char *argv[]) {
    std::cout << "argc: " << argc << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
    }

    bool simulate_full_app = false;
    for (int i = 1; i < argc; ++i) {
        QString arg = QString::fromLocal8Bit(argv[i]);
        std::cout << "Checking arg: '" << arg.toStdString() << "'" << std::endl;
        if (arg == "--full-mode" || arg == "--simulate-full" || arg.contains("simulate-full")) {
            simulate_full_app = true;
            std::cout << "FOUND simulate-full flag!" << std::endl;
            break;
        }
    }

    std::cout << "Final simulate_full_app: " << simulate_full_app << std::endl;
    return 0;
}
