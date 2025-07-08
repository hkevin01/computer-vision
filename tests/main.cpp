#include <QApplication>
#include <gtest/gtest.h>

int main(int argc, char **argv) {
  // Initialize Google Test
  ::testing::InitGoogleTest(&argc, argv);

  // Create a QApplication instance for GUI tests
  QApplication app(argc, argv);

  // Run all tests
  return RUN_ALL_TESTS();
}
