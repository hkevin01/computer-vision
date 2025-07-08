#ifdef BUILD_GUI
#include "gui/main_window.hpp"
#include <QApplication>
#include <QSignalSpy>
#include <QTest>
#include <QTimer>
#endif
#include <gtest/gtest.h>

#ifdef BUILD_GUI
class GuiTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up any necessary objects for the GUI tests
    // Note: QApplication is created in main.cpp
  }

  // void TearDown() override {}
};

TEST_F(GuiTest, MainWindowCreation) {
  // This test checks if the main window can be created without crashing.
  stereo_vision::gui::MainWindow *w = nullptr;
  ASSERT_NO_THROW(w = new stereo_vision::gui::MainWindow());
  if (w) {
    // Optionally, show the window for a very short time to see if it renders
    w->show();
    QTest::qWait(100); // Wait 100ms
    w->hide();
    delete w;
  }
}

TEST_F(GuiTest, PlaceholderGuiTest) {
  // A placeholder test for future GUI interaction tests
  QString str = "hello";
  QCOMPARE(str.toUpper(), QString("HELLO"));
}

// Test window initialization
TEST_F(GuiTest, WindowInitialization) {
  stereo_vision::gui::MainWindow mainWindow;
  EXPECT_TRUE(mainWindow.isVisible() == false); // Should be hidden by default

  // Test window title
  mainWindow.setWindowTitle("Test Window");
  EXPECT_EQ(mainWindow.windowTitle().toStdString(), "Test Window");
}

// Test GUI responsiveness
TEST_F(GuiTest, WindowShowHide) {
  stereo_vision::gui::MainWindow mainWindow;

  // Show the window
  mainWindow.show();
  QTest::qWait(50);
  EXPECT_TRUE(mainWindow.isVisible());

  // Hide the window
  mainWindow.hide();
  QTest::qWait(50);
  EXPECT_FALSE(mainWindow.isVisible());
}

// Test Qt Framework integration
TEST_F(GuiTest, QtFrameworkIntegration) {
  // Test QString functionality
  QString testString = "StereoVision";
  EXPECT_EQ(testString.length(), 12);
  EXPECT_TRUE(testString.contains("Vision"));

  // Test QTimer functionality
  QTimer timer;
  timer.setSingleShot(true);
  timer.setInterval(100);

  QSignalSpy spy(&timer, &QTimer::timeout);
  timer.start();

  QTest::qWait(150);
  EXPECT_EQ(spy.count(), 1);
}

#endif // BUILD_GUI
