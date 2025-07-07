#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    QLabel label("Qt is working! This is a test.");
    label.show();
    
    return app.exec();
}
