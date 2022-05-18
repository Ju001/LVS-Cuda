#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <vtkVolumeHandler.h>
#include <vtkRendererCollection.h>
#include <vtkGenericOpenGLRenderWindow.h>

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

public slots:
    virtual void openFile();
    virtual void setTransferFunction();
    virtual void filterVolume();

private:
    Ui::MainWindow *ui;

protected:
    /*!
     * RenderWindow for the unfiltered volume.
     */
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow1;
    /*!
     * RenderWindow for the filtered volume.
     */
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow2;
    /*!
     * The unfiltered volume.
     */
    std::unique_ptr<LVS::VtkVolumeHandler> volume1;
    /*!
     * The filtered volume.
     */
    std::unique_ptr<LVS::VtkVolumeHandler> volume2;
};
#endif // MAINWINDOW_H
