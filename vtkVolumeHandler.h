#ifndef VTKVOLUMEHANDLER_H
#define VTKVOLUMEHANDLER_H

#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkVolume.h>
#include <vtkRenderer.h>
#include <vtkSmartVolumeMapper.h>
#include <vector>
#include <array>
#include <vtkImageData.h>
namespace LVS
{
    class VtkVolumeHandler
    {
    private:
        vtkNew<vtkVolume> volume;
        vtkNew<vtkSmartVolumeMapper> mapper;
        vtkNew<vtkImageData> input;
        vtkSmartPointer<vtkRenderer> renderer;
        std::array<double, 3> spacings;
        std::array<int, 3> dimensions;
        int dataType;

    public:
        VtkVolumeHandler();
        void setTransferFunction(double opacityWindow, double opacityLevel);
        void loadFromMHD(std::string path);
        void importRawData(std::vector<float> &rawdata, int dataType, std::array<int, 3> dimensions, std::array<double, 3> spacings);
        int getDatatype();
        vtkSmartPointer<vtkRenderer> getRenderer();
        std::vector<float> exportRawData();
        std::array<int, 3> getDimensions();
        std::array<double, 3> getSpacings();

    private:
        template <class T>
        std::vector<float> exportAndNormaliseRawData();
        template <class T>
        void importAndDenormaliseRawData(std::vector<float> &rawdata, int datatype, std::array<int, 3> dimensions, std::array<double, 3> spacings);
    };
}

#endif // VTKVOLUMEHANDLER_H
