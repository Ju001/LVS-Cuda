#include "vtkVolumeHandler.h"
#include <vtkMetaImageReader.h>
#include <vtkVolumeProperty.h>
#include <vtkColorTransferFunction.h>
#include <vtkImageImport.h>
#include <vtkImageExport.h>
#include <vtkPiecewiseFunction.h>
#include <iostream>
#include <memory>
#include <limits>
// #include <algorithm>
namespace LVS
{
    VtkVolumeHandler::VtkVolumeHandler() : renderer(vtkRenderer::New())
    {
        // Create the property and attach the transfer functions
        vtkNew<vtkVolumeProperty> property;
        property->SetIndependentComponents(true);
        property->SetInterpolationTypeToLinear();

        // connect up the volume to the property and the mapper
        volume->SetProperty(property);

        mapper->SetBlendModeToComposite();
        property->ShadeOff();

        this->setTransferFunction(50, 50);
        renderer->AddVolume(volume);
    }

    // Extract TransferFunction
    void VtkVolumeHandler::setTransferFunction(double opacityWindow, double opacityLevel)
    {
        vtkSmartPointer<vtkColorTransferFunction> colorFun = vtkColorTransferFunction::New();
        vtkSmartPointer<vtkPiecewiseFunction> opacityFun = vtkPiecewiseFunction::New();
        volume->GetProperty()->SetColor(colorFun);
        volume->GetProperty()->SetScalarOpacity(opacityFun);
        colorFun->AddRGBSegment(opacityLevel - 0.5 * opacityWindow, 0.0, 0.0, 0.0,
                                opacityLevel + 0.5 * opacityWindow, 1.0, 1.0, 1.0);
        opacityFun->AddSegment(opacityLevel - 0.5 * opacityWindow, 0.0,
                               opacityLevel + 0.5 * opacityWindow, 1.0);
    }

    void VtkVolumeHandler::loadFromMHD(std::string path)
    {
        cout << "Loading from " << path << endl;
        // Read the data
        vtkNew<vtkMetaImageReader> metaReader;
        metaReader->SetFileName(path.c_str());
        metaReader->Update();
        input->DeepCopy(metaReader->GetOutput());
        mapper->SetInputData(input);
        volume->SetMapper(mapper);
        this->dataType = metaReader->GetDataScalarType();
        double *spacings_c = metaReader->GetDataSpacing();
        std::copy(spacings_c, spacings_c + this->spacings.size(), this->spacings.begin());
        int *dimensions_c = this->input->GetDimensions();
        std::copy(dimensions_c, dimensions_c + dimensions.size(), dimensions.begin());
    }

    template <class T>
    std::vector<float> VtkVolumeHandler::exportAndNormaliseRawData()
    {
        vtkNew<vtkImageExport> exporter;
        exporter->SetInputData(input);
        exporter->ImageLowerLeftOn();
        exporter->Update();
        std::array<int, 3> dimensions = this->getDimensions();
        int arraySize = dimensions[0] * dimensions[1] * dimensions[2];
        std::vector<T> cImage(arraySize, 0);
        std::vector<float> normalisedVoxels(arraySize, 0.0f);
        T min = std::numeric_limits<T>::min();
        T max = std::numeric_limits<T>::max();
        exporter->SetExportVoidPointer(&cImage.front());
        exporter->Export();
        // Normalise raw Data
        for (int i = 0; i < arraySize; i++)
        {
            float value = (float)(cImage[i] - min) / (float)(max - min);
            normalisedVoxels[i] = (float)value;
        }
        return normalisedVoxels;
    }

    std::vector<float> VtkVolumeHandler::exportRawData()
    {
        std::vector<float> normalisedVoxels;
        int dataType = this->getDatatype();
        switch (dataType)
        {
        case 3:
        {
            normalisedVoxels = this->exportAndNormaliseRawData<unsigned char>();
            break;
        }
        case 4:
        {
            normalisedVoxels = this->exportAndNormaliseRawData<short>();
            break;
        }
        case 5:
        {
            normalisedVoxels = this->exportAndNormaliseRawData<unsigned short>();
            break;
        }
        }
        return normalisedVoxels;
    }

    template <class T>
    void VtkVolumeHandler::importAndDenormaliseRawData(std::vector<float> &rawdata, int dataType, std::array<int, 3> dimensions, std::array<double, 3> spacings)
    {

        int arraySize = dimensions[0] * dimensions[1] * dimensions[2];
        std::vector<T> cImage(arraySize, 0);
        T min = std::numeric_limits<T>::min();
        T max = std::numeric_limits<T>::max();
        for (int i = 0; i < arraySize; i++)
        {
            T value = (T)(rawdata[i] * (max - min)) + min;
            cImage[i] = (T)value;
        }
        vtkNew<vtkImageImport> importer;
        importer->SetDataSpacing(spacings[0], spacings[1], spacings[2]);
        importer->SetDataOrigin(0, 0, 0);
        importer->SetWholeExtent(0, dimensions[0] - 1, 0, dimensions[1] - 1, 0, dimensions[2] - 1);
        importer->SetDataExtentToWholeExtent();
        importer->SetDataScalarType(dataType);
        importer->SetNumberOfScalarComponents(1);
        importer->CopyImportVoidPointer((void *)&cImage.front(), sizeof(T) * arraySize);
        importer->Update();
        input->DeepCopy(importer->GetOutput());
        mapper->SetInputData(input);
        volume->SetMapper(mapper);
    }

    void VtkVolumeHandler::importRawData(std::vector<float> &rawdata, int dataType, std::array<int, 3> dimensions, std::array<double, 3> spacings)
    {
        switch (dataType)
        {
        case 3:
        {
            this->importAndDenormaliseRawData<unsigned char>(rawdata, dataType, dimensions, spacings);
            break;
        }
        case 4:
        {
            this->importAndDenormaliseRawData<short>(rawdata, dataType, dimensions, spacings);
            break;
        }
        case 5:
        {
            this->importAndDenormaliseRawData<unsigned short>(rawdata, dataType, dimensions, spacings);
            break;
        }
        }

        this->spacings = spacings;
        this->dimensions = dimensions;
        this->dataType = dataType;
    }
    std::array<int, 3> VtkVolumeHandler::getDimensions()
    {
        return this->dimensions;
    }
    std::array<double, 3> VtkVolumeHandler::getSpacings()
    {
        return this->spacings;
    }
    int VtkVolumeHandler::getDatatype()
    {
        return this->dataType;
    }
    vtkSmartPointer<vtkRenderer> VtkVolumeHandler::getRenderer()
    {
        return renderer;
    }
}
