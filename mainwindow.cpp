#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "vtkVolumeHandler.h"
#include "kernel/LVSKernel.cuh"
#include <qfiledialog.h>
#include <memory>
#include <array>
#include <string>
#include <map>
MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent), ui(new Ui::MainWindow), volume1(std::make_unique<LVS::VtkVolumeHandler>()), volume2(std::make_unique<LVS::VtkVolumeHandler>())
{
	ui->setupUi(this);
	renderWindow1->AddRenderer(volume1->getRenderer());
	this->ui->view_unfiltered->setRenderWindow(renderWindow1);

	renderWindow2->AddRenderer(volume2->getRenderer());
	this->ui->view_filtered->setRenderWindow(renderWindow2);

	connect(this->ui->actionOpen, SIGNAL(triggered()), this, SLOT(openFile()));
	connect(this->ui->doubleSpinBox_opWindow, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),
			this, &MainWindow::setTransferFunction);
	connect(this->ui->doubleSpinBox_opLevel, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),
			this, &MainWindow::setTransferFunction);
	connect(this->ui->button_filter, SIGNAL(pressed()), this, SLOT(filterVolume()));
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::filterVolume()
{
	int samples = ui->doubleSpinbox_samples->value();
	int angles = ui->doubleSpinBox_angles->value();
	int integration = ui->doubleSpinBox_steps->value();

	// Filter volume.
	std::unique_ptr<LVS::LVSKernel> lvs = std::make_unique<LVS::LVSKernel>();
	std::vector<float> normalisedVoxels = volume1->exportRawData();
	std::array<int, 3> dimensions = volume1->getDimensions();
	std::map<std::string, int> params = {{"samples", samples},
										 {"integrationSteps", integration},
										 {"sampleAngles", angles}};
	std::vector<float> filteredVoxels = lvs->filter(normalisedVoxels, dimensions, params);
	volume2->importRawData(filteredVoxels, volume1->getDatatype(), dimensions, volume1->getSpacings());
	renderWindow2->GetRenderers()->GetFirstRenderer()->ResetCamera();
	renderWindow2->Render();
	return;
}

void MainWindow::openFile()
{
	QString title = "Please select metaImage Header Data";
	QString filter = "All files (*.*);;MHD (*.mhd)";
	QString selfilter = "MHD (*.mhd)";
	QString headerName = QFileDialog::getOpenFileName(this, title, nullptr, filter, &selfilter);
	if (headerName != nullptr)
	{
		this->volume1->loadFromMHD(headerName.toStdString());
		renderWindow1->GetRenderers()->GetFirstRenderer()->ResetCamera();
		renderWindow1->Render();
		this->ui->button_filter->setEnabled(true);
	}
}

void MainWindow::setTransferFunction()
{
	double opacityLevel = this->ui->doubleSpinBox_opLevel->value();
	double opacityWindow = this->ui->doubleSpinBox_opWindow->value();
	// Create our transfer function
	volume1->setTransferFunction(opacityLevel, opacityWindow);
	volume2->setTransferFunction(opacityLevel, opacityWindow);
	renderWindow1->Render();
	renderWindow2->Render();
}