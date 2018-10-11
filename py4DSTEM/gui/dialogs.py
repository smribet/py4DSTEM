#!/Users/Ben/Code/anaconda2/envs/py3/bin/python

import sys
from PyQt5 import QtCore, QtWidgets, QtGui

# Set global style parameters

titleFont = QtGui.QFont()
titleFont.setPointSize(12)
titleFont.setItalic(True)
titleFont.setBold(True)

sectionFont = QtGui.QFont()
sectionFont.setPointSize(12)
sectionFont.setItalic(True)
sectionFont.setBold(False)

normalFont = QtGui.QFont()
normalFont.setPointSize(12)
normalFont.setItalic(False)
normalFont.setBold(False)

smallFont = QtGui.QFont()
smallFont.setPointSize(10)
smallFont.setItalic(False)
smallFont.setBold(False)

control_panel_width=300


class ControlPanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # Container widget        
        scrollableWidget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self)

        ############## Make sub-widgets ##############
        # Provide handles to connect to their widgets

        # Load, Preprocess, Save
        self.widget_LoadPreprocessSave = HideableWidget('Load, Preprocess, Save',LoadPreprocessSaveWidget())
        self.lineEdit_LoadFile = self.widget_LoadPreprocessSave.widget.lineEdit_LoadFile
        self.pushButton_BrowseFiles = self.widget_LoadPreprocessSave.widget.pushButton_BrowseFiles
        self.spinBox_Nx = self.widget_LoadPreprocessSave.widget.spinBox_Nx
        self.spinBox_Ny = self.widget_LoadPreprocessSave.widget.spinBox_Ny
        self.spinBox_Bin_Real = self.widget_LoadPreprocessSave.widget.spinBox_Bin_Real
        self.spinBox_Bin_Diffraction = self.widget_LoadPreprocessSave.widget.spinBox_Bin_Diffraction
        self.pushButton_BinData = self.widget_LoadPreprocessSave.widget.pushButton_BinData
        self.checkBox_Crop_Real = self.widget_LoadPreprocessSave.widget.checkBox_Crop_Real
        self.checkBox_Crop_Diffraction = self.widget_LoadPreprocessSave.widget.checkBox_Crop_Diffraction
        self.pushButton_CropData = self.widget_LoadPreprocessSave.widget.pushButton_CropData
        self.pushButton_EditFileMetadata = self.widget_LoadPreprocessSave.widget.pushButton_EditFileMetadata
        self.pushButton_EditDirectoryMetadata = self.widget_LoadPreprocessSave.widget.pushButton_EditDirectoryMetadata
        self.pushButton_SaveFile = self.widget_LoadPreprocessSave.widget.pushButton_SaveFile
        self.pushButton_SaveDirectory = self.widget_LoadPreprocessSave.widget.pushButton_SaveDirectory

        # Data cube size and shape
        self.sizeAndShapeEditor = HideableWidget('Reshape',DataCubeSizeAndShapeWidget(),initial_state=False)
        self.spinBox_Nx_depre = self.sizeAndShapeEditor.widget.spinBox_Nx
        self.spinBox_Ny_depre = self.sizeAndShapeEditor.widget.spinBox_Ny
        self.lineEdit_Binning_depre = self.sizeAndShapeEditor.widget.lineEdit_Binning
        self.pushButton_BinData_depre = self.sizeAndShapeEditor.widget.pushButton_BinData
        self.pushButton_SetCropWindow_depre = self.sizeAndShapeEditor.widget.pushButton_SetCropWindow
        self.pushButton_CropData_depre = self.sizeAndShapeEditor.widget.pushButton_CropData

        # Create and set layout
        layout.addWidget(self.widget_LoadPreprocessSave,0,QtCore.Qt.AlignTop)
        layout.addWidget(self.sizeAndShapeEditor,0,QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        scrollableWidget.setLayout(layout)

        # Scroll Area Properties
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scrollArea.setWidgetResizable(True)
        scrollArea.setWidget(scrollableWidget)
        scrollArea.setFrameStyle(QtWidgets.QFrame.NoFrame)

        # Set the scroll area container to fill the layout of the entire ControlPanel widget
        vLayout = QtWidgets.QVBoxLayout(self)
        vLayout.addWidget(scrollArea)
        vLayout.setSpacing(0)
        vLayout.setContentsMargins(0,0,0,0)
        self.setLayout(vLayout)

        # Set geometry
        #self.setFixedHeight(600)
        #self.setFixedWidth(300)



class LoadPreprocessSaveWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # Load
        load_widget = QtWidgets.QWidget()
        load_widget_layout = QtWidgets.QVBoxLayout()

        self.label_Filename = QtWidgets.QLabel("Filename")
        self.lineEdit_LoadFile = QtWidgets.QLineEdit("")
        self.pushButton_BrowseFiles = QtWidgets.QPushButton("Browse")

        self.label_Filename.setFont(normalFont)
        self.lineEdit_LoadFile.setFont(normalFont)
        self.pushButton_BrowseFiles.setFont(normalFont)

        line1 = QtWidgets.QHBoxLayout()
        line1.addWidget(self.label_Filename,stretch=0)
        line1.addWidget(self.lineEdit_LoadFile,stretch=1)
        line2 = QtWidgets.QHBoxLayout()
        line2.addWidget(self.pushButton_BrowseFiles,0,QtCore.Qt.AlignRight)

        load_widget_layout.addLayout(line1)
        load_widget_layout.addLayout(line2)
        load_widget_layout.setSpacing(0)
        load_widget_layout.setContentsMargins(0,0,0,0)
        load_widget.setLayout(load_widget_layout)

        # Preprocess
        preprocess_widget = PreprocessingWidget()
        self.spinBox_Nx = preprocess_widget.spinBox_Nx
        self.spinBox_Ny = preprocess_widget.spinBox_Ny
        self.spinBox_Bin_Real = preprocess_widget.spinBox_Bin_Real
        self.spinBox_Bin_Diffraction = preprocess_widget.spinBox_Bin_Diffraction
        self.pushButton_BinData = preprocess_widget.pushButton_BinData
        self.checkBox_Crop_Real = preprocess_widget.checkBox_Crop_Real
        self.checkBox_Crop_Diffraction = preprocess_widget.checkBox_Crop_Diffraction
        self.pushButton_CropData = preprocess_widget.pushButton_CropData
        self.pushButton_EditFileMetadata = preprocess_widget.pushButton_EditFileMetadata
        self.pushButton_EditDirectoryMetadata = preprocess_widget.pushButton_EditDirectoryMetadata

        # Save
        save_widget = QtWidgets.QWidget()
        save_widget_layout = QtWidgets.QHBoxLayout()

        self.pushButton_SaveFile = QtWidgets.QPushButton("File")
        self.pushButton_SaveDirectory = QtWidgets.QPushButton("Directory")

        self.pushButton_SaveFile.setMaximumWidth(80)
        self.pushButton_SaveDirectory.setMaximumWidth(100)

        self.pushButton_SaveFile.setFont(normalFont)
        self.pushButton_SaveDirectory.setFont(normalFont)

        save_widget_layout.addWidget(self.pushButton_SaveDirectory)
        save_widget_layout.addWidget(self.pushButton_SaveFile)
        save_widget.setLayout(save_widget_layout)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(SectionLabel('Load'))
        layout.addWidget(load_widget)
        layout.addWidget(SectionLabel('Preprocess'))
        layout.addWidget(preprocess_widget)
        layout.addWidget(SectionLabel('Save'))
        layout.addWidget(save_widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)
        self.setFixedWidth(control_panel_width)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))

class PreprocessingWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # Reshape
        self.spinBox_Nx = QtWidgets.QSpinBox()
        self.spinBox_Ny = QtWidgets.QSpinBox()
        self.spinBox_Nx.setMinimum(1)
        self.spinBox_Ny.setMinimum(1)
        self.spinBox_Nx.setMaximum(1000000)
        self.spinBox_Ny.setMaximum(1000000)
        self.label_Nx = QtWidgets.QLabel("Nx")
        self.label_Ny = QtWidgets.QLabel("Ny")
        self.label_Reshape = QtWidgets.QLabel("Reshape  ")

        self.spinBox_Nx.setFont(normalFont)
        self.spinBox_Ny.setFont(normalFont)
        self.label_Nx.setFont(smallFont)
        self.label_Ny.setFont(smallFont)
        self.label_Reshape.setFont(normalFont)

        layout_Reshape_Nx = QtWidgets.QHBoxLayout()
        layout_Reshape_Nx.addWidget(self.label_Nx,0,QtCore.Qt.AlignLeft)
        layout_Reshape_Nx.addWidget(self.spinBox_Nx,1,QtCore.Qt.AlignLeft)
        layout_Reshape_Ny = QtWidgets.QHBoxLayout()
        layout_Reshape_Ny.addWidget(self.label_Ny,0,QtCore.Qt.AlignLeft)
        layout_Reshape_Ny.addWidget(self.spinBox_Ny,1,QtCore.Qt.AlignLeft)

        layout_Reshape_N = QtWidgets.QVBoxLayout()
        layout_Reshape_N.addLayout(layout_Reshape_Nx,0)
        layout_Reshape_N.addLayout(layout_Reshape_Ny,0)

        layout_Reshape = QtWidgets.QHBoxLayout()
        layout_Reshape.addWidget(self.label_Reshape,1,QtCore.Qt.AlignRight)
        layout_Reshape.addLayout(layout_Reshape_N,1)
        layout_Reshape.setContentsMargins(0,0,0,13)

        # Bin
        self.spinBox_Bin_Real = QtWidgets.QSpinBox()
        self.spinBox_Bin_Diffraction = QtWidgets.QSpinBox()
        self.spinBox_Bin_Real.setMaximum(1000)
        self.spinBox_Bin_Diffraction.setMaximum(1000)
        self.pushButton_BinData = QtWidgets.QPushButton("Bin")

        self.spinBox_Bin_Real.setFont(normalFont)
        self.spinBox_Bin_Diffraction.setFont(normalFont)
        self.label_Bin_Q = QtWidgets.QLabel("Q ")
        self.label_Bin_R = QtWidgets.QLabel("R ")
        self.label_Bin_Q.setFont(smallFont)
        self.label_Bin_R.setFont(smallFont)
        self.pushButton_BinData.setFont(normalFont)

        layout_Bin_Diffraction = QtWidgets.QHBoxLayout()
        layout_Bin_Diffraction.addWidget(self.label_Bin_Q,0,QtCore.Qt.AlignRight)
        layout_Bin_Diffraction.addWidget(self.spinBox_Bin_Diffraction,0,QtCore.Qt.AlignCenter)
        layout_Bin_Real = QtWidgets.QHBoxLayout()
        layout_Bin_Real.addWidget(self.label_Bin_R,0,QtCore.Qt.AlignRight)
        layout_Bin_Real.addWidget(self.spinBox_Bin_Real,0,QtCore.Qt.AlignCenter)

        layout_Bin_SpinBoxes = QtWidgets.QVBoxLayout()
        layout_Bin_SpinBoxes.addLayout(layout_Bin_Diffraction)
        layout_Bin_SpinBoxes.addLayout(layout_Bin_Real)

        layout_Bin = QtWidgets.QHBoxLayout()
        layout_Bin.addLayout(layout_Bin_SpinBoxes,2)
        layout_Bin.addWidget(self.pushButton_BinData,1,QtCore.Qt.AlignCenter)
        layout_Bin.setContentsMargins(10,0,0,0)

        # Crop
        self.checkBox_Crop_Real = QtWidgets.QCheckBox()
        self.checkBox_Crop_Diffraction = QtWidgets.QCheckBox()
        self.pushButton_CropData = QtWidgets.QPushButton("Crop")
        self.label_Crop_Q = QtWidgets.QLabel("Q ")
        self.label_Crop_R = QtWidgets.QLabel("R ")

        self.pushButton_CropData.setFont(normalFont)
        self.label_Crop_Q.setFont(smallFont)
        self.label_Crop_R.setFont(smallFont)

        layout_Crop_Diffraction = QtWidgets.QHBoxLayout()
        layout_Crop_Diffraction.addWidget(self.label_Crop_Q,1,QtCore.Qt.AlignRight)
        layout_Crop_Diffraction.addWidget(self.checkBox_Crop_Diffraction,0,QtCore.Qt.AlignRight)
        layout_Crop_Real = QtWidgets.QHBoxLayout()
        layout_Crop_Real.addWidget(self.label_Crop_R,1,QtCore.Qt.AlignRight)
        layout_Crop_Real.addWidget(self.checkBox_Crop_Real,0,QtCore.Qt.AlignRight)

        layout_Crop_CheckBoxes = QtWidgets.QVBoxLayout()
        layout_Crop_CheckBoxes.addLayout(layout_Crop_Diffraction)
        layout_Crop_CheckBoxes.addLayout(layout_Crop_Real)

        layout_Crop = QtWidgets.QHBoxLayout()
        layout_Crop.addLayout(layout_Crop_CheckBoxes,4)
        layout_Crop.addWidget(self.pushButton_CropData,1,QtCore.Qt.AlignLeft)
        layout_Crop.setSpacing(0)
        layout_Crop.setContentsMargins(0,0,10,0)

        # Crop and Bin

        layout_CropAndBin = QtWidgets.QHBoxLayout()
        layout_CropAndBin.addLayout(layout_Crop)
        layout_CropAndBin.addLayout(layout_Bin)
        layout_CropAndBin.setContentsMargins(0,0,0,6)

        # Edit Metadata
        self.pushButton_EditFileMetadata = QtWidgets.QPushButton("File")
        self.pushButton_EditDirectoryMetadata = QtWidgets.QPushButton("Directory")
        self.label_EditMetadata = QtWidgets.QLabel("Edit Metadata")

        self.pushButton_EditFileMetadata.setMaximumWidth(80)
        self.pushButton_EditDirectoryMetadata.setMaximumWidth(100)

        self.pushButton_EditFileMetadata.setFont(normalFont)
        self.pushButton_EditDirectoryMetadata.setFont(normalFont)
        self.label_EditMetadata.setFont(normalFont)

        layout_EditMetadata_Buttons = QtWidgets.QHBoxLayout()
        layout_EditMetadata_Buttons.addWidget(self.pushButton_EditDirectoryMetadata)
        layout_EditMetadata_Buttons.addWidget(self.pushButton_EditFileMetadata)

        layout_EditMetadata = QtWidgets.QVBoxLayout()
        layout_EditMetadata.addWidget(self.label_EditMetadata,0,QtCore.Qt.AlignCenter)
        layout_EditMetadata.addLayout(layout_EditMetadata_Buttons)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_Reshape)
        layout.addLayout(layout_CropAndBin)
        layout.addLayout(layout_EditMetadata)
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)

        self.setLayout(layout)
        self.setFixedWidth(control_panel_width)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))


class DataCubeSizeAndShapeWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # Reshaping - Nx and Ny
        self.spinBox_Nx = QtWidgets.QSpinBox()
        self.spinBox_Ny = QtWidgets.QSpinBox()
        self.spinBox_Nx.setMaximum(100000)
        self.spinBox_Ny.setMaximum(100000)

        layout_spinBoxRow = QtWidgets.QHBoxLayout()
        layout_spinBoxRow.addWidget(QtWidgets.QLabel("Nx"),0,QtCore.Qt.AlignRight)
        layout_spinBoxRow.addWidget(self.spinBox_Nx)
        layout_spinBoxRow.addWidget(QtWidgets.QLabel("Ny"),0,QtCore.Qt.AlignRight)
        layout_spinBoxRow.addWidget(self.spinBox_Ny)

        layout_Reshaping = QtWidgets.QVBoxLayout()
        #layout_Reshaping.addWidget(QtWidgets.QLabel("Scan shape"),0,QtCore.Qt.AlignCenter)
        layout_Reshaping.addLayout(layout_spinBoxRow)

        # Binning
        self.lineEdit_Binning = QtWidgets.QLineEdit("")
        self.pushButton_BinData = QtWidgets.QPushButton("Bin Data")

        layout_binningRow = QtWidgets.QHBoxLayout()
        layout_binningRow.addWidget(QtWidgets.QLabel("Bin by:"))
        layout_binningRow.addWidget(self.lineEdit_Binning)
        layout_binningRow.addWidget(self.pushButton_BinData)

        layout_Binning = QtWidgets.QVBoxLayout()
        layout_Binning.addWidget(QtWidgets.QLabel("Binning"),0,QtCore.Qt.AlignCenter)
        layout_Binning.addLayout(layout_binningRow)

        # Cropping
        self.pushButton_SetCropWindow = QtWidgets.QPushButton("Set Crop Window")
        self.pushButton_CropData = QtWidgets.QPushButton("Crop Data")

        layout_croppingRow = QtWidgets.QHBoxLayout()
        layout_croppingRow.addWidget(self.pushButton_SetCropWindow)
        layout_croppingRow.addWidget(self.pushButton_CropData)

        layout_Cropping = QtWidgets.QVBoxLayout()
        layout_Cropping.addWidget(QtWidgets.QLabel("Cropping"),0,QtCore.Qt.AlignCenter)
        layout_Cropping.addLayout(layout_croppingRow)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(SectionLabel('Scan Shape'))
        layout.addLayout(layout_Reshaping)
        layout.addLayout(layout_Binning)
        layout.addLayout(layout_Cropping)
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)

        self.setLayout(layout)
        self.setFixedWidth(control_panel_width)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))

class SaveWidget(QtWidgets.QWidget):
    """
    Takes one argument - save_path - a string with a filename for the output file.
    """
    def __init__(self, save_path):
        QtWidgets.QWidget.__init__(self)

        # Label, Line Edit
        self.label_SaveAs = QtWidgets.QLabel("Save as: ")
        self.lineEdit_SavePath = QtWidgets.QLineEdit(save_path)
        self.pushButton_Execute = QtWidgets.QPushButton("Save")
        self.pushButton_Cancel = QtWidgets.QPushButton("Cancel")

        # Layout
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.label_SaveAs, stretch=0)
        top_row.addWidget(self.lineEdit_SavePath, stretch=5)

        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.addWidget(self.pushButton_Cancel,0,QtCore.Qt.AlignLeft)
        bottom_row.addWidget(self.pushButton_Execute,0,QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_row)
        layout.addLayout(bottom_row)

        self.setLayout(layout)
        #self.setFixedWidth(260)
        #self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))

class EditMetadataWidget(QtWidgets.QWidget):
    """
    Creates a widget for viewing and editing metadata. Must receive a DataCube object as an
    argument, to populate metadata fields.
    """
    def __init__(self, datacube):
        QtWidgets.QWidget.__init__(self)

        self.tab_microscope = self.make_tab(datacube.metadata.microscope)
        self.tab_sample = self.make_tab(datacube.metadata.sample)
        self.tab_user = self.make_tab(datacube.metadata.user)
        self.tab_processing = self.make_tab(datacube.metadata.processing)
        self.tab_calibration = self.make_tab(datacube.metadata.calibration)

        # Comments tab - make separately to create larger text box
        tab_comments_layout = QtWidgets.QVBoxLayout()
        for key,value in datacube.metadata.comments.items():
            current_comment = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel(key)
            try:
                text = value.decode('utf-8')
            except AttributeError:
                text = str(value)
            textedit = QtWidgets.QPlainTextEdit(text)
            current_comment.addWidget(label,0,QtCore.Qt.AlignLeft)
            current_comment.addWidget(textedit)
            tab_comments_layout.addLayout(current_comment)
        self.tab_comments = QtWidgets.QWidget()
        self.tab_comments.setLayout(tab_comments_layout)

        # Add all tabs to TabWidget
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.tab_microscope,"Microscope")
        self.tabs.addTab(self.tab_sample,"Sample")
        self.tabs.addTab(self.tab_user,"User")
        self.tabs.addTab(self.tab_processing,"Processing")
        self.tabs.addTab(self.tab_calibration,"Calibration")
        self.tabs.addTab(self.tab_comments,"Comments")

        # Excute
        self.pushButton_Save = QtWidgets.QPushButton("Save")
        self.pushButton_Cancel = QtWidgets.QPushButton("Cancel")

        layout_Execute = QtWidgets.QHBoxLayout()
        layout_Execute.addWidget(self.pushButton_Cancel)
        layout_Execute.addWidget(self.pushButton_Save)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addLayout(layout_Execute)

        self.setLayout(layout)
        #self.setFixedWidth(260)
        #self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))

    @staticmethod
    def make_tab(metadata_dict):
        tab_layout = QtWidgets.QVBoxLayout()
        for key,value in metadata_dict.items():
            current_row = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(key)
            try:
                text = value.decode('utf-8')
            except AttributeError:
                text = str(value)
            lineedit = QtWidgets.QLineEdit(text)
            lineedit.setFixedWidth(180)
            current_row.addWidget(label,0,QtCore.Qt.AlignRight)
            current_row.addWidget(lineedit,0,QtCore.Qt.AlignRight)
            tab_layout.addLayout(current_row)
        tab = QtWidgets.QWidget()
        tab.setLayout(tab_layout)
        return tab

class SectionLabel(QtWidgets.QWidget):
    def __init__(self,section_title):
        QtWidgets.QWidget.__init__(self)

        line_left = QtWidgets.QFrame()
        line_left.setFrameShape(QtWidgets.QFrame.HLine)
        line_left.setFrameShadow(QtWidgets.QFrame.Sunken)
        line_left.setLineWidth(1)
        line_right = QtWidgets.QFrame()
        line_right.setFrameShape(QtWidgets.QFrame.HLine)
        line_right.setFrameShadow(QtWidgets.QFrame.Sunken)
        line_right.setLineWidth(1)

        label = QtWidgets.QLabel(section_title)
        label.setFont(sectionFont)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(line_left)
        layout.addWidget(label,0,QtCore.Qt.AlignCenter)
        layout.addWidget(line_right)

        self.setLayout(layout)

class HideableWidget(QtWidgets.QWidget):

    def __init__(self, title, widget, initial_state=True):
        """
        Makes a widget with a bar at the top with the title and a checkbox controlling the
        widget's visibility.
        Accepts:
            title:  str
            widget: QWidget object
            initial_state: bool, indicating if the widget is visible or not on loading
        """
        QtWidgets.QWidget.__init__(self)
        self.widget = widget

        # Checkbox controlling whether widget is hidden
        self.checkBox_ToggleHiding = QtWidgets.QCheckBox()
        # Title
        self.label_Title = QtWidgets.QLabel(title)
        self.label_Title.setFont(titleFont)

        title_layout = QtWidgets.QHBoxLayout()
        title_layout.addWidget(self.checkBox_ToggleHiding,0,QtCore.Qt.AlignLeft)
        title_layout.addWidget(self.label_Title,1,QtCore.Qt.AlignLeft)
        #title_layout.setSpacing(0)
        title_layout.setContentsMargins(0,0,0,0)
        title_frame = QtWidgets.QFrame()
        title_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        title_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        title_frame.setLineWidth(1)
        title_frame.setFixedWidth(control_panel_width)
        title_frame.setLayout(title_layout)

		# Change hide/show checkboxes to triangles
        title_frame.setStyleSheet(
			"QCheckBox::indicator {width:14;height: 14px;}"
			"QCheckBox::indicator:checked { image:url(./gui/icons/arrow_open.png)}"
			"QCheckBox::indicator:unchecked { image:url(./gui/icons/arrow_closed.png)}"
			)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title_frame,0)
        layout.addWidget(self.widget,1,QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(10,4,4,0)
        self.setLayout(layout)

        # Connect checkbox to toggling visibility
        self.checkBox_ToggleHiding.stateChanged.connect(widget.setVisible)

        self.checkBox_ToggleHiding.setChecked(initial_state)
        self.widget.setVisible(initial_state)

#def set_stylesheet(widget):
#    widget.setStyleSheet(
#			"QLayout::setContentsMargins(0,0,0,0)"
#            "QLayout::setSpacing(0)"
#            )


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    controlPanel = ControlPanel()
    controlPanel.show()

    app.exec_()





#app = QtWidgets.QApplication(sys.argv)
#controlPanel = ControlPanel()
#controlPanel.show()
#sys.exit(app.exec_())


