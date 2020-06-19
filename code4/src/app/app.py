# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
import os
from svm import main
from model import Model


class Ui_Dialog(object):
    def __init__(self):
        super().__init__()
        self.model = Model()

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1006, 837)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        Dialog.setFont(font)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(6, 39, 991, 791))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_training = QtWidgets.QWidget()
        self.tab_training.setObjectName("tab_training")
        self.frame = QtWidgets.QFrame(self.tab_training)
        self.frame.setGeometry(QtCore.QRect(10, 10, 961, 771))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.layoutWidget = QtWidgets.QWidget(self.frame)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 40, 491, 29))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.ip_test_path = QtWidgets.QLineEdit(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        self.ip_test_path.setFont(font)
        self.ip_test_path.setObjectName("ip_test_path")
        self.horizontalLayout_4.addWidget(self.ip_test_path)
        self.btn_import_test = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        self.btn_import_test.setFont(font)
        self.btn_import_test.setObjectName("btn_import_test")
        self.horizontalLayout_4.addWidget(self.btn_import_test)
        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 3)
        self.horizontalLayout_4.setStretch(2, 1)
        self.layoutWidget_5 = QtWidgets.QWidget(self.frame)
        self.layoutWidget_5.setGeometry(QtCore.QRect(10, 640, 391, 29))
        self.layoutWidget_5.setObjectName("layoutWidget_5")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.layoutWidget_5)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_9 = QtWidgets.QLabel(self.layoutWidget_5)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_9.addWidget(self.label_9)
        self.ip_model_name = QtWidgets.QLineEdit(self.layoutWidget_5)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        self.ip_model_name.setFont(font)
        self.ip_model_name.setObjectName("ip_model_name")
        self.horizontalLayout_9.addWidget(self.ip_model_name)
        self.horizontalLayout_9.setStretch(0, 1)
        self.horizontalLayout_9.setStretch(1, 3)
        self.btn_training = QtWidgets.QPushButton(self.frame)
        self.btn_training.setGeometry(QtCore.QRect(110, 680, 291, 31))
        self.btn_training.setObjectName("btn_training")
        self.layoutWidget1 = QtWidgets.QWidget(self.frame)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 10, 491, 29))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.ip_train_path = QtWidgets.QLineEdit(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        self.ip_train_path.setFont(font)
        self.ip_train_path.setObjectName("ip_train_path")
        self.horizontalLayout.addWidget(self.ip_train_path)
        self.btn_import_train = QtWidgets.QPushButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        self.btn_import_train.setFont(font)
        self.btn_import_train.setObjectName("btn_import_train")
        self.horizontalLayout.addWidget(self.btn_import_train)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout.setStretch(2, 1)
        self.layoutWidget2 = QtWidgets.QWidget(self.frame)
        self.layoutWidget2.setGeometry(QtCore.QRect(10, 76, 391, 31))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_5 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        self.cb_algorithm = QtWidgets.QComboBox(self.layoutWidget2)
        self.cb_algorithm.setObjectName("cb_algorithm")
        self.cb_algorithm.addItem("")
        self.cb_algorithm.addItem("")
        self.horizontalLayout_5.addWidget(self.cb_algorithm)
        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 3)
        self.layoutWidget3 = QtWidgets.QWidget(self.frame)
        self.layoutWidget3.setGeometry(QtCore.QRect(10, 120, 391, 511))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.layoutWidget3)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_10 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_10.addWidget(self.label_10)
        self.txt_config = QtWidgets.QTextBrowser(self.layoutWidget3)
        self.txt_config.setObjectName("txt_config")
        self.horizontalLayout_10.addWidget(self.txt_config)
        self.horizontalLayout_10.setStretch(0, 1)
        self.horizontalLayout_10.setStretch(1, 3)
        self.lb_percent = QtWidgets.QLabel(self.frame)
        self.lb_percent.setGeometry(QtCore.QRect(580, 220, 251, 191))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.lb_percent.setFont(font)
        self.lb_percent.setText("")
        self.lb_percent.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_percent.setObjectName("lb_percent")
        self.tabWidget.addTab(self.tab_training, "")
        self.tab_demo = QtWidgets.QWidget()
        self.tab_demo.setObjectName("tab_demo")
        self.tabWidget.addTab(self.tab_demo, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_4.setText(_translate("Dialog", "File Test"))
        self.btn_import_test.setText(_translate("Dialog", "Browse"))
        self.label_9.setText(_translate("Dialog", "Model Name"))
        self.btn_training.setText(_translate("Dialog", "Training"))
        self.label.setText(_translate("Dialog", "File Training"))
        self.btn_import_train.setText(_translate("Dialog", "Browse"))
        self.label_5.setText(_translate("Dialog", "Algorithm"))
        self.cb_algorithm.setItemText(0, _translate("Dialog", "Naive Bayes"))
        self.cb_algorithm.setItemText(1, _translate("Dialog", "SVM"))
        self.label_10.setText(_translate("Dialog", "Config"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_training), _translate("Dialog", "Training"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_demo), _translate("Dialog", "Demo"))
    
        # Config
        self.btn_import_train.clicked.connect(self.import_file_train)
        self.btn_import_test.clicked.connect(self.import_file_test)
        self.cb_algorithm.currentIndexChanged.connect(self.get_file_config)
        self.btn_training.clicked.connect(self.training)

        # Open file config
        self.get_file_config(self.cb_algorithm.currentIndex())

    def training(self):
        setting_path='src/app/settings.py'
        self.model.writeDoc(self.txt_config.toPlainText())
        self.model.setFileName(setting_path)
        text='TRAIN_PATH="'+self.ip_train_path.text() + '"\n'
        text+='TEST_PATH="'+self.ip_test_path.text() + '"'
        self.model.writeDoc(text)

        score = main(self.cb_algorithm.currentIndex(), self.ip_model_name.text())

        self.lb_percent.setText('' + str(round(score, 2) ) + ' %')

    def import_file_train(self):
        self.open_dialog_box('train')

    def import_file_test(self):
        self.open_dialog_box('test')

    def open_dialog_box(self, name):
        filename = QFileDialog.getOpenFileName(filter="Excel (*.xls *.xlsx)")
        path = filename[0]

        if name == 'train':
            self.ip_train_path.setText(path)
        elif name == 'test':
            self.ip_test_path.setText(path)

    def get_file_config(self, value):
        file_config_svm = 'src/app/config_svm.py'
        file_config_naive_bayes = 'src/app/config_naive_bayes.py'
        
        if value == 0:
            if self.model.isValid(file_config_svm):
                self.model.setFileName(file_config_svm)
                self.txt_config.setText(self.model.getFileContents())
        elif value == 1:
            if self.model.isValid(file_config_naive_bayes):
                self.model.setFileName(file_config_naive_bayes)
                self.txt_config.setText(self.model.getFileContents())

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
