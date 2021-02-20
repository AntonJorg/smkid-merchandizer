from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit
from PyQt5 import uic

from keypoints import calculate_keypoints, annotate_frame
from geometry import *
from parse import *


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.merchstatus = 'Tørstig?'

        self.torsoim = cv2.imread('hoodie/black.png')

        self.forearm_L = cv2.imread('hoodie/black.png')#forearm_L.png')
        self.overarm_L = cv2.imread('hoodie/black.png')#overarm_L.png')

        self.forearm_R = cv2.imread('hoodie/black.png')#forearm_R.png')
        self.overarm_R = cv2.imread('hoodie/black.png')#overarm_R.png')

        self.logoim = cv2.imread('hoodie/logo_smkidsort.png')

        

    def run(self):

        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            coords = parse(calculate_keypoints(cv_img, blob_size=183))

            if self.merchstatus == "Hoodie":

                # Neck: 0, Right Shoulder: 1, Right Elbow: 2, Right Wrist: 3, Left Shoulder: 4, Left Elbow: 5, Left Wrist: 6,
                # Right Hip: 7, Left Hip: 8, Chest: 9

                torso_corners = torso(coords[9, :], coords[4, :], coords[1, :], coords[7, :], coords[8, :])

                linefit, meanx = body_line(coords[0, :], coords[9, :], coords[4, :], coords[1, :], coords[8, :],
                                           coords[7, :])

                loverarm = arm_box(coords[4, :], coords[5, :])

                roverarm = arm_box(coords[1, :], coords[2, :])

                lforearm = arm_box(coords[5, :], coords[6, :])

                rforearm = arm_box(coords[2, :], coords[3, :])

                logoc = logo(linefit, meanx)

                lelbow = connect_polygon(loverarm, lforearm)

                relbow = connect_polygon(roverarm, rforearm)

                cv_img = imtransform(cv_img, self.torsoim, torso_corners)

                cv_img = imtransform(cv_img, self.forearm_L, lforearm)

                cv_img = imtransform(cv_img, self.overarm_L, loverarm)

                cv_img = imtransform(cv_img, self.forearm_R, rforearm)
                cv_img = imtransform(cv_img, self.overarm_R, roverarm)
                cv_img = imtransform(cv_img, self.logoim, logoc)


                try:


                    cv_img = imtransform(cv_img, self.overarm_R, lelbow)
                    cv_img = imtransform(cv_img, self.overarm_R, relbow)

                except:
                    print("Bruh!")

            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("uiV1.ui", self)
        self.disply_width = 700
        self.display_height = 700
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.move(150,40)

        # create a text label
        self.textLabel = QLabel('Webcam')

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.goButton.clicked.connect(self.pressed)
        # start the thread
        self.thread.start()

    def pressed(self):
        self.thread.merchstatus = self.merchSelect.currentText()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
