from windows import ocr_ui
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import os

if __name__ == '__main__':
    if 'images' not in os.listdir('.'):
        os.mkdir('images')
    if 'output' not in os.listdir('.'):
        os.mkdir('./output')
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    ui = ocr_ui.Ui_OCR()
    flag = ui.start()
    if flag:
        ui.setupUi(main_window)
        main_window.show()
        sys.exit(app.exec_())