import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QCheckBox, QPushButton, QLabel, QLineEdit, QComboBox
import joblib

class DiseasePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.model = joblib.load('E:/GitHub/advanced-ml-course/disease-symptoms/app/Python/models/trained/random_forest_model_dt1.pkl')
        self.symptoms = [
            'Sốt', 'Ho', 'Mệt mỏi', 'Khó thở', 'Tuổi', 'Giới tính', 'Huyết áp', 'Mức cholesterol'
        ]
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.checkboxes = {}
        for symptom in self.symptoms:
            if symptom in ['Tuổi', 'Giới tính', 'Huyết áp', 'Mức cholesterol']:
                if symptom == 'Giới tính':
                    self.checkboxes[symptom] = QComboBox()
                    self.checkboxes[symptom].addItems(['Nam', 'Nữ'])
                else:
                    self.checkboxes[symptom] = QLineEdit()
                self.layout.addWidget(QLabel(symptom))
                self.layout.addWidget(self.checkboxes[symptom])
            else:
                checkbox = QCheckBox(symptom)
                self.checkboxes[symptom] = checkbox
                self.layout.addWidget(checkbox)

        self.predict_button = QPushButton('Dự đoán bệnh')
        self.predict_button.clicked.connect(self.predict_disease)
        self.layout.addWidget(self.predict_button)

        self.result_label = QLabel('')
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)
        self.setWindowTitle('Dự đoán bệnh')
        self.show()

    def predict_disease(self):
        input_data = {}
        for symptom, widget in self.checkboxes.items():
            if isinstance(widget, QCheckBox):
                input_data[symptom] = 1 if widget.isChecked() else 0
            elif isinstance(widget, QLineEdit):
                input_data[symptom] = widget.text()
            elif isinstance(widget, QComboBox):
                input_data[symptom] = widget.currentText()

        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)
        self.result_label.setText(f'Bệnh dự đoán: {prediction[0]}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DiseasePredictor()
    sys.exit(app.exec_())