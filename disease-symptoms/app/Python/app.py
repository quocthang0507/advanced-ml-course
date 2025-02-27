import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QCheckBox, QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox
import joblib
from pathlib import Path

class DiseasePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.model, self.feature_names, self.class_mapping = self.load_model_and_mapping(
            Path(__file__).parent / 'models/trained/random_forest_model_dt1.pkl',
            Path(__file__).parent / 'models/trained/random_forest_class_mapping_dt1.csv'
        )
        self.symptoms = [
            'Sốt', 'Ho', 'Mệt mỏi', 'Khó thở', 'Tuổi', 'Giới tính', 'Huyết áp', 'Mức cholesterol'
        ]
        self.initUI()

    def load_model_and_mapping(self, model_filepath, mapping_filepath):
        model_data = joblib.load(model_filepath)
        model = model_data['model']
        feature_names = model_data.get('feature_names', None)
        class_mapping = self.load_class_mapping(mapping_filepath)
        return model, feature_names, class_mapping

    def load_class_mapping(self, filepath):
        class_mapping = {}
        with open(filepath, 'r') as f:
            for line in f:
                index, class_name = line.strip().split(',')
                class_mapping[int(index)] = class_name
        return class_mapping

    def initUI(self):
        self.layout = QVBoxLayout()

        self.checkboxes = {}
        for symptom in self.symptoms:
            if symptom in ['Tuổi', 'Giới tính', 'Huyết áp', 'Mức cholesterol']:
                if symptom == 'Giới tính':
                    self.checkboxes[symptom] = QComboBox()
                    self.checkboxes[symptom].addItems(['Nam', 'Nữ'])
                elif symptom == 'Huyết áp':
                    self.checkboxes[symptom] = QComboBox()
                    self.checkboxes[symptom].addItems(['Thấp', 'Bình thường', 'Cao'])
                elif symptom == 'Mức cholesterol':
                    self.checkboxes[symptom] = QComboBox()
                    self.checkboxes[symptom].addItems(['Thấp', 'Bình thường', 'Cao'])
                elif symptom == 'Tuổi':
                    self.checkboxes[symptom] = QSpinBox()
                    self.checkboxes[symptom].setRange(0, 120)  # Set the range for age
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
            elif isinstance(widget, QSpinBox):
                input_data[symptom] = widget.value()

        input_df = pd.DataFrame([input_data], columns=self.feature_names)
        prediction = self.model.predict(input_df)

        # Map prediction to disease name using the loaded class mapping
        disease_name = self.class_mapping.get(prediction[0], 'Không xác định')

        self.result_label.setText(f'Bệnh dự đoán: {disease_name}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DiseasePredictor()
    sys.exit(app.exec_())