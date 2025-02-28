import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QCheckBox, QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox
import joblib
from pathlib import Path

class DiseasePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.model, self.feature_names, self.class_mapping = self.load_model_and_mapping(
            Path(__file__).parent / 'models/trained/random_forest_model_dt2.pkl',
            Path(__file__).parent / 'models/trained/random_forest_class_mapping_dt2.csv'
        )
        self.symptoms = [
            'Ngứa', 'Phát ban da', 'Nổi mụn trên da', 'Hắt hơi liên tục', 'Rùng mình', 'Ớn lạnh', 'Đau khớp', 'Đau bụng', 
            'Axit', 'Loét lưỡi', 'Teo cơ', 'Nôn mửa', 'Đau khi tiểu', 'Tiểu ra máu', 'Mệt mỏi', 'Tăng cân', 'Lo lắng', 
            'Lạnh tay chân', 'Thay đổi tâm trạng', 'Giảm cân', 'Bồn chồn', 'Lờ đờ', 'Đốm trong cổ họng', 'Mức đường không đều', 
            'Ho', 'Sốt cao', 'Mắt trũng', 'Khó thở', 'Đổ mồ hôi', 'Mất nước', 'Khó tiêu', 'Đau đầu', 'Vàng da', 'Nước tiểu sẫm màu', 
            'Buồn nôn', 'Mất cảm giác thèm ăn', 'Đau sau mắt', 'Đau lưng', 'Táo bón', 'Đau bụng', 'Tiêu chảy', 'Sốt nhẹ', 
            'Nước tiểu vàng', 'Vàng mắt', 'Suy gan cấp tính', 'Tích nước', 'Sưng bụng', 'Sưng hạch bạch huyết', 'Mệt mỏi', 
            'Mờ mắt', 'Đờm', 'Kích ứng cổ họng', 'Đỏ mắt', 'Áp lực xoang', 'Chảy nước mũi', 'Nghẹt mũi', 'Đau ngực', 
            'Yếu chân tay', 'Nhịp tim nhanh', 'Đau khi đi tiêu', 'Đau vùng hậu môn', 'Phân có máu', 'Kích ứng hậu môn', 
            'Đau cổ', 'Chóng mặt', 'Chuột rút', 'Bầm tím', 'Béo phì', 'Sưng chân', 'Sưng mạch máu', 'Mặt và mắt sưng', 
            'Tuyến giáp to', 'Móng tay giòn', 'Sưng tứ chi', 'Đói quá mức', 'Quan hệ ngoài hôn nhân', 'Khô và ngứa môi', 
            'Nói lắp', 'Đau đầu gối', 'Đau khớp hông', 'Yếu cơ', 'Cứng cổ', 'Sưng khớp', 'Cứng khớp', 'Chuyển động quay', 
            'Mất thăng bằng', 'Không ổn định', 'Yếu một bên cơ thể', 'Mất khứu giác', 'Khó chịu bàng quang', 'Nước tiểu có mùi hôi', 
            'Cảm giác tiểu liên tục', 'Đi qua khí', 'Ngứa bên trong', 'Nhìn độc hại (typhos)', 'Trầm cảm', 'Cáu gắt', 'Đau cơ', 
            'Thay đổi cảm giác', 'Đốm đỏ trên cơ thể', 'Đau bụng', 'Kinh nguyệt không đều', 'Đốm màu', 'Chảy nước mắt', 
            'Tăng cảm giác thèm ăn', 'Tiểu nhiều', 'Tiền sử gia đình', 'Đờm nhầy', 'Đờm gỉ sắt', 'Thiếu tập trung', 'Rối loạn thị giác', 
            'Nhận truyền máu', 'Nhận tiêm không vô trùng', 'Hôn mê', 'Chảy máu dạ dày', 'Bụng phình', 'Tiền sử uống rượu', 
            'Tích nước', 'Máu trong đờm', 'Tĩnh mạch nổi trên bắp chân', 'Đánh trống ngực', 'Đau khi đi bộ', 'Mụn mủ', 
            'Mụn đầu đen', 'Sẹo', 'Bong tróc da', 'Bụi như bạc', 'Vết lõm nhỏ trên móng tay', 'Móng tay viêm', 'Phồng rộp', 
            'Vết loét đỏ quanh mũi', 'Vết mủ vàng', 'Tiên lượng'
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
        self.layout = QGridLayout()

        self.checkboxes = {}
        row = 0
        col = 0
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
                self.layout.addWidget(QLabel(symptom), row, col)
                self.layout.addWidget(self.checkboxes[symptom], row, col + 1)
            else:
                checkbox = QCheckBox(symptom)
                self.checkboxes[symptom] = checkbox
                self.layout.addWidget(checkbox, row, col)
            col += 2
            if col >= 4:
                col = 0
                row += 1

        self.predict_button = QPushButton('Dự đoán bệnh')
        self.predict_button.clicked.connect(self.predict_disease)
        self.layout.addWidget(self.predict_button, row + 1, 0, 1, 4)

        self.result_label = QLabel('')
        self.layout.addWidget(self.result_label, row + 2, 0, 1, 4)

        self.setLayout(self.layout)
        self.setWindowTitle('Dự đoán bệnh')
        self.showMaximized()

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
        prediction_prob = self.model.predict_proba(input_df)[0]

        # Get the top 5 disease probabilities
        top_5_indices = prediction_prob.argsort()[-5:][::-1]
        top_5_diseases = [(self.class_mapping[i], prediction_prob[i]) for i in top_5_indices]

        result_text = 'Top 5 bệnh dự đoán:\n'
        for disease, prob in top_5_diseases:
            result_text += f'{disease}: {prob:.2f}\n'

        self.result_label.setText(result_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DiseasePredictor()
    sys.exit(app.exec_())