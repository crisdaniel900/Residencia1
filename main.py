import os
import sys
import threading

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from PyQt5.QtCore import QObject, QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton, QPlainTextEdit,
    QVBoxLayout, QWidget,
)
from keras.models import load_model

from capture_samples import capture_samples
from constants import (
    FRAME_ACTIONS_PATH, KEYPOINTS_PATH, MIN_LENGTH_FRAMES, MODEL_FRAMES, MODEL_PATH,
)
from constants import ROOT_PATH, WORDS_JSON_PATH, words_text
from create_keypoints import create_keypoints
from evaluate_model import normalize_keypoints
from helpers import (
    draw_keypoints, extract_keypoints, get_available_word_ids, get_word_ids,
    mediapipe_detection, there_hand,
)
from training_model import training_model


class ActionWorker(QObject):
    frame_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, action, **kwargs):
        super().__init__()
        self.action = action
        self.kwargs = kwargs
        self.stop_event = threading.Event()

    def run(self):
        try:
            if self.action == 'capture':
                capture_samples(
                    self.kwargs['path'],
                    camera_index=self.kwargs['camera_index'],
                    stop_event=self.stop_event,
                    frame_callback=self._capture_frame,
                    status_callback=self.status_changed.emit,
                )
                self.finished.emit('Captura finalizada.')
            else:
                self.status_changed.emit('Entrenando modelo...')
                word_ids = get_available_word_ids(self.kwargs['data_path'])
                if not word_ids:
                    raise FileNotFoundError(
                        f'No hay carpetas de palabras con muestras en {self.kwargs["data_path"]}.'
                    )
                for word_id in word_ids:
                    frames_path = os.path.join(self.kwargs['data_path'], word_id)
                    hdf_path = os.path.join(self.kwargs['keypoints_path'], f'{word_id}.h5')
                    create_keypoints(word_id, self.kwargs['data_path'], hdf_path)
                training_model(
                    MODEL_PATH,
                    epochs=self.kwargs['epochs'],
                    keypoints_path=self.kwargs['keypoints_path'],
                    word_ids=word_ids,
                )
                self.finished.emit('Entrenamiento finalizado.')
        except Exception as error:
            self.failed.emit(str(error))

    def _capture_frame(self, image, results):
        draw_keypoints(image, results)
        self.frame_ready.emit(image.copy())


class LsmLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LSM - Lanzador')
        self.resize(860, 600)
        self.capture = None
        self.holistic_model = None
        self.model = None
        self.kp_seq = []
        self.sentence = []
        self.count_frame = 0
        self.fix_frames = 0
        self.recording = False
        self.last_detected_word = None
        self.detection_cooldown = 0
        self.worker_thread = None
        self.worker = None
        # Preview attributes
        self.preview_capture = None
        self.preview_timer = None
        self.preview_enabled = False
        self.data_path = FRAME_ACTIONS_PATH
        self.keypoints_path = KEYPOINTS_PATH
        self._build_ui()
        self.refresh_cameras()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        title = QLabel('Prototipo Interfaz')
        title.setStyleSheet('font-size: 24px; font-weight: 700;')
        layout.addWidget(title)
        subtitle = QLabel('Lanza la captura, el entrenamiento o el reconocimiento de palabras LSM desde una sola ventana.')
        layout.addWidget(subtitle)

        camera_box = QGroupBox('Cámara')
        camera_layout = QHBoxLayout(camera_box)
        camera_layout.addWidget(QLabel('Dispositivo:'))
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(165)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selected)
        camera_layout.addWidget(self.camera_combo)
        self.scan_button = QPushButton('Buscar cámaras')
        self.scan_button.clicked.connect(self.refresh_cameras)
        camera_layout.addWidget(self.scan_button)
        self.camera_count = QLabel()
        camera_layout.addWidget(self.camera_count)
        # Agregar checkbox para preview
        self.preview_checkbox = QCheckBox('Mostrar preview')
        self.preview_checkbox.stateChanged.connect(self.on_preview_toggled)
        camera_layout.addWidget(self.preview_checkbox)
        camera_layout.addStretch()
        layout.addWidget(camera_box)

        actions_box = QGroupBox('Acciones')
        actions_layout = QVBoxLayout(actions_box)
        buttons_layout = QHBoxLayout()
        self.capture_button = QPushButton('Capturar secuencias')
        self.capture_button.clicked.connect(self.start_capture)
        self.train_button = QPushButton('Entrenar modelo LSTM')
        self.train_button.clicked.connect(self.start_training)
        self.live_button = QPushButton('Reconocer en vivo')
        self.live_button.clicked.connect(self.toggle_live)
        for button in (self.capture_button, self.train_button, self.live_button):
            button.setMinimumHeight(40)
            buttons_layout.addWidget(button)
        actions_layout.addLayout(buttons_layout)
        utility_layout = QHBoxLayout()
        self.stop_button = QPushButton('Detener proceso activo')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_process)
        utility_layout.addWidget(self.stop_button)
        self.open_data_button = QPushButton('Abrir carpeta MP_Data')
        self.open_data_button.clicked.connect(self.open_data_folder)
        utility_layout.addWidget(self.open_data_button)
        self.select_data_button = QPushButton('Seleccionar carpeta MP_Data')
        self.select_data_button.clicked.connect(self.select_data_folder)
        utility_layout.addWidget(self.select_data_button)
        self.open_keypoints_button = QPushButton('Abrir carpeta KeyPoints')
        self.open_keypoints_button.clicked.connect(self.open_keypoints_folder)
        utility_layout.addWidget(self.open_keypoints_button)
        self.select_keypoints_button = QPushButton('Seleccionar carpeta KeyPoints')
        self.select_keypoints_button.clicked.connect(self.select_keypoints_folder)
        utility_layout.addWidget(self.select_keypoints_button)
        utility_layout.addStretch()
        actions_layout.addLayout(utility_layout)
        self.data_path_label = QLabel(f'Datos: {self.data_path}')
        actions_layout.addWidget(self.data_path_label)
        self.keypoints_path_label = QLabel(f'KeyPoints: {self.keypoints_path}')
        actions_layout.addWidget(self.keypoints_path_label)
        self.status_label = QLabel('Listo.')
        actions_layout.addWidget(self.status_label)
        layout.addWidget(actions_box)

        state_box = QGroupBox('Estado')
        state_layout = QVBoxLayout(state_box)
        current_layout = QHBoxLayout()
        current_layout.addWidget(QLabel('Tarea actual:'))
        self.task_value = QLabel('Ninguna')
        current_layout.addWidget(self.task_value)
        current_layout.addStretch()
        state_layout.addLayout(current_layout)
        project_layout = QHBoxLayout()
        project_layout.addWidget(QLabel('Proyecto:'))
        project_layout.addWidget(QLabel(ROOT_PATH))
        project_layout.addStretch()
        state_layout.addLayout(project_layout)
        layout.addWidget(state_box)

        output_box = QGroupBox('Salida')
        output_layout = QVBoxLayout(output_box)
        self.detected_label = QLabel('Detectado: ---')
        self.detected_label.setAlignment(Qt.AlignCenter)
        self.detected_label.setStyleSheet(
            'font-size: 28px; font-weight: 700; color: #1769aa; padding: 10px;'
        )
        output_layout.addWidget(self.detected_label)
        self.video_label = QLabel('Vista de cámara inactiva')
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(120)
        self.video_label.setStyleSheet('background: #20252b; color: #c7ced6;')
        output_layout.addWidget(self.video_label)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlainText('Listo. Elige una acción para empezar.')
        output_layout.addWidget(self.output)
        layout.addWidget(output_box, 1)

    def on_camera_selected(self):
        """Se ejecuta cuando cambia la cámara seleccionada"""
        if self.preview_checkbox.isChecked():
            self.stop_preview()
            self.start_preview()

    def on_preview_toggled(self, state):
        """Se ejecuta cuando se activa/desactiva el checkbox de preview"""
        if state == Qt.Checked:
            self.start_preview()
        else:
            self.stop_preview()

    def start_preview(self):
        """Inicia el preview de la cámara seleccionada"""
        if self.preview_capture is not None:
            return
        
        camera_index = self.selected_camera()
        self.preview_capture = cv2.VideoCapture(camera_index)
        
        if not self.preview_capture.isOpened():
            QMessageBox.warning(self, 'Error', 'No se pudo abrir la cámara.')
            self.preview_capture = None
            return
        
        self.preview_enabled = True
        self.video_label.setText('')
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_preview_frame)
        self.preview_timer.start(30)
        self.set_status('Mostrando preview de cámara...')

    def update_preview_frame(self):
        """Actualiza el frame del preview"""
        if self.preview_capture is None:
            return
        
        ret, frame = self.preview_capture.read()
        if not ret:
            return
        
        # Convertir BGR a RGB para mostrar correctamente
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        q_image = QImage(image.data, width, height, image.strides[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def stop_preview(self):
        """Detiene el preview de la cámara"""
        if self.preview_capture is None:
            return
        
        self.preview_enabled = False
        if self.preview_timer:
            self.preview_timer.stop()
            self.preview_timer = None
        
        self.preview_capture.release()
        self.preview_capture = None
        self.video_label.setText('Vista de cámara inactiva')
        self.video_label.setStyleSheet('background: #20252b; color: #c7ced6;')
        self.set_status('Preview detenido.')

    def refresh_cameras(self):
        self.stop_preview()
        self.preview_checkbox.setChecked(False)
        self.camera_combo.clear()
        found = 0
        for camera_index in range(5):
            camera = cv2.VideoCapture(camera_index)
            if camera.isOpened():
                # Obtener información de la cámara
                width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(camera.get(cv2.CAP_PROP_FPS))
                
                # Crear etiqueta descriptiva
                label = f'Cámara {camera_index + 1} - {width}x{height} @ {fps if fps else "30"} FPS'
                self.camera_combo.addItem(label, camera_index)
                found += 1
            camera.release()
        self.camera_count.setText(f'{found} encontrada(s)')

    def selected_camera(self):
        return self.camera_combo.currentData() if self.camera_combo.count() else 0

    def start_capture(self):
        if self.worker_thread:
            return
        # Detener preview cuando inicia captura
        self.stop_preview()
        self.preview_checkbox.setEnabled(False)
        self.camera_combo.setEnabled(False)
        
        word, accepted = QInputDialog.getText(self, 'Nueva secuencia', 'Palabra o frase:')
        if not accepted or not word.strip():
            self.preview_checkbox.setEnabled(True)
            self.camera_combo.setEnabled(True)
            return
        word_id = word.strip().lower().replace(' ', '_')
        path = os.path.join(self.data_path, word_id)
        self.start_worker('capture', path=path, camera_index=self.selected_camera())
        self.output.appendPlainText(f'Capturando muestras para: {word_id}')

    def start_training(self):
        if self.worker_thread:
            return
        os.makedirs(self.keypoints_path, exist_ok=True)
        self.start_worker(
            'training',
            epochs=500,
            data_path=self.data_path,
            keypoints_path=self.keypoints_path,
        )

    def start_worker(self, action, **kwargs):
        self.worker_thread = QThread(self)
        self.worker = ActionWorker(action, **kwargs)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.status_changed.connect(self.set_status)
        self.worker.finished.connect(self.worker_finished)
        self.worker.failed.connect(self.worker_failed)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.start()
        self.set_busy(True, 'Captura' if action == 'capture' else 'Entrenamiento')

    def worker_finished(self, message):
        self.set_status(message)
        self.output.appendPlainText(message)
        self.finish_worker()

    def worker_failed(self, message):
        self.set_status('Error.')
        self.output.appendPlainText(f'Error: {message}')
        QMessageBox.critical(self, 'Error', message)
        self.finish_worker()

    def finish_worker(self):
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.worker_thread = None
        self.worker = None
        self.set_busy(False, 'Ninguna')
        # Reactivar controles de cámara
        self.preview_checkbox.setEnabled(True)
        self.camera_combo.setEnabled(True)
        # Reiniciar preview si estaba activado
        if self.preview_checkbox.isChecked():
            self.start_preview()

    def stop_process(self):
        if self.worker:
            self.worker.stop_event.set()
        self.stop_live()
        self.set_status('Proceso detenido.')

    def set_busy(self, busy, task):
        self.task_value.setText(task)
        self.stop_button.setEnabled(busy or self.capture is not None)
        self.capture_button.setEnabled(not busy)
        self.train_button.setEnabled(not busy)
        self.live_button.setEnabled(not busy or self.capture is not None)

    def set_status(self, status):
        self.status_label.setText(status)

    def toggle_live(self):
        if self.capture is None:
            self.start_live()
        else:
            self.stop_live()

    def start_live(self):
        if not os.path.exists(MODEL_PATH):
            QMessageBox.warning(self, 'Modelo no encontrado', f'Entrena primero el modelo en {MODEL_PATH}.')
            return
        self.capture = cv2.VideoCapture(self.selected_camera())
        self.holistic_model = Holistic()
        self.model = load_model(MODEL_PATH)
        self.kp_seq, self.sentence = [], []
        self.count_frame = self.fix_frames = 0
        self.recording = False
        self.last_detected_word = None
        self.detection_cooldown = 0
        self.live_button.setText('Detener reconocimiento')
        self.set_busy(True, 'Reconocimiento en vivo')
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_live_frame)
        self.timer.start(30)

    def update_live_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mediapipe_detection(frame, self.holistic_model)
        if there_hand(results):
            self.count_frame += 1
            self.kp_seq.append(extract_keypoints(results))
            if self.detection_cooldown > 0:
                self.detection_cooldown -= 1

            if len(self.kp_seq) >= MODEL_FRAMES and self.detection_cooldown == 0:
                normalized = normalize_keypoints(self.kp_seq[-MODEL_FRAMES:], int(MODEL_FRAMES))
                result = self.model.predict(np.expand_dims(normalized, axis=0), verbose=0)[0]
                predicted_index = int(np.argmax(result))
                confidence = float(result[predicted_index])
                word_ids = get_available_word_ids(self.data_path)
                if predicted_index < len(word_ids):
                    word_id = word_ids[predicted_index].split('-')[0]
                    sentence = words_text.get(word_id, word_id.upper())
                    self.detected_label.setText(
                        f'Predicción: {sentence} ({confidence * 100:.1f}%)'
                    )
                    if confidence >= 0.5 and word_id != self.last_detected_word:
                        self.last_detected_word = word_id
                        self.sentence.insert(0, sentence)
                        self.output.appendPlainText(sentence)
                        self.detection_cooldown = MODEL_FRAMES
                        self.kp_seq = []
        else:
            self.kp_seq = []
            self.count_frame = 0
            self.detection_cooldown = 0
            self.last_detected_word = None
        self.show_frame(image, results)
        self.status_label.setText('Reconociendo...' if self.recording else 'Listo.')

    def show_frame(self, image, results=None):
        if results is not None:
            draw_keypoints(image, results)
        height, width = image.shape[:2]
        q_image = QImage(image.data, width, height, image.strides[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def stop_live(self):
        if self.capture is None:
            return
        self.timer.stop()
        self.capture.release()
        self.capture = None
        if self.holistic_model:
            self.holistic_model.close()
        self.holistic_model = self.model = None
        self.live_button.setText('Reconocer en vivo')
        if not self.worker_thread:
            self.set_busy(False, 'Ninguna')

    def open_data_folder(self):
        folder = self.data_path
        os.makedirs(folder, exist_ok=True)
        if sys.platform == 'win32':
            os.startfile(folder)
        else:
            QFileDialog.getOpenFileName(self, 'Abrir carpeta MP_Data', folder)

    def open_keypoints_folder(self):
        folder = self.keypoints_path
        os.makedirs(folder, exist_ok=True)
        if sys.platform == 'win32':
            os.startfile(folder)
        else:
            QFileDialog.getOpenFileName(self, 'Abrir carpeta KeyPoints', folder)

    def select_data_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            'Seleccionar carpeta MP_Data',
            self.data_path,
        )
        if not folder:
            return
        self.data_path = os.path.normpath(folder)
        self.data_path_label.setText(f'Datos: {self.data_path}')
        self.set_status('Carpeta MP_Data actualizada.')

    def select_keypoints_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            'Seleccionar carpeta KeyPoints',
            self.keypoints_path,
        )
        if not folder:
            return
        self.keypoints_path = os.path.normpath(folder)
        os.makedirs(self.keypoints_path, exist_ok=True)
        self.keypoints_path_label.setText(f'KeyPoints: {self.keypoints_path}')
        self.set_status('Carpeta KeyPoints actualizada.')

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop_event.set()
        self.stop_preview()
        self.stop_live()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LsmLauncher()
    window.show()
    sys.exit(app.exec_())
