import streamlit as st
import torch
import numpy as np
import cv2
import time
import serial
from PIL import Image


class deteksi_objek:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        membuat video capture
        """
        return cv2.VideoCapture(self.capture_index)

    def load_model(self):
        """
        load yolov5 model
        """
        model_path = "final.pt"
        model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=model_path, force_reload=True
        )
        return model

    def score_frame(self, frame):
        """
        menerima sebuah frame tunggal (dalam format numpy/list/tuple) sebagai masukan,
        dan menilai frame tersebut menggunakan model yolo5.
        Model tersebut dipindahkan ke perangkat yang ditentukan dan frame masukan
        diterjemahkan menjadi sebuah daftar dengan satu frame.
        Model yolo5 kemudian memproses frame dan mengeluarkan label dan koordinat dari objek yang terdeteksi di frame.
        Fungsi ini mengembalikan label dan koordinat tersebut sebagai sebuah tuple.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def get_label_percentages(self, results):
        labels, _ = results
        n = len(labels)

        label_count = {}

        for i in range(n):
            label = self.class_to_label(labels[i])

            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        label_percentages = {}
        for label, count in label_count.items():
            percentage = count / n * 100
            label_percentages[label] = percentage

        return label_percentages

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Fungsi untuk membuat kotak dan label dengan warna yang berbeda untuk setiap label.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        label_colors = {
            "benda asing": (0, 0, 255),  # Merah
            "chalky": (0, 255, 0),  # Hijau
            "gabah": (255, 0, 0),  # Biru
            "hama": (255, 255, 0),  # Kuning
            "kepala": (255, 0, 255),  # Ungu
            "ketan": (0, 255, 255),  # Aqua
            "menir": (255, 165, 0),  # Oranye
            "patah": (128, 0, 128),  # Maroon
            "sosoh": (0, 128, 128),  # Teal
            "utuh": (128, 128, 0),  # Olive
        }

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = (
                    int(row[0] * x_shape),
                    int(row[1] * y_shape),
                    int(row[2] * x_shape),
                    int(row[3] * y_shape),
                )
                label = self.class_to_label(labels[i])
                bgr = label_colors.get(label, (0, 0, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    bgr,
                    2,
                )

        return frame

    def __call__(self):
        """
        looping untuk frame video
        """
        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            assert ret

            frame = cv2.resize(frame, (640, 640))

            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            # print(f"Frames Per Second : {fps}")

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                2,
            )

            cv2.imshow("YOLOv5 Detection", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()


def main():
    st.set_page_config(page_title="Rice Detection", page_icon="🍛")

    logo = Image.open("assets/logo.png")

    st.image([logo])

    st.title("Rice Quality Detection")

    # Create an object to detect objects
    detector = deteksi_objek(capture_index=0)

    # Start the object detection process
    stframe = st.empty()
    video_capture = detector.get_video_capture()

    label_percentages = {}
    video_processing = True  # Variabel untuk mengontrol pemrosesan video

    stop_button = st.button(
        "Capture and Info Rice"
    )  # Tombol untuk menghentikan pemrosesan video

    while video_processing:
        ret, frame = video_capture.read()
        assert ret

        results = detector.score_frame(frame)
        frame = detector.plot_boxes(results, frame)

        label_percentages = detector.get_label_percentages(results)

        stframe.image(frame, channels="BGR")

        if stop_button:
            video_processing = (
                False  # Menghentikan pemrosesan video jika tombol "Stop" ditekan
            )

    total_detections = sum(label_percentages.values())

    # Calculate percentages
    asing_percentage = label_percentages.get("benda asing", 0) / total_detections * 100
    chalky_percentage = label_percentages.get("chalky", 0) / total_detections * 100
    gabah_percentage = label_percentages.get("gabah", 0) / total_detections * 100
    hama_percentage = label_percentages.get("hama", 0) / total_detections * 100
    kepala_percentage = label_percentages.get("kepala", 0) / total_detections * 100
    ketan_percentage = label_percentages.get("ketan", 0) / total_detections * 100
    menir_percentage = label_percentages.get("menir", 0) / total_detections * 100
    patah_percentage = label_percentages.get("patah", 0) / total_detections * 100
    sosoh_percentage = 100 - (
        label_percentages.get("sosoh", 0) / total_detections * 100
    )
    utuh_percentage = label_percentages.get("utuh", 0) / total_detections * 100

    # Update teks dengan persentase deteksi setelah pemrosesan video selesai
    st.text(f"Persentase Deteksi Kepala: {kepala_percentage:.2f}%")
    st.text(f"Persentase Deteksi Patah: {patah_percentage:.2f}%")
    st.text(f"Persentase Deteksi Menir: {menir_percentage:.2f}%")
    st.text(f"Persentase Deteksi Asing: {asing_percentage:.2f}%")
    st.text(f"Persentase Deteksi Putih: {ketan_percentage:.2f}%")
    st.text(f"Persentase Deteksi Utuh: {utuh_percentage:.2f}%")
    st.text(f"Persentase Deteksi Chalky: {chalky_percentage:.2f}%")
    st.text(f"Persentase Deteksi Gabah: {gabah_percentage:.2f}%")
    st.text(f"Persentase Deteksi Hama: {hama_percentage:.2f}%")
    st.text(f"Persentase Deteksi Sosoh: {sosoh_percentage:.2f}%")

    video_capture.release()

    ser = serial.Serial("/dev/cu.usbmodem1201", 9600)

    def read_data():
        data = ser.readline().decode().strip()
        return data

    st.header("Hasil Pengukuran Sensor")
    start_time = time.time()  # Waktu awal
    timeout = 60  # Waktu timeout dalam detik
    message_displayed = False
    message_container = st.empty()

    while True:
        data = read_data()
        data_split = data.split(",")
        index0 = int(data_split[0].strip("{").strip())
        value1 = int(data_split[1].strip("}").strip())

        if not message_displayed:
            remaining_time = int(timeout - (time.time() - start_time))
            if remaining_time > 0:
                message_container.text(f"Tunggu {remaining_time} Detik")
            else:
                message_displayed = True  # Setelah pesan pertama kali ditampilkan, pesan tidak akan muncul lagi
                message_container.empty()  # Hapus pesan setelah timeout

        if message_displayed:
            st.text(f"Kadar Air : {value1}%")
            if index0 == 0:
                st.text(f"Aroma Beras : Normal")
                break
            elif index0 == 1:
                st.text(f"Aroma Beras : Apek")
                break
            elif index0 == 2:
                st.text(f"Aroma Beras : Pestisida")
                break
            break


if __name__ == "__main__":
    main()
