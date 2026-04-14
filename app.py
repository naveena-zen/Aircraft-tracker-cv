import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

class CornerDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Corner Detection App")

        # Styling
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        self.label_title = tk.Label(self.root, text="Corner Detection of Aeroplane", font=("Arial", 20), bg="#f0f0f0")
        self.label_title.pack(pady=20)

        self.select_button = tk.Button(self.root, text="Select Video", command=self.select_video, bg="#4caf50", fg="white", font=("Arial", 14))
        self.select_button.pack(pady=10)

        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="white")
        self.canvas.pack()

        # Video properties
        self.cap = None
        self.video_path = ''

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if self.video_path:
            self.start_detection()

    def start_detection(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.detect_corners()

    def detect_corners(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
                if corners is not None:
                    corners = np.int0(corners)
                    for corner in corners:
                        x, y = corner.ravel()
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                frame = cv2.resize(frame, (640, 480))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tostring())

                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            self.root.after(10, self.detect_corners)

def main():
    root = tk.Tk()
    app = CornerDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()