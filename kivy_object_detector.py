from pathlib import Path

import cv2
import numpy as np
# from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
# from kivy.core.camera import camera_android
from kivy.logger import Logger
# from kivy.uix.button impordt Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.utils import platform
from kivy_garden.xcamera import XCamera

from kivymd.app import MDApp as App
from kivymd.uix.button import MDFlatButton as Button
from utils import define_img_size, find_faces, draw_on_faces, make_new_texture_frame

Logger.info(f"Versions: Numpy {np.__version__}")
Logger.info(f"Versions: Opencv {cv2.__version__}")

image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
strides = [8.0, 16.0, 32.0, 64.0]


class CameraCV(XCamera):
    def on_tex(self, *l):
        self.image_bytes = self._camera.texture.pixels
        self.image_size = self._camera.texture.size


class CamApp(App):
    def build(self):
        model_dir = Path("RFB-320")
        Logger.info(f"Model: Model directory path: {model_dir.__str__()}")
        self.model = cv2.dnn.readNetFromCaffe(Path(model_dir, "RFB-320.prototxt").__str__(),
                                              Path(model_dir, "RFB-320.caffemodel").__str__())
        Logger.info(f"Model: Model has loaded successfully ({self.model})")

        overlap_path = Path("Thug-Life-Glasses-PNG.png")
        self.overlap_image = cv2.imread(overlap_path.__str__(), cv2.IMREAD_UNCHANGED)
        self.overlap_image = cv2.resize(self.overlap_image, (320, 240))
        self.thug_life = True

        self.image_mean = np.array([127, 127, 127])
        self.image_std = 128.0
        self.iou_threshold = 0.3
        self.center_variance = 0.1
        self.size_variance = 0.2
        self.min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
        self.strides = [8.0, 16.0, 32.0, 64.0]
        self.threshold = 0.7
        self.size_ratio = None

        self.input_size = (320, 240)
        self.width = self.input_size[0]
        self.height = self.input_size[1]
        self.priors = define_img_size(self.input_size)

        self.check_window_size()

        self.img1 = Image(pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.speed_button = Button(text="Change display size", size_hint=(0.2, 0.4),
                                   pos_hint={'center_x': 0.25, 'center_y': 0.125})
        self.thug_button = Button(text="switch thug life", size_hint=(0.2, 0.4),
                                  pos_hint={'center_x': 0.75, 'center_y': 0.125})
        self.thug_button.bind(on_press=self.switch_thug_life)
        self.speed_button.bind(on_press=self.set_display_speed)
        layout = FloatLayout(size=Window.size)
        layout.add_widget(self.img1)
        layout.add_widget(self.speed_button)
        layout.add_widget(self.thug_button)

        self.display_speed = 2  # 0 for best resolution, 1 for medium, 2 for fastest display
        desired_resolution = (720, 480)
        self.camCV = CameraCV(play=True, resolution=desired_resolution)
        self.camCV.image_bytes = False

        Clock.schedule_interval(self.update_texture, 1.0 / 60.0)
        return layout

    def set_display_speed(self, instance):
        if self.display_speed == 2:
            self.display_speed = 0
        else:
            self.display_speed += 1

    def check_window_size(self):
        self.window_shape = Window.size
        self.window_width = self.window_shape[0]
        self.window_height = self.window_shape[1]
        Logger.info(f"Screen: Window size is {self.window_shape}")

    def switch_thug_life(self, instance):
        self.thug_life = not self.thug_life

    def update_texture(self, instance):
        self.check_window_size()
        if type(self.camCV.image_bytes) == bool:
            Logger.info("Camera: No valid frame")
            return
        Logger.info(f"Camera: image bytes {len(self.camCV.image_bytes)}")
        Logger.info(f"Camera: image size {self.camCV.image_size}")
        if not self.size_ratio:
            self.camera_width = self.camCV.image_size[0]
            self.camera_height = self.camCV.image_size[1]
            self.size_ratio = self.camera_height / self.camera_width

        Logger.info(f"Camera: update texture")
        self.extract_frame()
        self.process_frame()
        self.display_frame()

        Logger.info(f"Camera: converted to gray and back to rgba")

    def extract_frame(self):
        self.frame = np.frombuffer(self.camCV.image_bytes, np.uint8)
        Logger.info(f"Camera: frame exist")
        self.frame = self.frame.reshape((self.camCV.image_size[1], self.camCV.image_size[0], 4))
        Logger.info(f"Camera: frame size {self.frame.shape}")

    def process_frame(self):
        boxes = find_faces(self, platform)
        self.frame = draw_on_faces(self, boxes, platform)

    def display_frame(self):
        self.img1.texture = make_new_texture_frame(self)


if __name__ == '__main__':
    CamApp().run()
