import config
import cv2
import os
import subprocess  # FFMPEG'i çağırmak için gerekli


class VideoProcessor:
    # __init__, extract_frames, get_frames fonksiyonlarınız aynı kalacak...

    def __init__(self):
        self.video_path = config.video_path
        self.skip_frame = config.skip_frame
        self.extracted_frames = []
        self.fps = 0
        self.width = 0
        self.height = 0

    def extract_frames(self):
        if not os.path.exists(self.video_path):
            print(f"Error: Video file not found at '{self.video_path}'")
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self.fps or self.fps < 1:
            print(f"UYARI: Geçersiz FPS değeri ({self.fps}) algılandı. Varsayılan olarak 24 FPS kullanılacak.")
            self.fps = 24.0
        print(f"Video properties: {self.width}x{self.height} @ {self.fps:.2f} FPS")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.skip_frame == 0:
                self.extracted_frames.append(frame)
            frame_count += 1
        cap.release()
        print(f"Frame extraction complete. Total frames extracted: {len(self.extracted_frames)}")

    def create_new_video(self, output_path, frames_to_use=None):
        """
        Önce OpenCV ile geçici bir video oluşturur, ardından FFMPEG ile
        bu videoyu web uyumlu, standart bir MP4 formatına dönüştürür.

        GÜNCELLENDİ: Her bir işlenmiş kareyi, atlanan kare sayısını telafi etmek
        için `self.skip_frame` kadar videoya yazar.
        """
        frames = frames_to_use if frames_to_use is not None else self.extracted_frames
        if not frames:
            print("HATA: Video oluşturmak için kullanılacak kare bulunamadı.")
            return

        try:
            video_height, video_width, _ = frames[0].shape
        except (IndexError, AttributeError):
            print("HATA: Karelerin boyutları belirlenemedi.")
            return
        video_fps = self.fps if self.fps > 0 else 24.0

        temp_output_path = output_path + ".temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_output_path, fourcc, video_fps, (video_width, video_height))

        if not writer.isOpened():
            print("HATA: Geçici video için VideoWriter başlatılamadı.")
            return

        # --- DEĞİŞİKLİK BURADA ---
        # Her bir işlenmiş kareyi 'self.skip_frame' sayısı kadar yazarak
        # videonun orijinal süresini koruyoruz.
        print(f"Writing frames to temporary video... Each frame will be duplicated {self.skip_frame} times.")
        for frame in frames:
            for _ in range(self.skip_frame):
                writer.write(frame)
        # --- DEĞİŞİKLİK SONA ERDİ ---

        writer.release()

        print("FFMPEG ile video web uyumlu formata dönüştürülüyor...")
        ffmpeg_command = [
            'ffmpeg',
            '-i', temp_output_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-y',
            output_path
        ]

        try:
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Video başarıyla '{output_path}' konumuna kaydedildi.")
        except FileNotFoundError:
            print("HATA: FFMPEG komutu bulunamadı. Lütfen FFMPEG'in kurulu ve sistem PATH'inde olduğundan emin olun.")
        except subprocess.CalledProcessError as e:
            print("HATA: FFMPEG video dönüştürme sırasında bir hata oluştu.")
            print("FFMPEG Hata Mesajı:", e.stderr.decode())
        finally:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
    def get_frames(self):
        return self.extracted_frames