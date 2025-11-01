import imageio
import numpy as np
import io

class VideoRecorder:
    """ A simple class to record a series of Matplotlib frames and save them to a video file """
    def __init__(self, fig, filename="simulation_video.mp4", fps=10):
        """
        Initializes the recorder
        :param fig: The Matplotlib figure object to capture
        :param filename: The output video file name
        :param fps: The frames per second for the output video
        """
        self.fig = fig
        self.filename = filename
        self.fps = fps
        self._frames = []
        print(f"VideoRecorder initialized. Video will be saved to '{self.filename}' at {self.fps} FPS.")

    def capture_frame(self):
        """
        Captures the current state of the figure and stores it in memory.
        This should be called after the plot has been fully drawn in each loop.
        """
        with io.BytesIO() as buf:
            self.fig.savefig(buf, format='png', dpi=150) # dpi should be adjusted for quality
            buf.seek(0)

            frame = imageio.imread(buf)
            self._frames.append(frame)

    def save_video(self):
        """
        Compiles all captured frames into a video file and saves it.
        This should be called once at the very end of the simulation.
        """
        if not self._frames:
            print("Warning: No frames were captured. Cannot save video.")
            return

        print(f"Saving video with {len(self._frames)} frames to '{self.filename}'...")
        imageio.mimsave(self.filename, self._frames, fps=self.fps)
        print("Video saved successfully.")