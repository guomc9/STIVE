from moviepy.editor import VideoFileClip



# clip = VideoFileClip("assets/jeep/concat.gif")
# clip.write_videofile("assets/jeep-unet-full-supvis/concat.mp4", codec="libx264", fps=24)

clip = VideoFileClip("assets/jeep-unet-full-supvis/concat.gif")
clip.write_videofile("assets/jeep-unet-full-supvis/concat.mp4", codec="libx264", fps=24)