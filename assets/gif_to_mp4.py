from moviepy.editor import VideoFileClip



# clip = VideoFileClip("assets/jeep/concat.gif")
# clip.write_videofile("assets/jeep-unet-full-supvis/concat.mp4", codec="libx264", fps=24)

# clip = VideoFileClip("assets/jeep-unet-full-supvis/concat.gif")
# clip.write_videofile("assets/jeep-unet-full-supvis/concat.mp4", codec="libx264", fps=8)

# clip = VideoFileClip("assets/man-skate-unet-full-supvis/concat.gif")
# clip.write_videofile("assets/man-skate-unet-full-supvis/concat.mp4", codec="libx264", fps=8)

clip = VideoFileClip("assets/tesla-unet-full-supvis/concat.gif")
clip.write_videofile("assets/tesla-unet-full-supvis/concat.mp4", codec="libx264", fps=8)