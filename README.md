# Stratford_Light_Festival
Designing a python and YOLOv8-based computer vision system for the Stratford Light Festival - aims to create a colored silhouette outline that radiate outwards depending on quantity of movement

To develop this project, I implemented a trained human-model and integrated it using YOLOv8. Combined with cv2 and numpy libraries, I was able to track silhouett outlines for humans. From there, I implemented a black background and a glowing effect that radiates more based on the amount of movemnet (tracked with pixels).
