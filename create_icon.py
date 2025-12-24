#!/usr/bin/env python3
"""Create a simple app icon"""
from PIL import Image, ImageDraw, ImageFont
import os

# Create a 1024x1024 icon
size = 1024
img = Image.new('RGBA', (size, size), (70, 130, 180, 255))  # Steel blue background
draw = ImageDraw.Draw(img)

# Draw a simple "SW" text
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 300)
except:
    font = ImageFont.load_default()

text = "SW"
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
x = (size - text_width) // 2
y = (size - text_height) // 2 - bbox[1]
draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

# Save as PNG
img.save('icon_1024x1024.png', 'PNG')
print("Created icon_1024x1024.png")

