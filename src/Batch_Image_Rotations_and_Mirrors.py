from PIL import Image

image_list =["Screenshot from 2016-12-29 14-02-31.png_2_1.jpg",
"Screenshot from 2016-12-29 14-02-31.png_2_2.jpg"]

print "Processing {0} pictures into a total of {1} rotated & mirrored \
       images...\n".format(len(image_list), len(image_list)*8)

print "Running Mirror 0 Degrees..."
# Mirror 0 Degrees
for image in image_list:
    img = Image.open(image)
    img_mir = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_mir.save(image + '_mir.jpg')

print "Completed Mirror 0 Degrees"

print "Running Rotation 90 Degrees..."
# Rotation 90 Degrees
for image in image_list:
    img = Image.open(image)
    img_rot = img.rotate(90)
    img_rot.save(image + '_deg90.jpg')

print "Completed Rotation 90 Degrees"

print "Running Mirror 90 Degrees..."
# Mirror 90 Degrees
for image in image_list:
    img = Image.open(image)
    img_mir = img.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
    img_mir.save(image + '_deg90_mir.jpg')

print "Completed Mirror 90 Degrees"

print "Running Rotation 180 Degrees..."
# Rotation 180 Degrees
for image in image_list:
    img = Image.open(image)
    img_rot = img.rotate(180)
    img_rot.save(image + '_deg180.jpg')

print "Completed Rotation 180 Degrees"

print "Running Mirror 180 Degrees..."
# Mirror 180 Degrees
for image in image_list:
    img = Image.open(image)
    img_mir = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    img_mir.save(image + '_deg180_mir.jpg')

print "Completed Mirror 180 Degrees"

print "Running Rotation 270 Degrees..."
# Rotation 270 Degrees
for image in image_list:
    img = Image.open(image)
    img_rot = img.rotate(270)
    img_rot.save(image + '_deg270.jpg')

print "Completed Rotation 270 Degrees"

print "Running Mirror 270 Degrees..."
# Mirror 180 Degrees
for image in image_list:
    img = Image.open(image)
    img_mir = img.rotate(270).transpose(Image.FLIP_LEFT_RIGHT)
    img_mir.save(image + '_deg270_mir.jpg')

print "Completed Mirror 270 Degrees"
