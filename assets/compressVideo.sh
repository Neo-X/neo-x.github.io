#!/bin/bash

for i in *.mp4;
  do name=`echo "$i" | cut -d'.' -f1`
  echo "$name"
  ffmpeg -i "$i" -vcodec libx265 -crf 38 -r 30 "${name}-compressed.mp4"
done

for i in *.gif;
  do name=`echo "$i" | cut -d'.' -f1`
  echo "$name"
  ffmpeg -i "$i" -vcodec libx265 -crf 28 -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "${name}-compressed.mp4"
done

mogrify -resize 1080x920 *.png

