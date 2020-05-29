import os
path = '/media/amax/guo/Guo_dataset/Guo/yt_bb_detection_train/1'
videos = os.listdir(path)
path_crop = '/data0/youtubebb/crop511/yt_bb_detection_train/1'
video_crop = os.listdir(path_crop)
num_have = len(video_crop)
num_amount = len(videos)
num_miss = 0
num_corr = 0
for video in videos:
    video_name = video.split('+')
    if video_name[0] in video_crop:
        if int(video_name[2][0]) > 0:
            print(video_name[0])
            num_have += 1
        continue
    else:
        num_miss += 1
print('num_amount:',num_amount,'num_have:',num_have,'num_miss:',num_miss)