import sys
import glob
import subprocess
from pathlib import Path


main_dir = "/media/hdd/acances/Friends"
shots_dir = main_dir + "/shots"
videos_dir = main_dir + "/videos"
subvideos_dir = main_dir + "/subvideos"

FPS = 23.98


def convert_to_ffpmeg_time_format(seconds):
    milliseconds = seconds * 1000
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds = float(milliseconds) / 1000
    s = "%i:%02i:%06.3f" % (hours, minutes, seconds)
    return s


def gather_shots(episode_number):
    shots_file = "{}/episode{:02d}_shots.txt".format(shots_dir, episode_number)
    shots = []
    with open(shots_file, "r") as f:
        for line in f:
            b, e = tuple(map(int, line.strip().split()))
            shots.append([b, e])
    return shots


def make_subvideo(episode_number, shot, shot_id):
    begin_frame, end_frame = shot
    begin_time = begin_frame / FPS
    end_time = end_frame / FPS
    nb_frames = end_frame - begin_frame + 1
    print(begin_frame)
    print(nb_frames)
    print(convert_to_ffpmeg_time_format(begin_time))
    print(convert_to_ffpmeg_time_format(end_time))

    original_video = "{}/friends.s03e{:02d}.720p.bluray.sujaidr.mkv".format(videos_dir, episode_number)
    new_video = "{}/episode{:02d}/episode{:02d}_shot{:03d}.mkv".format(subvideos_dir, episode_number, episode_number, shot_id)
    Path("{}/episode{:02d}".format(subvideos_dir, episode_number)).mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y",
           "-i", original_video,
        #    "-c", "copy",
           "-ss", convert_to_ffpmeg_time_format(begin_time),
           "-to", convert_to_ffpmeg_time_format(end_time),
           "-c:v", "libx264",
           "-crf", "30",
        #    "-c:a", "aac",
        #    "-frames:v", str(nb_frames),
        #    "-shortest",
           new_video]
    
    subprocess.call(cmd)


def make_all_subvideos(episode_number):
    shots = gather_shots(episode_number)
    for shot_id, shot in enumerate(shots):
        make_subvideo(episode_number, shot, shot_id)


if __name__ == "__main__":
    episode_number = int(sys.argv[1])
    shots = gather_shots(episode_number)

    if len(sys.argv) > 2:
        shot_id = int(sys.argv[2])
        shot = shots[shot_id]
        make_subvideo(episode_number, shot, shot_id)
    else:
        make_all_subvideos(episode_number)
