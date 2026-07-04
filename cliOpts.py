import os
import sys
import argparse
from Bcolors import Bcolors


def cliOpts():
    bc = Bcolors()
    parser = argparse.ArgumentParser(
        description="Ken Burns: GPU-accelerated slideshow with zoom and fade-to-black transitions",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------
    grp_io = parser.add_argument_group("File I/O")
    grp_io.add_argument("--img_path", action="store", type=str, dest="img_path", default=None,
                        help=f"Source directory of images\n"
                             f"{bc.BOLD+bc.Red_f}Required{bc.ENDC}")
    grp_io.add_argument("-o", "--output", action="store", type=str, dest="output", default=None,
                        help=f"Full output path for the generated video  (directory + filename)\n"
                             f"  Example: /home/user/Videos/slideshow.mp4\n"
                             f"{bc.BOLD+bc.Red_f}Required  {bc.Dark_Gray_f}(not required for --calculate){bc.ENDC}")

    # -------------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------------
    grp_time = parser.add_argument_group("Timing")
    grp_time.add_argument("-d", "--duration", action="store", type=float, dest="duration", default=8.0,
                          help=f"Per-slide display duration in seconds\n"
                               f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}8.0s{bc.ENDC}")
    grp_time.add_argument("-f", "--fps", action="store", type=int, dest="fps", default=30,
                          help=f"Frames per second\n"
                               f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}30{bc.ENDC}")
    grp_time.add_argument("--ftb", action="store", type=float, dest="ftb", default=1.0,
                          help=f"Fade-to-black duration in seconds per side of each slide\n"
                               f"  Each slide fades in from black at the start\n"
                               f"  and fades out to black at the end\n"
                               f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}1.0s{bc.ENDC}")

    # -------------------------------------------------------------------------
    # Zoom
    # -------------------------------------------------------------------------
    grp_zoom = parser.add_argument_group("Zoom")
    grp_zoom.add_argument("-z", "--zoom_in", action="store_true", dest="zoom_in", default=False,
                          help=f"Start first slide zooming in  (default: zoom out)\n"
                               f"  Direction alternates per slide with randomised streaks\n"
                               f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}zoom-out{bc.ENDC}")

    # -------------------------------------------------------------------------
    # Encoding
    # -------------------------------------------------------------------------
    grp_enc = parser.add_argument_group("Encoding")
    grp_enc.add_argument("--h264", action="store_true", dest="h264", default=False,
                         help=f"Use H.264 instead of default HEVC (H.265) via NVENC\n"
                              f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}HEVC{bc.ENDC}")

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------
    grp_util = parser.add_argument_group("Utility")
    grp_util.add_argument("--calculate", action="store_true", dest="calculate", default=False,
                          help=f"Calculate and display estimated video duration based on current settings\n"
                               f"{bc.BOLD+bc.Magenta_f}Requires: {bc.OKBLUE}--img_path{bc.ENDC}")

    options = parser.parse_args()

    # ---- all modes require --img_path ----
    if options.img_path is None:
        print(f"{bc.BOLD+bc.Red_f}--img_path is required.{bc.ENDC}")
        sys.exit(1)
    tmp_img_path = os.path.expanduser(options.img_path)
    if not os.path.isdir(tmp_img_path):
        print(f"{bc.BOLD+bc.Red_f}Invalid --img_path: {bc.OKBLUE}{tmp_img_path}{bc.ENDC}")
        sys.exit(1)

    # ---- calculate: only needs --img_path ----
    if options.calculate:
        return options, tmp_img_path, None

    # ---- render mode requires --output ----
    if options.output is None:
        print(f"{bc.BOLD+bc.Red_f}--output is required for rendering.{bc.ENDC}")
        sys.exit(1)

    tmp_output = os.path.expanduser(options.output)
    tmp_out_dir = os.path.dirname(os.path.abspath(tmp_output))
    if not os.path.isdir(tmp_out_dir):
        print(f"{bc.BOLD+bc.Red_f}Output directory does not exist: {bc.OKBLUE}{tmp_out_dir}{bc.ENDC}")
        sys.exit(1)

    # ---- ftb validation ----
    if options.ftb <= 0:
        print(f"{bc.BOLD+bc.Red_f}--ftb must be > 0.{bc.ENDC}")
        sys.exit(1)
    if 2.0 * options.ftb >= options.duration:
        print(f"{bc.BOLD+bc.Red_f}--ftb ({options.ftb}s × 2 = {2*options.ftb}s) "
              f"meets or exceeds --duration ({options.duration}s).\n"
              f"Reduce --ftb or increase --duration.{bc.ENDC}")
        sys.exit(1)

    return options, tmp_img_path, tmp_output
