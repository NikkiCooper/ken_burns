import os
import sys
import argparse
from Bcolors import Bcolors


def cliOpts():
    bc = Bcolors()
    parser = argparse.ArgumentParser(
        description="Fade: GPU-accelerated slideshow with crossfade dissolve or fade-to-black transitions",
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
                             f"{bc.BOLD+bc.Red_f}Required  {bc.Dark_Gray_f}(not required for --calculate / --gen_filter_map){bc.ENDC}")

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
    grp_time.add_argument("--trans_time", action="store", type=float, dest="trans_time", default=1.0,
                          help=f"Transition duration in seconds\n"
                               f"  Crossfade : overlap duration between consecutive slides\n"
                               f"  --ftb     : per-side fade duration (fade-out + fade-in at each boundary)\n"
                               f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}1.0s{bc.ENDC}")

    # -------------------------------------------------------------------------
    # Transitions
    # -------------------------------------------------------------------------
    grp_trans = parser.add_argument_group("Transitions")
    grp_trans.add_argument("--ftb", action="store_true", dest="ftb", default=False,
                           help=f"Fade-to-black: each slide fades to black, then the next fades in from black\n"
                                f"  Default mode (without this flag) is crossfade dissolve\n"
                                f"  Mutually exclusive with --random\n"
                                f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}off{bc.ENDC}")
    grp_trans.add_argument("--random", action="store_true", dest="random", default=False,
                           help=f"Randomly choose crossfade or fade-to-black per slide boundary\n"
                                f"  Mutually exclusive with --ftb\n"
                                f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}off{bc.ENDC}")
    grp_trans.add_argument("--drift", action="store", type=float, dest="drift", default=0.0,
                           help=f"Subtle per-slide zoom drift amount\n"
                                f"  0.0      = disabled (pure static hold)\n"
                                f"  Range    : 0.01 – 1.0\n"
                                f"  Suggested: 0.33  (~5%% crop change — visible but subtle)\n"
                                f"  At 1.0   : equivalent to full Ken Burns zoom (~15%% crop change)\n"
                                f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}0.0 (disabled){bc.ENDC}")
    grp_trans.add_argument("--rand_drift", action="store_true", dest="rand_drift", default=False,
                           help=f"Randomize zoom direction (in or out) independently per slide\n"
                                f"  Without this flag direction alternates evenly across slides\n"
                                f"  Has no effect unless --drift > 0\n"
                                f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}off{bc.ENDC}")

    # -------------------------------------------------------------------------
    # Image Processing
    # -------------------------------------------------------------------------
    grp_proc = parser.add_argument_group("Image Processing")
    grp_proc.add_argument("--smooth", action="store", type=str, dest="smooth", default=None,
                          choices=["subtle", "light", "medium", "strong"],
                          help=f"Apply CUDA bilateral filter to reduce ESRGAN jagged edges\n"
                               f"  subtle : d=3,  sigmaColor=15,  sigmaSpace=15,  intensity=40%%\n"
                               f"  light  : d=5,  sigmaColor=35,  sigmaSpace=35,  intensity=50%%\n"
                               f"  medium : d=9,  sigmaColor=75,  sigmaSpace=75,  intensity=70%%\n"
                               f"  strong : d=13, sigmaColor=120, sigmaSpace=120, intensity=85%%\n"
                               f"  See --filter_help for full guidance\n"
                               f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}off{bc.ENDC}")
    grp_proc.add_argument("--sharpen", action="store", type=str, dest="sharpen", default=None,
                          choices=["subtle", "light", "medium", "strong"],
                          help=f"Apply CUDA Laplacian edge boost to recover sharpness (apply after --smooth)\n"
                               f"  subtle : boost_strength=0.10\n"
                               f"  light  : boost_strength=0.25\n"
                               f"  medium : boost_strength=0.45\n"
                               f"  strong : boost_strength=0.65\n"
                               f"  See --filter_help for full guidance\n"
                               f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}off{bc.ENDC}")
    grp_proc.add_argument("--smooth_intensity", action="store", type=float, dest="smooth_intensity",
                          default=None, metavar="0-100",
                          help=f"Override bilateral filter blend intensity  (0=no effect, 100=full filter)\n"
                               f"  Overrides the preset default intensity value\n"
                               f"  Preset defaults:\n"
                               f"    subtle=40%%  light=50%%  medium=70%%  strong=85%%\n"
                               f"  Lower values apply the filter more gently without changing\n"
                               f"  the filter kernel (d, sigmaColor, sigmaSpace)\n"
                               f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}preset value{bc.ENDC}")
    grp_proc.add_argument("--use_filter_map", action="store_true", dest="use_filter_map", default=False,
                          help=f"Apply per-image smooth/sharpen settings from filter_map.json\n"
                               f"  The map must be located in --img_path (generated by --gen_filter_map)\n"
                               f"  Images in the map override global --smooth/--sharpen for that slide\n"
                               f"  Images not in the map fall back to global --smooth/--sharpen\n"
                               f"  See --fmap_help for full guidance\n"
                               f"{bc.BOLD+bc.Magenta_f}Requires: {bc.OKBLUE}--img_path  {bc.Dark_Gray_f}(map path: <img_path>/filter_map.json){bc.ENDC}")

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
    grp_util.add_argument("--filter_help", action="store_true", dest="filter_help", default=False,
                          help=f"Print comprehensive guidance on --smooth, --sharpen, --smooth_intensity\n"
                               f"{bc.BOLD+bc.Magenta_f}No other arguments required{bc.ENDC}")
    grp_util.add_argument("--fmap_help", action="store_true", dest="fmap_help", default=False,
                          help=f"Print comprehensive guidance on filter maps\n"
                               f"  Covers --gen_filter_map, --use_filter_map, JSON format, examples\n"
                               f"{bc.BOLD+bc.Magenta_f}No other arguments required{bc.ENDC}")
    grp_util.add_argument("--gen_filter_map", action="store_true", dest="gen_filter_map", default=False,
                          help=f"Analyse images and write filter_map.json to --img_path\n"
                               f"  Prints score table (lowest→highest) for threshold calibration\n"
                               f"  Images above --fmap_threshold are flagged; smooth preset is chosen\n"
                               f"  automatically by score band: +0→subtle +15→light +30→medium +45→strong\n"
                               f"  Sharpen is NOT auto-suggested — add manually to JSON entries if needed\n"
                               f"  or use global --sharpen at render time as a fallback\n"
                               f"  Then render with --use_filter_map to apply per-image settings\n"
                               f"  See --fmap_help for full guidance\n"
                               f"{bc.BOLD+bc.Magenta_f}Requires: {bc.OKBLUE}--img_path{bc.ENDC}")
    grp_util.add_argument("--fmap_threshold", action="store", type=float, dest="fmap_threshold", default=30.0,
                          help=f"Staircase score threshold for --gen_filter_map  (range 0–100)\n"
                               f"  Images scoring above this are flagged for filtering\n"
                               f"  0 = perfectly consistent edge directions (no staircasing)\n"
                               f"  100 = maximally inconsistent (pure H/V zigzag staircasing)\n"
                               f"  Run once, inspect the score table, re-run with adjusted value\n"
                               f"{bc.BOLD+bc.Magenta_f}Default: {bc.OKBLUE}30.0{bc.ENDC}")

    options = parser.parse_args()

    # ---- filter_help: no other args needed ----
    if options.filter_help:
        from fade import print_filter_help
        print_filter_help()
        sys.exit(0)

    # ---- fmap_help: no other args needed ----
    if options.fmap_help:
        from fade import print_fmap_help
        print_fmap_help()
        sys.exit(0)

    # ---- all remaining modes require --img_path ----
    if options.img_path is None:
        print(f"{bc.BOLD+bc.Red_f}--img_path is required.{bc.ENDC}")
        sys.exit(1)
    tmp_img_path = os.path.expanduser(options.img_path)
    if not os.path.isdir(tmp_img_path):
        print(f"{bc.BOLD+bc.Red_f}Invalid --img_path: {bc.OKBLUE}{tmp_img_path}{bc.ENDC}")
        sys.exit(1)

    # ---- gen_filter_map: only needs --img_path ----
    if options.gen_filter_map:
        return options, tmp_img_path, None

    # ---- calculate: only needs --img_path ----
    if options.calculate:
        return options, tmp_img_path, None

    # ---- mutual exclusion: --ftb / --random ----
    if options.ftb and options.random:
        print(f"{bc.BOLD+bc.Red_f}--ftb and --random are mutually exclusive.{bc.ENDC}")
        sys.exit(1)

    # ---- smooth_intensity range ----
    if options.smooth_intensity is not None:
        if not (0.0 <= options.smooth_intensity <= 100.0):
            print(f"{bc.BOLD+bc.Red_f}--smooth_intensity must be in range 0–100 (got {options.smooth_intensity}){bc.ENDC}")
            sys.exit(1)
        options.smooth_intensity = options.smooth_intensity / 100.0  # normalise to 0.0–1.0

    # ---- drift range ----
    if not (0.0 <= options.drift <= 1.0):
        print(f"{bc.BOLD+bc.Red_f}--drift must be in range 0.0–1.0 (got {options.drift:.4f}){bc.ENDC}")
        sys.exit(1)

    # ---- warn if --rand_drift given without --drift ----
    if options.rand_drift and options.drift == 0.0:
        print(f"{bc.BOLD+bc.Yellow_f}Warning: --rand_drift has no effect without --drift > 0{bc.ENDC}")

    # ---- normal render mode requires --output ----
    if options.output is None:
        print(f"{bc.BOLD+bc.Red_f}--output is required for rendering.{bc.ENDC}")
        sys.exit(1)

    tmp_output = os.path.expanduser(options.output)
    tmp_out_dir = os.path.dirname(os.path.abspath(tmp_output))
    if not os.path.isdir(tmp_out_dir):
        print(f"{bc.BOLD+bc.Red_f}Output directory does not exist: {bc.OKBLUE}{tmp_out_dir}{bc.ENDC}")
        sys.exit(1)

    # ---- timing validation ----
    if options.trans_time <= 0:
        print(f"{bc.BOLD+bc.Red_f}--trans_time must be > 0.{bc.ENDC}")
        sys.exit(1)
    if 2.0 * options.trans_time >= options.duration:
        print(f"{bc.BOLD+bc.Red_f}--trans_time ({options.trans_time}s × 2 = {2*options.trans_time}s) "
              f"meets or exceeds --duration ({options.duration}s).\n"
              f"Reduce --trans_time or increase --duration.{bc.ENDC}")
        sys.exit(1)

    return options, tmp_img_path, tmp_output
