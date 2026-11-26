#!/usr/bin/env python3
import os
import sys
import argparse
import cv2

def main():
    output_folder = None
    parser = argparse.ArgumentParser(description="Extract slideshow frames manually from a slideshow video.")
    parser.add_argument("video",help="Path to the input MP4 video file")
    parser.add_argument("-t", "--transition", type=float, required=True, help="Transition duration in seconds + 1 sec")
    parser.add_argument("-s", "--still", type=float, required=True, help="Photo still duration in seconds")
    parser.add_argument("--drift", type=float, default=0.0, help="Manual drift correction in seconds per slide (e.g., 0.01 or -0.01)")
    parser.add_argument("-w", "--width", type=int, default=3840, help="Target width")
    parser.add_argument("-e", "--height", type=int, default=2160, help="Target height")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Simulate extraction and show total slide count without writing files")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only output Total calculated slides")

    args = parser.parse_args()

    # Check CUDA availability on video player startup
    #cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    #print(f"🎬 Hardware query: CUDA devices available: {cuda_devices}")
    #print()


    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video stream for '{args.video}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_secs = total_frames / fps

    # Base timing loop
    slide_duration_secs = args.transition + args.still
    first_snapshot_offset_secs = args.transition + (args.still / 2.0)

    print(f"\n--- Vetting Video: {args.video} ---")
    print(f"FPS: {fps:.2f} | Total Frames: {total_frames} | Total Duration: {video_duration_secs:.2f}s")
    print(f"Base Slide Loop: {slide_duration_secs}s | First Frame Target: {first_snapshot_offset_secs}s")

    if args.dry_run:
        print("MODE: [DRY-RUN SIMULATION] - No files will be written.\n")
    else:
        if not args.quiet:
            print(f"MODE: [LIVE EXTRACTION] - Upscaling to {args.width}x{args.height} via RTX A4500 CUDA...\n")
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_folder = f"{video_name}_extracted_frames"
        os.makedirs(output_folder, exist_ok=True)
        gpu_frame_src = cv2.cuda_GpuMat()

    slide_count = 1
    current_target_time = first_snapshot_offset_secs

    while True:
        # Convert target time strictly to an exact frame integer
        target_frame = int(round(current_target_time * fps))

        # Break if we overshoot the video timeline
        if target_frame >= total_frames:
            break

        if args.dry_run:
            # Quick mathematical simulation output
            print(f"  [Simulated] Slide {slide_count:03d} would hit frame {target_frame} (~{current_target_time:.2f}s)")
        else:
            # Full live extraction and upscale
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            success, frame = cap.read()
            if not success:
                break

            if not args.quiet:
                print(f"  -> Extracting Slide {slide_count:03d} at frame {target_frame} (~{current_target_time:.2f}s)...")

            # RTX A4500 CUDA Processing Pipeline
            gpu_frame_src.upload(frame)
            gpu_frame_dst = cv2.cuda.resize(gpu_frame_src, (args.width, args.height), interpolation=cv2.INTER_CUBIC)
            scaled_frame = gpu_frame_dst.download()

            # Save output frame
            output_path = os.path.join(output_folder, f"slide_{slide_count:03d}.jpg")
            cv2.imwrite(output_path, scaled_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Advance time timeline
        current_target_time += (slide_duration_secs + args.drift)
        slide_count += 1

    cap.release()

    total_extracted = slide_count - 1
    if args.dry_run:
        print(f"\n[DRY-RUN DONE] Total calculated slides for these parameters: {total_extracted}\n")

    if args.quiet:
        # Output ONLY the integer count to STDERR for easy Bash capturing
        sys.stderr.write(f"{total_extracted}\n")
    else:
        # Standard descriptive output to STDOUT for human reading
        print(f"\n[SUCCESS] Extracted {total_extracted} frames into folder: './{output_folder}/'\n")

if __name__ == "__main__":
    main()
