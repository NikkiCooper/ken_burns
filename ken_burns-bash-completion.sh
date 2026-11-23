#!/usr/bin/env bash
# Bash Completion Script for ken_burns.py - v0.1
# Source this file in your terminal for live testing:
# $ source /path/to/ken_burns-bash-completion.sh

_ken_burns_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Argument groups
    local file_opts="--img_path -o --output"
    local timing_opts="-d --duration -f --fps --trans_time"
    local kb_opts="--ken_burns -z --zoom_in"
    local trans_opts="--ftb --random --drift --rand_drift"
    local img_proc_opts="--smooth --sharpen --smooth_intensity --use_filter_map"
    local encoding_opts="--h264 --pix_fmt --cq --dryRun --encoding_help"
    local util_opts="--calculate --filter_help --fmap_help --gen_filter_map --fmap_threshold"

    local all_opts="--help $file_opts $timing_opts $kb_opts $trans_opts $img_proc_opts $encoding_opts $util_opts"

    # GOLDEN RULE: If the user types a hyphen, they want a flag, not a value!
    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "$all_opts" -- "$cur") )
        return 0
    fi

    # Scan backwards to find the last option flag used on the command line
    local last_opt=""
    local opt_index=-1
    for ((i=COMP_CWORD-1; i>0; i--)); do
        if [[ "${COMP_WORDS[i]}" =~ ^--? ]]; then
            last_opt="${COMP_WORDS[i]}"
            opt_index=i
            break
        fi
    done

    # Handle single-argument flags (reset last_opt after 1 argument)
    if [[ -n "$last_opt" && "$opt_index" -lt $((COMP_CWORD - 1)) ]]; then
        case "$last_opt" in
            # Add any multi-argument flags here if applicable (none in ken_burns by default)
            *)
                last_opt=""
                ;;
        esac
    fi

    # Match against the active option context
    case "$last_opt" in
        --img_path)
            COMPREPLY=( $(compgen -d -- "$cur") )  # Source directory of images
            return 0
            ;;
        -o|--output)
            COMPREPLY=( $(compgen -f -- "$cur") )  # Full output video path
            return 0
            ;;
        --smooth)
            # Print cleanly to stderr so it doesn't corrupt the prompt buffer
            echo -e "\n  [ --smooth Presets ]" >&2
            echo -e "  subtle  -> d=3,  sigmaColor=15,  sigmaSpace=15,  intensity=40%" >&2
            echo -e "  light   -> d=5,  sigmaColor=35,  sigmaSpace=35,  intensity=50%" >&2
            echo -e "  medium  -> d=9,  sigmaColor=75,  sigmaSpace=75,  intensity=70%" >&2
            echo -e "  strong  -> d=13, sigmaColor=120, sigmaSpace=120, intensity=85%" >&2

            COMPREPLY=( $(compgen -W "subtle light medium strong" -- "$cur") )
            return 0
            ;;
        --sharpen)
            echo -e "\n  [ --sharpen Presets ]" >&2
            echo -e "  subtle  -> boost_strength=0.10" >&2
            echo -e "  light   -> boost_strength=0.25" >&2
            echo -e "  medium  -> boost_strength=0.45" >&2
            echo -e "  strong  -> boost_strength=0.65" >&2

            COMPREPLY=( $(compgen -W "subtle light medium strong" -- "$cur") )
            return 0
            ;;
        --smooth_intensity)
            COMPREPLY=( $(compgen -W "{0..100}" -- "$cur") )
            return 0
            ;;
        --drift)
            COMPREPLY=( $(compgen -W "0.0 0.1 0.2 0.33 0.5 0.75 1.0" -- "$cur") )
            return 0
            ;;
        --fmap_threshold)
            COMPREPLY=( $(compgen -W "10.0 20.0 30.0 40.0 50.0 75.0" -- "$cur") )
            return 0
            ;;
        -d|--duration|--trans_time)
            COMPREPLY=( $(compgen -W "1.0 2.0 3.0 5.0 8.0 10.0" -- "$cur") )
            return 0
            ;;
        --pix_fmt)
            COMPREPLY=( $(compgen -W "auto yuv420p p010le yuv444p16le" -- "$cur") )
            return 0
            ;;
        --cq)
            COMPREPLY=( $(compgen -W "18 20 22 24 26 28 30" -- "$cur") )
            return 0
            ;;
        -f|--fps)
            COMPREPLY=( $(compgen -W "24 30 50 60" -- "$cur") )
            return 0
            ;;
        *)
            COMPREPLY=( $(compgen -W "$all_opts" -- "$cur") )
            ;;
    esac
}

# -o nosort ensures numeric ranges and choices stay in exact defined order
complete -o nosort -F _ken_burns_complete ken_burns.py
complete -o nosort -F _ken_burns_complete ken_burns
