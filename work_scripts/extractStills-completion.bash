# bash completion for extractStills.py                         -*- shell-script -*-
#
# Install (pick one):
#   source /path/to/extractStills-completion.bash                 # current shell
#   echo 'source /path/to/extractStills-completion.bash' >> ~/.bashrc
#   cp extractStills-completion.bash \
#      ~/.local/share/bash-completion/completions/extractStills.py   # lazy-loaded
#
# Completes when the script is run directly (extractStills.py, ./extractStills.py,
# /any/path/extractStills.py) or through a symlink named extractStills.
# Uses bash-completion's helpers when available, plain compgen otherwise.

_extractStills_files() {               # $1: extension filter (e.g. mp4) or empty
    if declare -F _filedir >/dev/null; then
        _filedir ${1:+"$1"}
    else
        local IFS=$'\n' word
        word=$(_extractStills_dequote "$cur")
        compopt -o filenames 2>/dev/null
        COMPREPLY=($(compgen -d -- "$word"))
        if [[ -n $1 ]]; then            # no extglob needed: lower- and upper-case pass
            COMPREPLY+=($(compgen -f -X "!*.$1" -- "$word"))
            COMPREPLY+=($(compgen -f -X "!*.${1^^}" -- "$word"))
        else
            COMPREPLY+=($(compgen -f -- "$word"))
        fi
    fi
}

_extractStills_dirs() {
    if declare -F _filedir >/dev/null; then
        _filedir -d
    else
        local IFS=$'\n'
        compopt -o filenames 2>/dev/null
        COMPREPLY=($(compgen -d -- "$(_extractStills_dequote "$cur")"))
    fi
}

_extractStills_dequote() {             # fallback only: Nikki\ Cooper -> Nikki Cooper, drop opening quote
    local w=${1//\\ / }
    w=${w#[\"\']}
    printf '%s' "${w/#\~/$HOME}"
}

_extractStills() {
    local cur prev words cword split=false
    if declare -F _init_completion >/dev/null; then
        _init_completion -s || return       # -s: split --opt=value
    else
        cur=${COMP_WORDS[COMP_CWORD]}
        prev=${COMP_WORDS[COMP_CWORD-1]}
        words=("${COMP_WORDS[@]}")
        cword=$COMP_CWORD
        # '=' is a word break: "--opt=val" arrives as "--opt" "=" "val"
        if [[ $prev == "=" ]]; then
            prev=${COMP_WORDS[COMP_CWORD-2]}
        elif [[ $cur == "=" ]]; then
            prev=${COMP_WORDS[COMP_CWORD-1]}; cur=""
        fi
    fi

    local opts="--video --project-dir --output_base --resolution --png --suffix
                --overwrite --mode --transition --still --drift --frame-pick --margin
                --dryRun --verify --min-confidence --quiet --verbose --logfile --color
                --cpu --version --help --help-all"

    case $prev in
        --resolution)       COMPREPLY=($(compgen -W "2160p 1080p" -- "$cur")); return ;;
        --mode)             COMPREPLY=($(compgen -W "auto manual" -- "$cur")); return ;;
        --frame-pick)       COMPREPLY=($(compgen -W "keyframe center" -- "$cur")); return ;;
        --color)            COMPREPLY=($(compgen -W "auto always never" -- "$cur")); return ;;
        --project-dir|--project_dir|--output_base|--output-base)
                            _extractStills_dirs; return ;;
        --logfile)          # optional value: offer files unless an option is being typed
                            if [[ $cur != -* ]]; then _extractStills_files ""; return; fi ;;
        -t|--transition|-s|--still|--drift|--margin|--min-confidence|--suffix)
                            return ;;       # free-form value
    esac
    $split && return

    # --video takes a list: keep offering .mp4 files until an option is typed.
    if [[ $cur != -* ]]; then
        local i
        for ((i = cword - 1; i > 0; i--)); do
            if [[ ${words[i]} == -* ]]; then
                [[ ${words[i]} == --video ]] && _extractStills_files mp4
                return
            fi
        done
        return
    fi

    COMPREPLY=($(compgen -W "$opts" -- "$cur"))
}

complete -F _extractStills extractStills.py extractStills
