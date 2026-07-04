#  Bcolors.py Copyright (c) 2025 Nikki Cooper
#
#  This program and the accompanying materials are made available under the
#  terms of the GNU Lesser General Public License, version 3.0 which is available at
#  https://www.gnu.org/licenses/gpl-3.0.html#license-text
#
import os

class Bcolors(object):
    """
    Provides a utility class for ANSI escape codes that define text formatting,
    foreground colors, and background colors to enhance console-based applications.

    The `Bcolors` class encapsulates ANSI escape codes used for text styling and
    coloring in terminal applications. It includes predefined foreground and
    background colors, attributes for text formatting, and reset options. This
    class is suitable for creating visually distinct console outputs.

    Attributes:
        Foreground Colors:
            Default_f: Default foreground color.
            Black_f: Black foreground color.
            Red_f: Red foreground color.
            Green_f: Green foreground color.
            Yellow_f: Yellow foreground color.
            Blue_f: Blue foreground color.
            Magenta_f: Magenta foreground color.
            Cyan_f: Cyan foreground color.
            Light_Gray_f: Light gray foreground color.
            Dark_Gray_f: Dark gray foreground color.
            Light_Red_f: Light red foreground color.
            Light_Green_f: Light green foreground color.
            Light_Yellow_f: Light yellow foreground color.
            Light_Blue_f: Light blue foreground color.
            Light_Magenta_f: Light magenta foreground color.
            Light_Cyan_f: Light cyan foreground color.
            White_f: White foreground color.

        Background Colors:
            Default_b: Default background color.
            Black_b: Black background color.
            Red_b: Red background color.
            Green_b: Green background color.
            Yellow_b: Yellow background color.
            Blue_b: Blue background color.
            Magenta_b: Magenta background color.
            Cyan_b: Cyan background color.
            Light_Gray_b: Light gray background color.
            Dark_Gray_b: Dark gray background color.
            Light_Red_b: Light red background color.
            Light_Green_b: Light green background color.
            Light_Yellow_b: Light yellow background color.
            Light_Blue_b: Light blue background color.
            Light_Magenta_b: Light magenta background color.
            Light_Cyan_b: Light cyan background color.
            White_b: White background color.

        Formatting:
            HEADER: Header text style.
            OKBLUE: OK blue text style.
            OKGREEN: OK green text style.
            WARNING: Warning text style.
            FAIL: Failure text style.
            DESC: Description text style.
            DESC_VALUE: Description value text style.
            BOOL_TRUE: Boolean true depiction color.
            BOOL_FALSE: Boolean false depiction color.

        Text Attributes:
            ENDC: Default reset to end all attributes.
            BOLD: Bold text style.
            DIM: Dimmed text style.
            UNDERLINE: Underlined text style.
            BLINK: Blinking text style.
            INVERTED: Inverted (reverse) text style.

        Reset Attributes:
            RESET: Reset all formatting and styles.
            RESET_BOLD: Reset bold text style.
            RESET_DIM: Reset dimmed text style.
            RESET_UNDERLINED: Reset underlined text style.
            RESET_BLINK: Reset blinking text style.
            RESET_REVERSE: Reset reverse text style.

    Methods:
        clear():
            Clears the console or terminal screen based on the underlying operating
            system. Handles both Windows and UNIX-based systems.

    Raises:
        None
    """
    def __init__(self):
        """

        This class provides a collection of ANSI escape codes for manipulating the appearance of terminal text, such as colors,
        boldness, underlining, and other formatting options. It defines foreground and background colors, as well as attributes
        for various formatting effects. These predefined codes can be used to apply consistent styles to terminal output easily.

        Attributes:
            Default_f: str
                The ANSI code for the default foreground color.
            Black_f: str
                The ANSI code for black foreground color.
            Red_f: str
                The ANSI code for red foreground color.
            Green_f: str
                The ANSI code for green foreground color.
            Yellow_f: str
                The ANSI code for yellow foreground color.
            Blue_f: str
                The ANSI code for blue foreground color.
            Magenta_f: str
                The ANSI code for magenta foreground color.
            Cyan_f: str
                The ANSI code for cyan foreground color.
            Light_Gray_f: str
                The ANSI code for light gray foreground color.
            Dark_Gray_f: str
                The ANSI code for dark gray foreground color.
            Light_Red_f: str
                The ANSI code for light red foreground color.
            Light_Green_f: str
                The ANSI code for light green foreground color.
            Light_Yellow_f: str
                The ANSI code for light yellow foreground color.
            Light_Blue_f: str
                The ANSI code for light blue foreground color.
            Light_Magenta_f: str
                The ANSI code for light magenta foreground color.
            Light_Cyan_f: str
                The ANSI code for light cyan foreground color.
            White_f: str
                The ANSI code for white foreground color.
            Default_b: str
                The ANSI code for the default background color.
            Black_b: str
                The ANSI code for black background color.
            Red_b: str
                The ANSI code for red background color.
            Green_b: str
                The ANSI code for green background color.
            Yellow_b: str
                The ANSI code for yellow background color.
            Blue_b: str
                The ANSI code for blue background color.
            Magenta_b: str
                The ANSI code for magenta background color.
            Cyan_b: str
                The ANSI code for cyan background color.
            Light_Gray_b: str
                The ANSI code for light gray background color.
            Dark_Gray_b: str
                The ANSI code for dark gray background color.
            Light_Red_b: str
                The ANSI code for light red background color.
            Light_Green_b: str
                The ANSI code for light green background color.
            Light_Yellow_b: str
                The ANSI code for light yellow background color.
            Light_Blue_b: str
                The ANSI code for light blue background color.
            Light_Magenta_b: str
                The ANSI code for light magenta background color.
            Light_Cyan_b: str
                The ANSI code for light cyan background color.
            White_b: str
                The ANSI code for white background color.
            HEADER: str
                The ANSI code for header text styling, typically magenta.
            OKBLUE: str
                The ANSI code for blue success or information styling, typically light blue.
            OKGREEN: str
                The ANSI code for green success or confirmation styling, typically light green.
            WARNING: str
                The ANSI code for warning or alert styling, typically light yellow.
            FAIL: str
                The ANSI code for failure or error styling, typically red.
            DESC: str
                The ANSI code for descriptive text styling, typically magenta.
            DESC_VALUE: str
                The ANSI code for descriptive value styling, typically light cyan.
            BOOL_TRUE: str
                The ANSI code for representing a boolean 'True' value, typically green.
            BOOL_FALSE: str
                The ANSI code for representing a boolean 'False' value, typically light yellow.
            ENDC: str
                The ANSI code to reset all attributes to their defaults.
            BOLD: str
                The ANSI code for bold text styling.
            DIM: str
                The ANSI code for dim text styling.
            UNDERLINE: str
                The ANSI code to underline text.
            BLINK: str
                The ANSI code for blinking text styling.
            INVERTED: str
                The ANSI code to invert foreground and background colors.
            RESET: str
                The ANSI code to reset all styles to default.
            RESET_BOLD: str
                The ANSI code to reset bold text attribute.
            RESET_DIM: str
                The ANSI code to reset dim text attribute.
            RESET_UNDERLINED: str
                The ANSI code to reset underline text attribute.
            RESET_BLINK: str
                The ANSI code to reset blinking text attribute.
            RESET_REVERSE: str
                The ANSI code to reset inverted text attribute.
        """
        # Foreground Colors
        self.Default_f = '\x1B[39m'
        self.Black_f = '\x1B[30m'
        self.Red_f = '\x1B[31m'
        self.Green_f = '\x1B[32m'
        self.Yellow_f = '\x1B[33m'
        self.Blue_f = '\x1B[34m'
        self.Magenta_f = '\x1B[35m'
        self.Cyan_f = '\x1B[36m'
        self.Light_Gray_f = '\x1B[37m'
        self.Dark_Gray_f  = '\x1B[90m'
        self.Light_Red_f = '\x1B[91m'
        self.Light_Green_f = '\x1B[92m'
        self.Light_Yellow_f = '\x1B[93m'
        self.Light_Blue_f = '\x1B[94m'
        self.Light_Magenta_f = '\x1B[95m'
        self.Light_Cyan_f = '\x1B[96m'
        self.White_f = '\x1B[97m'

        # Background Colors
        self.Default_b = '\x1B[49m'
        self.Black_b = '\x1B[40m'
        self.Red_b = '\x1B[41m'
        self.Green_b = '\x1B[42m'
        self.Yellow_b = '\x1B[43m'
        self.Blue_b = '\x1B[44m'
        self.Magenta_b = '\x1B[45m'
        self.Cyan_b = '\x1B[46m'
        self.Light_Gray_b = '\x1B[47m'
        self.Dark_Gray_b = '\x1B[100m'
        self.Light_Red_b = '\x1B[101m'
        self.Light_Green_b = '\x1B[102m'
        self.Light_Yellow_b = '\x1B[103m'
        self.Light_Blue_b = '\x1B[104m'
        self.Light_Magenta_b = '\x1B[105m'
        self.Light_Cyan_b = '\x1B[106m'
        self.White_b = '\x1B[107m'

        self.HEADER = self.Magenta_f
        self.OKBLUE = self.Light_Blue_f
        self.OKGREEN = self.Light_Green_f
        self.WARNING = self.Light_Yellow_f
        self.FAIL = self.Red_f

        self.DESC = self.Magenta_f
        self.DESC_VALUE = self.Light_Cyan_f
        self.BOOL_TRUE = self.Green_f
        self.BOOL_FALSE = self.Light_Yellow_f

        # Attributes
        self.ENDC = '\x1B[0m'
        self.BOLD = '\x1B[1m'
        self.DIM = '\x1B[2m'
        self.UNDERLINE = '\x1B[4m'
        self.BLINK = '\x1B[5m'
        self.INVERTED = '\x1B[7m'

        # Reset Attributes
        self.RESET = '\x1B[0m'
        self.RESET_BOLD = '\x1B[21m'
        self.RESET_DIM = '\x1B[22m'
        self.RESET_UNDERLINED = '\x1B[24m'
        self.RESET_BLINK = '\x1B[25m'
        self.RESET_REVERSE = '\x1B[27m'

    @staticmethod
    def clear():
        """
        Clears the terminal screen based on the operating system.

        This static method determines the operating system in use and executes the
        appropriate command to clear the terminal screen. It supports Windows and
        Unix-based operating systems.

        @return: None
        """
        os.system('cls' if os.name == 'nt' else 'clear')
