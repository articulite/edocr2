"""The recognizers' alphabets after retraining.

The released dimension recognizer reads only 0-9 A a B C D R G H h M m n x Z t ( ) , . + - ± : / ° " ⌀ =,
so THRU, MIN, MAX, most fit letters, uppercase X and the hole symbols could not be read at all.
Added: the rest of the uppercase letters; the lowercase ISO 286 shaft letters; depth,
counterbore, countersink and square; the multiplication sign; brackets; minutes; % # *.
The old characters keep their order, so their trained weights carry over.
"""

DIMENSIONS_OLD = "0123456789AaBCDRGHhMmnxZt(),.+-±:/°\"⌀="
DIMENSIONS = DIMENSIONS_OLD + "EFIJKLNOPQSTUVWXYbcdefgjkprsuvyz↧⌴⌵□×[]'%#*"
GDT = "0123456789,.⌀ABCD⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰ⒺⒻⓁⓂⓅⓈⓉⓊ"

# What a text box may contain for the pipeline to treat it as a dimension (Pipeline.
# dimension_criteria). Kept apart from what the recognizer can write: routing by the widened
# alphabet sent title blocks and notes to the recognizer (MASSSTAB 1:2 came back a
# "dimension"). The original alphabet, plus only what hole notes need.
ROUTING = DIMENSIONS_OLD + "XTUIN↧⌴⌵□×"
