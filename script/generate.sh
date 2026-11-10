#!/bin/bash
#

#python generate.py \
#  --title "Course details" \
#  --dst ../details \
#  --source "/Users/peter/Nextcloud/onderwijs/Machine Learning/Lectures/01 Course details/CourseDetails.generic.key" \
#  --base-url "https://dlvu.github.io/details" \
#  --pdf-link "https://dlvu.github.io/lectures/Course_details.2024.pdf"

#python generate.py \
#    --title "Convolutions" \
#    --dst ../cnns \
#    --source /Users/peter/Nextcloud/onderwijs/Deep\ Learning/cNNs/lecture03.convolutions.key \
#    --base-url "https://dlvu.github.io/cnns" \
#    --pdf-link "https://dlvu.github.io/pdfs/lecture03.convolutions.pdf" \

python generate.py \
    --title "Tools of the Trade" \
    --dst ../tools \
    --source '/Users/peter/Nextcloud/onderwijs/Deep Learning/Tools/lecture04.tools.key' \
    --base-url "https://dlvu.github.io/tools" \
    --pdf-link "https://dlvu.github.io/pdfs/lecture04.tools.annotated.pdf" \

cd ..
git add style.css index.md script/ mlvu.script.js course-details cnns cnns-old pdfs tools
git commit -m "Auto-update."
git push

