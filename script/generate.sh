#!/bin/bash
#

#python generate.py \
#  --title "Course details" \
#  --dst ../details \
#  --source "/Users/peter/Nextcloud/onderwijs/Machine Learning/Lectures/01 Course details/CourseDetails.generic.key" \
#  --base-url "https://mlvu.github.io/details" \
#  --pdf-link "https://mlvu.github.io/lectures/Course_details.2024.pdf"

cd ..
git add style.css index.md script/ mlvu.script.js course-details
git commit -m "Auto-update."
git push

