#!/bin/bash

# Define arrays of examples
english_queries=(
  "List cities with the highest population."
  "List students who never took any courses."
  "Show the census ranking of cities whose status are not 'Village'."
)

scratch_outputs=(
  "SELECT building FROM department ORDER BY budget DESC LIMIT 1;"
  "SELECT Name FROM member WHERE Member_ID NOT IN (SELECT Member_ID FROM member_attendance);"
  "SELECTbuying_low_low_temperature) FROM weekly_weather;"
)

so13m_outputs=(
  "SELECT Name FROM city ORDER BY Population DESC LIMIT 1;"
  "SELECT id FROM Student EXCEPT SELECT StuID FROM Sportsinfo;"
  "SELECT cName FROM tryout WHERE pPos != 'Village';"
)

# Write each example to results_example.txt
for ((i=0; i<${#english_queries[@]}; i++)); do
  echo -e "\n\nEnglish: \"${english_queries[$i]}\"\n" >> results_example.txt
  echo -e "Jam-CGPT (scratch):\n    ${scratch_outputs[$i]}\n" >> results_example.txt
  echo -e " Jam-CGPT (SO13M):\n    ${so13m_outputs[$i]}" >> results_example.txt
done
